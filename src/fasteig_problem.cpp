#include "fasteig_problem.hpp"

#include "core/CsrMatrix.hpp"
#include "core/Executor.hpp"
#include "core/LinearOperator.hpp"
#include "core/LegacyOperationsOperator.hpp"
#if defined(USE_CUDA) && defined(USE_CUDSS)
#include "gpu/CudaShiftInvertOperator.hpp"
#endif

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

fasteig::core::CsrMatrixData make_core_csr_data(const fasteig_problem::CsrMatrixData& src,
                                                FASTEIG_INT n) {
  fasteig::core::CsrMatrixData out;
  out.rows = n;
  out.cols = n;
  out.rowptr = src.rowptr;
  out.colind = src.colind;
  out.values = src.values;
  return out;
}

SpMat sparse_from_csr(const fasteig_problem::CsrMatrixData& data, FASTEIG_INT n) {
  using RowSparse = Eigen::SparseMatrix<double, Eigen::RowMajor>;
  return Eigen::Map<const RowSparse>(n, n, static_cast<int>(data.values.size()),
                                     data.rowptr.data(), data.colind.data(), data.values.data());
}

bool is_krylov_solver(const std::string& name) {
  return name.rfind("Lanczos", 0) == 0 || name.rfind("BlockLanczos", 0) == 0 ||
         name.rfind("KrylovSchur", 0) == 0 || name.rfind("BlockKrylovSchur", 0) == 0;
}

void validate_generalized_solver(const std::string& name, bool use_sai) {
  if (!is_krylov_solver(name)) {
    throw std::invalid_argument("Generalized problems require Lanczos, BlockLanczos, KrylovSchur or BlockKrylovSchur.");
  }
  if (use_sai && name.rfind("KrylovSchur", 0) != 0) {
    throw std::invalid_argument("Generalized SAI is currently supported only by KrylovSchur.");
  }
}

void require_symmetric(const SpMat& matrix, const char* name) {
  const SpMat difference = matrix - SpMat(matrix.transpose());
  if (!std::isfinite(matrix.norm()) ||
      difference.norm() > 1e-12 * std::max(1.0, matrix.norm())) {
    throw std::invalid_argument(std::string("Generalized Krylov solvers require symmetric ") + name + ".");
  }
}

}  // namespace

// ------------------------
// Constructors
// ------------------------

fasteig_problem::fasteig_problem(SpMat H, std::string solver)
    : H_(std::move(H)), solver_(std::move(solver)) {
  if (!H_) throw std::invalid_argument("fasteig_problem: null matrix.");
  if (H_->rows() != H_->cols()) throw std::invalid_argument("fasteig_problem: matrix must be square.");
  n_ = static_cast<FASTEIG_INT>(H_->rows());
  build_csr_from_H();
  build_default_ops_from_H(opsA_);
}

fasteig_problem::fasteig_problem(operations ops, FASTEIG_INT n, std::string solver)
    : opsA_(std::move(ops)), n_(n), solver_(std::move(solver)) {
  if (n_ <= 0) throw std::invalid_argument("fasteig_problem: n must be > 0 for matrix-free ops.");
}

fasteig_problem::fasteig_problem(const fasteig_params& p)
    : params_(std::cref(p)), solver_(p.solver) {
  if (p.n <= 0) throw std::invalid_argument("fasteig_problem: params.n must be > 0.");
  n_ = p.n;

  if (p.matrixformat == MatrixFormat::MTX) {
    H_ = build_sparse_matrix_from_triplets(p.n, p.annz, p.arowind, p.acolind, p.anzvals);
    if (!p.bmatfname.empty()) {
      B_ = build_sparse_matrix_from_triplets(p.n, p.bnnz, p.browind, p.bcolind, p.bnzvals);
    }
  } else if (p.matrixformat == MatrixFormat::RSA || p.matrixformat == MatrixFormat::FEI) {
    H_ = build_sparse_matrix_from_RSA(p.n, p.annz, p.acolptr, p.arowind, p.anzvals);
    if (!p.bmatfname.empty()) {
      B_ = build_sparse_matrix_from_RSA(p.n, p.bnnz, p.bcolptr, p.browind, p.bnzvals);
    }
  }

  if (H_) {
    build_csr_from_H();
    build_default_ops_from_H(opsA_);
  }

  if (B_) {
    // Normalize storage for symmetric MatrixMarket B: store the lower triangle so LDLT<Lower> sees the matrix.
    // If the file stored the upper triangle, transpose the stored triangle (same matrix, different storage).
    if (p.matrixformat == MatrixFormat::MTX && p.b_mtx_symmetric && p.b_mtx_triangle_upper) {
      SpMat Bt = B_->transpose();
      *B_ = std::move(Bt);
      b_triangle_upper_for_ops_ = false;
    } else {
      b_triangle_upper_for_ops_ = p.b_mtx_triangle_upper;
    }

    build_default_ops_from_B(opsB_);
    csrB_ = build_csr_from_sparse_matrix(*B_,
        p.matrixformat == MatrixFormat::MTX && p.b_mtx_symmetric,
        p.matrixformat == MatrixFormat::MTX && p.b_mtx_skew_symmetric);
    // Defer factorization of B to prepare(opts) so we can time/diagnose it and guarantee it's done once.
  }
}

// ------------------------
// Public API
// ------------------------

void fasteig_problem::prepare(const fasteig_options& opts) {
  const bool use_sai = is_sai_which(opts.which());
  if (!use_sai && !B_) return;
  if (use_sai && !std::isfinite(opts.shift())) {
    throw std::invalid_argument("prepare(): shift must be finite.");
  }
  if (!H_ || !csrA_) throw std::invalid_argument("prepare(): factorization requires a stored matrix.");
  if (B_) {
    const std::string solver_name = opts.solver().empty() ? solver_ : opts.solver();
    validate_generalized_solver(solver_name.empty() ? "Lanczos" : solver_name, use_sai);
    require_symmetric(sparse_from_csr(*csrA_, n_), "A");
    require_symmetric(sparse_from_csr(*csrB_, n_), "B");
  }
  // The CUDA inverse operator owns its cuDSS factors. Do not also factor on CPU.
  if (executor(opts)->is_gpu()) return;
  if (use_sai ? (sai_factorized_ && sai_shift_ == opts.shift()) : b_factorized_) return;
  // A failed refactorization must not leave a previous shift marked reusable.
  if (use_sai) sai_factorized_ = false;
  else b_factorized_ = false;

  const auto setup_t0 = std::chrono::steady_clock::now();
  SpMat factor_matrix = sparse_from_csr(use_sai ? *csrA_ : *csrB_, n_);
  if (use_sai && B_) {
    // Sparse subtraction forms the union of both patterns, including diagonals
    // present only in B. The input MatrixMarket triangles have been expanded.
    factor_matrix -= opts.shift() * sparse_from_csr(*csrB_, n_);
  } else if (use_sai && opts.shift() != 0.0) {
    for (FASTEIG_INT i = 0; i < n_; ++i) factor_matrix.coeffRef(i, i) -= opts.shift();
  }
  factor_matrix.makeCompressed();
  auto& factor = use_sai ? ldlt_ : ldltB_;
  const auto analysis_t0 = std::chrono::steady_clock::now();
  factor.analyzePattern(factor_matrix);
  if (factor.info() != Eigen::Success) {
    throw std::runtime_error("prepare(): sparse factorization analysis failed.");
  }
  const auto analysis_t1 = std::chrono::steady_clock::now();
  factor.factorize(factor_matrix);
  const auto factor_t1 = std::chrono::steady_clock::now();
  if (factor.info() != Eigen::Success) {
    throw std::runtime_error("prepare(): factorization failed; check B or whether the shift is an eigenvalue.");
  }
#ifndef EIGEN_USE_MKL_ALL
  if (!factor.vectorD().allFinite()) {
    throw std::runtime_error("prepare(): factorization produced non-finite pivots.");
  }
  if (!use_sai && (factor.vectorD().array() <= 0.0).any()) {
    throw std::invalid_argument("prepare(): generalized Krylov solvers require positive-definite B.");
  }
#endif
  const double analysis_sec = std::chrono::duration<double>(analysis_t1 - analysis_t0).count();
  const double factor_sec = std::chrono::duration<double>(factor_t1 - analysis_t1).count();
  const double setup_sec = std::chrono::duration<double>(factor_t1 - setup_t0).count();
  if (use_sai) {
    opsSAI_.matvec = [this](const VectorXd& v, VectorXd& y) { y = ldlt_.solve(v); };
    opsSAI_.block_matvec = [this](const MatrixXd& v, MatrixXd& y) { y = ldlt_.solve(v); };
    sai_factorized_ = true;
    sai_shift_ = opts.shift();
    sai_analysis_sec_ = analysis_sec;
    sai_factorization_sec_ = factor_sec;
    sai_setup_sec_ = setup_sec;
  } else {
    opsB_.solve = [this](const VectorXd& v, VectorXd& y) { y = ldltB_.solve(v); };
    opsB_.block_solve = [this](const MatrixXd& v, MatrixXd& y) { y = ldltB_.solve(v); };
    opsInverseB_.matvec = opsB_.solve;
    opsInverseB_.block_matvec = opsB_.block_solve;
    b_factorized_ = true;
    b_factorization_sec_ = factor_sec;
  }
#ifdef EIGEN_USE_MKL_ALL
  const std::string backend = "MKL PardisoLDLT";
#else
  const std::string backend = "Eigen SimplicialLDLT";
#endif
  const auto shift = use_sai ? std::optional<double>(opts.shift()) : std::nullopt;
  const std::string matrix = use_sai ? (B_ ? "A - shift*B" : "A - shift*I") : "B";
  logger_.print_factorization_setup(backend, matrix, shift, analysis_sec, factor_sec, setup_sec);
  if (opts.verbosity() >= 1) {
    fasteig_logger console(&std::cout);
    console.print_factorization_setup(backend, matrix, shift, analysis_sec, factor_sec, setup_sec);
  }
}

FASTEIG_INT fasteig_problem::solve() {
  fasteig_options opts(n_);
  if (params_) {
    opts.configure_from(params_->get());
  } else {
    opts.set("solver", solver_);
    opts.set("nev", "10");
    opts.set("maxiter", "1000");
    opts.set("tol", "1e-8");
    opts.set("which", "LM");
  }
  return solve(opts);
}

void fasteig_problem::open_log(const fasteig_options& opts,
                               const std::string& solver_name,
                               bool append) {
  const std::string name = solver_name.empty() ? opts.solver() : solver_name;
  const std::string path = opts.output_path(name);
  if (path.empty()) return;
  logger_.open(path, append);
}

void fasteig_problem::set_log_path(const std::string& path, bool append) {
  logger_.open(path, append);
}

std::shared_ptr<fasteig::core::Executor> fasteig_problem::executor(const fasteig_options& opts) {
  if (!executor_) {
    executor_ = fasteig::core::make_executor(opts.gpu_backend());
  }
  return executor_;
}

void fasteig_problem::set_executor(std::shared_ptr<fasteig::core::Executor> exec) {
  if (!exec) {
    throw std::invalid_argument("fasteig_problem::set_executor: null executor.");
  }
  prepared_operator_.reset();
  matrix_operator_.reset();
  metric_operator_.reset();
  executor_ = std::move(exec);
}

void fasteig_problem::prepare_linear_operator(const fasteig_options& opts,
                                              const fasteig::core::Executor& exec) {
  prepared_operator_ = build_linear_operator(opts, exec);
}

std::unique_ptr<fasteig::core::LinearOperator> fasteig_problem::make_linear_operator(
    const fasteig_options& opts,
    const fasteig::core::Executor& exec) {
  if (prepared_operator_) {
    return std::move(prepared_operator_);
  }
  return build_linear_operator(opts, exec);
}

std::shared_ptr<const fasteig::core::LinearOperator> fasteig_problem::make_matrix_operator(
    const fasteig::core::Executor& exec) const {
  if (!matrix_operator_ || &matrix_operator_->executor() != &exec) {
    if (exec.is_gpu()) {
      if (!csrA_) throw std::invalid_argument("CUDA matrix operator requires a stored CSR matrix.");
      matrix_operator_ = exec.create_csr_matrix(make_core_csr_data(*csrA_, n_));
    } else {
      matrix_operator_ = std::make_shared<fasteig::core::LegacyOperationsOperator>(exec, n_, opsA_);
    }
  }
  return matrix_operator_;
}

std::shared_ptr<const fasteig::core::LinearOperator> fasteig_problem::make_metric_operator(
    const fasteig::core::Executor& exec) const {
  if (!B_) return nullptr;
  if (!metric_operator_ || &metric_operator_->executor() != &exec) {
    if (exec.is_gpu()) metric_operator_ = exec.create_csr_matrix(make_core_csr_data(*csrB_, n_));
    else metric_operator_ = std::make_shared<fasteig::core::LegacyOperationsOperator>(exec, n_, opsB_);
  }
  return metric_operator_;
}

std::unique_ptr<fasteig::core::LinearOperator> fasteig_problem::build_linear_operator(
    const fasteig_options& opts,
    const fasteig::core::Executor& exec) {
  const bool use_sai = is_sai_which(opts.which());
  if (!use_sai && !B_) {
    // The owning problem keeps this original operator alive throughout the solve.
    return std::make_unique<fasteig::core::ScaledLinearOperator>(*make_matrix_operator(exec), 1.0);
  }

  std::unique_ptr<fasteig::core::LinearOperator> inverse;
  if (exec.is_gpu()) {
#if defined(USE_CUDA) && defined(USE_CUDSS)
    if (!csrA_) throw std::invalid_argument("CUDA factorization requires a stored CSR matrix.");
    auto factor_data = make_core_csr_data(use_sai ? *csrA_ : *csrB_, n_);
    double identity_shift = use_sai ? opts.shift() : 0.0;
    if (use_sai && B_) {
      SpMat shifted = sparse_from_csr(*csrA_, n_) - opts.shift() * sparse_from_csr(*csrB_, n_);
      factor_data = make_core_csr_data(build_csr_from_sparse_matrix(shifted, false, false), n_);
      identity_shift = 0.0;  // A - shift*B has already been assembled.
    }
    // Reuse the established general/full cuDSS factorization, including for
    // indefinite shifted matrices. Only solve phases occur in apply().
    auto cuda_inverse = std::make_unique<fasteig::core::CudaShiftInvertOperator>(
        exec, std::move(factor_data), identity_shift, false);
    const auto& t = cuda_inverse->solver().timings();
    const auto shift = use_sai ? std::optional<double>(opts.shift()) : std::nullopt;
    const std::string matrix = use_sai ? (B_ ? "A - shift*B" : "A - shift*I") : "B";
    logger_.print_factorization_setup("NVIDIA cuDSS (General/Full)", matrix, shift,
                                      t.analysis_sec, t.factorization_sec, t.setup_sec, t.matrix_setup_sec);
    if (opts.verbosity() >= 1) {
      fasteig_logger console(&std::cout);
      console.print_factorization_setup("NVIDIA cuDSS (General/Full)", matrix, shift,
                                        t.analysis_sec, t.factorization_sec, t.setup_sec, t.matrix_setup_sec);
    }
    if (use_sai) sai_factorization_sec_ = t.factorization_sec;
    else b_factorization_sec_ = t.factorization_sec;
    inverse = std::move(cuda_inverse);
#else
    throw std::invalid_argument("CUDA SAI/generalized problems require USE_CUDA=1 USE_CUDSS=1.");
#endif
  } else {
    inverse = std::make_unique<fasteig::core::LegacyOperationsOperator>(
        exec, n_, use_sai ? opsSAI_ : opsInverseB_);
  }
  if (!B_) return inverse;
  // Ordinary: B^{-1} A. Shift-invert: (A - shift*B)^{-1} B.
  return std::make_unique<fasteig::core::ProductLinearOperator>(
      std::move(inverse), use_sai ? make_metric_operator(exec) : make_matrix_operator(exec));
}

FASTEIG_INT fasteig_problem::solve(const fasteig_options& opts) {
  opts.validate();
  last_iterations_.reset();
  prepared_operator_.reset();

  const FASTEIG_INT nev = opts.nev();
  const FASTEIG_INT n = n_;

  fasteig_options local = opts;
  const operations& active_ops = solver_ops(local);

  eigvals_.resize(static_cast<int>(nev));
  resnrms_.resize(static_cast<int>(nev));
  eigvecs_.resize(static_cast<int>(n), static_cast<int>(nev));
  eigvecs_.setZero();

  std::string solver_name = local.solver().empty() ? solver_ : local.solver();
  if (solver_name.empty()) solver_name = "Lanczos";

  if (solver_name.rfind("SPPC", 0) == 0) {
    ns_ = local.maxdim();
    if (ns_ <= 0) throw std::invalid_argument("SPPC requires maxdim to be set to ns.");
    if (active_ops.H0.rows() == 0 || active_ops.H0.rows() != ns_) {
      set_sppc_split(ns_);
    }
  }

  solver_ = solver_name;

  if (B_) validate_generalized_solver(solver_name, is_sai_which(local.which()));

  const bool caller_provided_log = logger_.enabled();
  if (!caller_provided_log) {
    open_log(local, solver_name);
  }

  if (logger_.enabled()) {
    logger_.print_matrix_summary(*this);
    logger_.print_run_header(local, solver_name);
  }

  const int verbosity = local.verbosity();
  fasteig_logger stdout_logger;
  if (verbosity >= 1) {
    stdout_logger.attach(&std::cout);
    stdout_logger.print_matrix_summary(*this);
    stdout_logger.print_run_header(local, solver_name);
  }

  const auto t0 = std::chrono::steady_clock::now();

  prepare(local);
  auto exec = executor(local);
  if (is_sai_which(local.which()) || B_) {
    prepare_linear_operator(local, *exec);
  }

  if (logger_.enabled()) {
    logger_.section("PROGRESS");
  }
  if (verbosity >= 2) {
    stdout_logger.section("PROGRESS");
  }

  auto solver = fasteig::make_fasteig_solver(solver_name, *this, local);
  if (!solver) {
    if (solver_name.rfind("auto", 0) == 0) {
      status_ = 0;
    } else {
      status_ = -1;
    }
    if (!caller_provided_log) close_log();
    return status_;
  }
  status_ = solver->solve();
  const auto t1 = std::chrono::steady_clock::now();
  const double runtime_sec = std::chrono::duration<double>(t1 - t0).count();
  last_iterations_ = solver->iterations();

  if (!is_krylov_solver(solver_name)) {
    resnrms_ = fasteig::fasteig_solver::compute_residual_norms(
        opsA_, B_ ? &opsB_ : nullptr, eigvals_, eigvecs_);
  }

  const auto b_setup = is_sai_which(local.which()) ? std::nullopt : b_factorization_sec_;
  if (logger_.enabled()) {
    logger_.write_results_text(status_, eigvals_, resnrms_,
                               b_setup, runtime_sec, last_iterations_);
    logger_.flush();
  }
  if (verbosity >= 1) {
    stdout_logger.print_final_results(status_, eigvals_, resnrms_,
                                      b_setup, runtime_sec, last_iterations_);
  }

  if (!caller_provided_log) close_log();
  return status_;
}

void fasteig_problem::set_sppc_split(FASTEIG_INT ns) {
  if (!H_) throw std::logic_error("set_sppc_split: no matrix stored.");
  const int n = H_->rows();
  if (ns <= 0 || ns > n) throw std::invalid_argument("set_sppc_split: invalid ns.");

  set_sppc_split(opsA_, ns);
  if (opsSAI_.matvec || opsSAI_.block_matvec) {
    set_sppc_split(opsSAI_, ns);
  }
}

operations& fasteig_problem::solver_ops(const fasteig_options& opts) noexcept {
  return const_cast<operations&>(std::as_const(*this).solver_ops(opts));
}

const operations& fasteig_problem::solver_ops(const fasteig_options& opts) const noexcept {
  const bool use_sai_ops = is_sai_which(opts.which()) && static_cast<bool>(opsSAI_.matvec);
  return use_sai_ops ? opsSAI_ : opsA_;
}

void fasteig_problem::set_sppc_split(operations& target_ops, FASTEIG_INT ns) {
  const int n = H_->rows();

  target_ops.H0 = H_->topLeftCorner(static_cast<int>(ns), static_cast<int>(ns));
  target_ops.H0.makeCompressed();

  target_ops.V21 = H_->middleRows(static_cast<int>(ns), n - static_cast<int>(ns))
                       .leftCols(static_cast<int>(ns));
  target_ops.V21.makeCompressed();
}

// ------------------------
// Private helpers
// ------------------------

SpMat fasteig_problem::build_sparse_matrix_from_triplets(
    const FASTEIG_INT n,
    const FASTEIG_INT nnz,
    const std::vector<FASTEIG_INT>& rowind,
    const std::vector<FASTEIG_INT>& colind,
    const std::vector<FASTEIG_DOUBLE>& nzvals) {
  if (n <= 0) throw std::runtime_error("Invalid matrix dimension n <= 0.");
  if (nnz < 0) throw std::runtime_error("Invalid nnz < 0.");

  if (static_cast<std::size_t>(nnz) != rowind.size() ||
      static_cast<std::size_t>(nnz) != colind.size() ||
      static_cast<std::size_t>(nnz) != nzvals.size()) {
    throw std::runtime_error("Triplet arrays inconsistent with nnz.");
  }

  std::vector<Eigen::Triplet<FASTEIG_DOUBLE>> triplets;
  triplets.reserve(static_cast<std::size_t>(nnz));

  for (FASTEIG_INT k = 0; k < nnz; ++k) {
    const auto i = rowind[static_cast<std::size_t>(k)];
    const auto j = colind[static_cast<std::size_t>(k)];
    if (i < 0 || j < 0 || i >= n || j >= n) {
      throw std::runtime_error("Triplet index out of bounds.");
    }
    triplets.emplace_back(static_cast<int>(i), static_cast<int>(j),
                          nzvals[static_cast<std::size_t>(k)]);
  }

  SpMat H(static_cast<int>(n), static_cast<int>(n));
  H.setFromTriplets(triplets.begin(), triplets.end(),
                    [](FASTEIG_DOUBLE a, FASTEIG_DOUBLE b) { return a + b; });
  H.makeCompressed();
  return H;
}

SpMat fasteig_problem::build_sparse_matrix_from_RSA(
    const FASTEIG_INT n,
    const FASTEIG_INT nnz,
    const std::vector<FASTEIG_INT>& colptr,
    const std::vector<FASTEIG_INT>& rowind,
    const std::vector<FASTEIG_DOUBLE>& nzvals) {

  if (n <= 0) throw std::runtime_error("Invalid matrix dimension n <= 0.");
  if (static_cast<std::size_t>(nnz) != rowind.size() ||
      static_cast<std::size_t>(nnz) != nzvals.size()) {
    throw std::runtime_error("RSA arrays inconsistent with nnz.");
  }
  if (static_cast<int>(colptr.size()) != n + 1) {
    throw std::runtime_error("colptr must have size n+1.");
  }
  if (colptr.front() != 0 || colptr.back() != nnz) {
    throw std::runtime_error("colptr must start at 0 and end at nnz.");
  }
  if (!std::is_sorted(colptr.begin(), colptr.end())) {
    throw std::runtime_error("colptr must be nondecreasing.");
  }

  // Robust: build via triplets (handles unsorted row indices and duplicates)
  std::vector<Eigen::Triplet<FASTEIG_DOUBLE>> triplets;
  triplets.reserve(static_cast<std::size_t>(nnz));

  for (FASTEIG_INT j = 0; j < n; ++j) {
    const FASTEIG_INT start = colptr[static_cast<std::size_t>(j)];
    const FASTEIG_INT end   = colptr[static_cast<std::size_t>(j + 1)];
    if (start < 0 || end < start || end > nnz) throw std::runtime_error("Invalid colptr range.");
    for (FASTEIG_INT k = start; k < end; ++k) {
      const FASTEIG_INT i = rowind[static_cast<std::size_t>(k)];
      if (i < 0 || i >= n) throw std::runtime_error("row index out of range.");
      triplets.emplace_back(static_cast<int>(i), static_cast<int>(j),
                            nzvals[static_cast<std::size_t>(k)]);
    }
  }

  SpMat A(static_cast<int>(n), static_cast<int>(n));
  A.setFromTriplets(triplets.begin(), triplets.end(),
                    [](FASTEIG_DOUBLE a, FASTEIG_DOUBLE b) { return a + b; });
  A.makeCompressed();
  return A;
}

fasteig_problem::CsrMatrixData fasteig_problem::build_csr_from_sparse_matrix(
    const SpMat& A,
    const bool expand_symmetric,
    const bool expand_skew) {
  if (A.rows() != A.cols()) {
    throw std::runtime_error("CSR conversion requires a square matrix.");
  }
  if (A.rows() > std::numeric_limits<int>::max()) {
    throw std::runtime_error("CSR conversion requires int-sized dimensions.");
  }

  if (!expand_symmetric && !expand_skew) {
    Eigen::SparseMatrix<double, Eigen::RowMajor> row_major = A;
    row_major.makeCompressed();

    CsrMatrixData out;
    out.rowptr.assign(row_major.outerIndexPtr(),
                      row_major.outerIndexPtr() + row_major.rows() + 1);
    out.colind.assign(row_major.innerIndexPtr(),
                      row_major.innerIndexPtr() + row_major.nonZeros());
    out.values.assign(row_major.valuePtr(),
                      row_major.valuePtr() + row_major.nonZeros());
    return out;
  }

  std::vector<Eigen::Triplet<double>> triplets;
  triplets.reserve(static_cast<std::size_t>(A.nonZeros()) * 2);
  for (int outer = 0; outer < A.outerSize(); ++outer) {
    for (SpMat::InnerIterator it(A, outer); it; ++it) {
      const int row = it.row();
      const int col = it.col();
      const double value = it.value();
      triplets.emplace_back(row, col, value);
      if (row != col) {
        triplets.emplace_back(col, row, expand_skew ? -value : value);
      }
    }
  }

  Eigen::SparseMatrix<double, Eigen::RowMajor> full(A.rows(), A.cols());
  full.setFromTriplets(triplets.begin(), triplets.end(),
                       [](double a, double b) { return a + b; });
  full.makeCompressed();

  if (full.nonZeros() > std::numeric_limits<int>::max()) {
    throw std::runtime_error("CSR conversion produced too many nonzeros for int indexing.");
  }

  CsrMatrixData out;
  out.rowptr.assign(full.outerIndexPtr(), full.outerIndexPtr() + full.rows() + 1);
  out.colind.assign(full.innerIndexPtr(), full.innerIndexPtr() + full.nonZeros());
  out.values.assign(full.valuePtr(), full.valuePtr() + full.nonZeros());
  return out;
}

void fasteig_problem::build_csr_from_H() {
  if (!H_) {
    csrA_.reset();
    return;
  }

  const fasteig_params* p = params_ ? &params_->get() : nullptr;
  const bool expand_symmetric =
      p && p->matrixformat == MatrixFormat::MTX && p->mtx_symmetric;
  const bool expand_skew =
      p && p->matrixformat == MatrixFormat::MTX && p->mtx_skew_symmetric;

  csrA_ = build_csr_from_sparse_matrix(*H_, expand_symmetric, expand_skew);
}

bool fasteig_problem::is_sai_which(const std::string& which) {
  std::string lower = which;
  std::transform(lower.begin(), lower.end(), lower.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return lower == "sai";
}

void fasteig_problem::build_default_ops_from_H(operations& target_ops) {
  if (!H_) throw std::logic_error("build_default_ops_from_H: no matrix stored.");

  const fasteig_params* p = params_ ? &params_->get() : nullptr;
  const bool use_selfadjoint =
      (p && p->matrixformat == MatrixFormat::MTX && p->mtx_symmetric);
  const bool use_skew =
      (p && p->matrixformat == MatrixFormat::MTX && p->mtx_skew_symmetric);

  if (use_selfadjoint) {
    const bool upper = p->mtx_triangle_upper;
    target_ops.matvec = [this, upper](const VectorXd& v, VectorXd& av) {
      if (upper) {
        av.noalias() = H_->selfadjointView<Eigen::Upper>() * v;
      } else {
        av.noalias() = H_->selfadjointView<Eigen::Lower>() * v;
      }
    };
    target_ops.block_matvec = [this, upper](const MatrixXd& V, MatrixXd& AV) {
      if (upper) {
        AV.noalias() = H_->selfadjointView<Eigen::Upper>() * V;
      } else {
        AV.noalias() = H_->selfadjointView<Eigen::Lower>() * V;
      }
    };
  } else if (use_skew) {
    // Skew-symmetric: interpret the stored triangle as a full skew-symmetric matrix
    // A = T - T^T (diagonal assumed 0). This is a best-effort path; most solvers
    // in this project assume real-symmetric problems.
    target_ops.matvec = [this](const VectorXd& v, VectorXd& av) {
      av.setZero(v.size());
      // y += T*v
      av.noalias() += (*H_) * v;
      // y -= T^T*v, but avoid subtracting the diagonal twice (diag should be 0 anyway)
      av.noalias() -= H_->transpose() * v;
    };
    target_ops.block_matvec = [this](const MatrixXd& V, MatrixXd& AV) {
      AV.setZero(V.rows(), V.cols());
      AV.noalias() += (*H_) * V;
      AV.noalias() -= H_->transpose() * V;
    };
  } else {
    target_ops.matvec = [this](const VectorXd& v, VectorXd& av) {
      av.noalias() = (*H_) * v;
    };
    target_ops.block_matvec = [this](const MatrixXd& V, MatrixXd& AV) {
      AV.noalias() = (*H_) * V;
    };
  }

  minresSolver_.setTolerance(1e-6);
  minresSolver_.setMaxIterations(200);

  target_ops.minres = [this](const SpMat& A, const VectorXd& b, VectorXd& x) {
    minresSolver_.compute(A);
    x = minresSolver_.solve(b);
    if (minresSolver_.info() != Eigen::Success) {
      std::cerr << "WARNING: MINRES did not converge!\n";
    }
  };

  cg_.setTolerance(1e-8);
  cg_.setMaxIterations(3);

  target_ops.CG = [this](const VectorXd& v, const FASTEIG_DOUBLE& shift, VectorXd& w) {
    // Placeholder (yours was effectively (1-shift)I). Keep structure; adjust math as intended.
    const int N = static_cast<int>(v.size());
    SpMat A(N, N);
    A.setIdentity();
    A *= (1.0 - static_cast<double>(shift));
    cg_.compute(A);
    w = cg_.solve(v);
  };
}

void fasteig_problem::build_default_ops_from_B(operations& target_ops) {
  if (!B_) throw std::logic_error("build_default_ops_from_B: no B matrix stored.");

  const fasteig_params* p = params_ ? &params_->get() : nullptr;
  const bool use_selfadjoint =
      (p && p->matrixformat == MatrixFormat::MTX && p->b_mtx_symmetric);
  const bool use_skew =
      (p && p->matrixformat == MatrixFormat::MTX && p->b_mtx_skew_symmetric);

  if (use_skew) {
    throw std::invalid_argument("build_default_ops_from_B: skew-symmetric B is not supported.");
  }

  if (use_selfadjoint) {
    const bool upper = b_triangle_upper_for_ops_;
    target_ops.matvec = [this, upper](const VectorXd& v, VectorXd& av) {
      if (upper) {
        av.noalias() = B_->selfadjointView<Eigen::Upper>() * v;
      } else {
        av.noalias() = B_->selfadjointView<Eigen::Lower>() * v;
      }
    };
    target_ops.block_matvec = [this, upper](const MatrixXd& V, MatrixXd& AV) {
      if (upper) {
        AV.noalias() = B_->selfadjointView<Eigen::Upper>() * V;
      } else {
        AV.noalias() = B_->selfadjointView<Eigen::Lower>() * V;
      }
    };
  } else {
    target_ops.matvec = [this](const VectorXd& v, VectorXd& av) { av.noalias() = (*B_) * v; };
    target_ops.block_matvec = [this](const MatrixXd& V, MatrixXd& AV) { AV.noalias() = (*B_) * V; };
  }
}

