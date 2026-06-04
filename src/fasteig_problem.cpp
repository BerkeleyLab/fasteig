#include "fasteig_problem.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

// ------------------------
// Constructors
// ------------------------

fasteig_problem::fasteig_problem(SpMat H, std::string solver)
    : H_(std::move(H)), solver_(std::move(solver)) {
  if (!H_) throw std::invalid_argument("fasteig_problem: null matrix.");
  if (H_->rows() != H_->cols()) throw std::invalid_argument("fasteig_problem: matrix must be square.");
  n_ = static_cast<FASTEIG_INT>(H_->rows());
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
    build_default_ops_from_H(opsA_);
    if (is_sai_which(p.which)) {
      build_sai_ops_from_H(p);
    }
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
    // Defer factorization of B to prepare(opts) so we can time/diagnose it and guarantee it's done once.
  }
}

// ------------------------
// Public API
// ------------------------

void fasteig_problem::prepare(const fasteig_options& opts) {
  if (!B_) return;
  if (b_factorized_) return;

  const std::string solver_name = opts.solver().empty() ? solver_ : opts.solver();
  if (solver_name.rfind("KrylovSchur", 0) != 0) {
    // Currently, generalized eigenproblems are only supported by KrylovSchur.
    return;
  }

  auto print_line = [&](const std::string& s) {
    if (log_stream_) (*log_stream_) << s << "\n";
    if (opts.verbosity() >= 1) std::cout << s << "\n";
  };

  auto print_kv = [&](const char* key, const std::string& value) {
    constexpr int kKeyWidth = 22;
    std::ostringstream ss;
    ss << "  " << std::left << std::setw(kKeyWidth) << key << ": " << value;
    print_line(ss.str());
  };

  print_line("");
  print_line("GENERALIZED SETUP");
  print_line("=================");
  print_line("");
  {
    std::string backend;
#ifdef EIGEN_USE_MKL_ALL
    backend = "MKL PardisoLDLT";
#else
    backend = "Eigen SimplicialLDLT";
#endif
    print_kv("Factorization Backend", backend);
  }
  print_line("  Performing B factorization (LDLT)...");

  const auto t0 = std::chrono::steady_clock::now();
  ldltB_.analyzePattern(*B_);
  ldltB_.factorize(*B_);
  const auto t1 = std::chrono::steady_clock::now();
  const double sec = std::chrono::duration<double>(t1 - t0).count();

  if (ldltB_.info() != Eigen::Success) {
    throw std::runtime_error("prepare(): B factorization failed (LDLT info != Success).");
  }

  opsB_.solve = [this](const VectorXd& v, VectorXd& x) { x = ldltB_.solve(v); };
  opsB_.block_solve = [this](const MatrixXd& V, MatrixXd& X) { X = ldltB_.solve(V); };

  b_factorized_ = true;
  b_factorization_sec_ = sec;

  {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6) << sec;
    print_kv("B Factorization (sec)", ss.str());
  }

  print_line("");
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

FASTEIG_INT fasteig_problem::solve(const fasteig_options& opts) {
  opts.validate();
  last_iterations_.reset();

  prepare(opts);

  const FASTEIG_INT nev = opts.nev();
  const FASTEIG_INT n = n_;

  fasteig_options local = opts;
  local.ensure_initial_vectors(n, nev);
  const operations& active_ops = solver_ops(local);

  eigvals_.resize(static_cast<int>(nev));
  resnrms_.resize(static_cast<int>(nev));
  eigvecs_ = local.v0();

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

  if (B_ && solver_name.rfind("KrylovSchur", 0) != 0) {
    throw std::invalid_argument("Generalized eigenproblems (bmatrix) are only supported by KrylovSchur for now.");
  }

  auto solver = fasteig::make_fasteig_solver(solver_name, *this, local);
  if (!solver) {
    if (solver_name.rfind("auto", 0) == 0) {
      status_ = 0;
    } else {
      status_ = -1;
    }
    return status_;
  }
  status_ = solver->solve();
  last_iterations_ = solver->iterations();
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

bool fasteig_problem::is_sai_which(const std::string& which) {
  std::string lower = which;
  std::transform(lower.begin(), lower.end(), lower.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return lower == "sai";
}

void fasteig_problem::build_sai_ops_from_H(const fasteig_params& p) {
  if (!H_) throw std::logic_error("build_sai_ops_from_H: no matrix stored.");

  if (p.shift != 0.0) {
    H_ ->diagonal().array() -= p.shift;
  }

  ldlt_.analyzePattern(*H_);
  ldlt_.factorize(*H_);

  opsSAI_.matvec = [this](const VectorXd& v, VectorXd& av) {
    av = ldlt_.solve(v);
  };
  opsSAI_.block_matvec = [this](const MatrixXd& V, MatrixXd& AV) {
    AV = ldlt_.solve(V);
  };

  // add the shift back to the diagonal for reporting Ritz values in the original scale (optional, but often desired for user-facing outputs)
  if (p.shift != 0.0) {
    H_ ->diagonal().array() += p.shift;
  }
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

