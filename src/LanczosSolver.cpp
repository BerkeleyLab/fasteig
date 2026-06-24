#include "LanczosSolver.hpp"

#include "KrylovSubspace.hpp"
#include "core/CsrMatrix.hpp"
#include "core/Executor.hpp"
#include "core/Operations.hpp"
#include "core/Vector.hpp"
#include "fasteig_logger.hpp"
#include "fasteig_problem.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fasteig {
namespace {

std::string lower_copy(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

void validate_which(const std::string& which) {
  if (which == "sr" || which == "lr" || which == "sm" ||
      which == "lm" || which == "be" || which == "sai") {
    return;
  }
  throw std::invalid_argument("LanczosSolver: invalid which option (use SR/LR/SM/LM/BE).");
}

core::CsrMatrixData make_core_csr(const fasteig_problem& problem, FASTEIG_INT n) {
  const auto* csr = problem.csr_matrix();
  if (!csr) {
    throw std::invalid_argument("LanczosSolver: executor path requires a stored CSR matrix.");
  }

  core::CsrMatrixData out;
  out.rows = n;
  out.cols = n;
  out.rowptr = csr->rowptr;
  out.colind = csr->colind;
  out.values = csr->values;
  return out;
}

std::shared_ptr<core::Executor> make_executor(const fasteig_options& options) {
  if (options.gpu_backend()) {
#ifdef USE_CUDA
    return core::make_cuda_executor();
#else
    throw std::invalid_argument("LanczosSolver: GPU backend requested but FASTEIG was not built with CUDA.");
#endif
  }
  return core::make_cpu_executor();
}

}  // namespace

LanczosSolver::LanczosSolver(::fasteig_problem& problem,
                             const fasteig_options& options)
    : fasteig_solver(problem, options) {}

FASTEIG_INT LanczosSolver::solve() {
  const operations& ops = active_ops();
  const FASTEIG_INT n = eigvecs_.rows();
  const FASTEIG_INT nev = options_.nev();
  const FASTEIG_INT maxiter = options_.maxiter();
  std::string which = lower_copy(options_.which());
  const int verbosity = options_.verbosity();
  const FASTEIG_INT print_every = options_.print_every();
  const FASTEIG_DOUBLE tol = options_.tol();

  if (maxiter <= 0) throw std::invalid_argument("LanczosSolver: maxiter must be > 0.");
  if (nev <= 0) throw std::invalid_argument("LanczosSolver: nev must be > 0.");
  if (maxiter < nev) throw std::invalid_argument("LanczosSolver: maxiter must be >= nev.");
  if (maxiter > n) throw std::invalid_argument("LanczosSolver: maxiter must be <= n.");
  if (!ops.matvec) throw std::invalid_argument("LanczosSolver: ops.matvec is not set.");
  validate_which(which);

  const bool use_executor_path = (which != "sai") && (problem_.csr_matrix() != nullptr);
  if (options_.gpu_backend() && !use_executor_path) {
    throw std::invalid_argument("LanczosSolver: GPU backend requires a stored matrix and does not support SAI.");
  }

  std::ostream* log = (print_every > 0) ? problem_.log_stream() : nullptr;
  fasteig_logger stdout_logger;
  if (verbosity >= 2) stdout_logger.attach(&std::cout);

  VectorXd f;
  if (options_.v0().size() > 0) {
    f = options_.v0().col(0);
  } else {
    f = VectorXd::Random(n);
  }

  MatrixXd V(n, maxiter);
  MatrixXd T(maxiter, maxiter);
  V.setZero();
  T.setZero();

  std::shared_ptr<core::Executor> exec;
  std::unique_ptr<core::CsrMatrix> A;
  std::unique_ptr<core::Vector> xf;
  std::unique_ptr<core::Vector> av;
  std::vector<std::unique_ptr<core::Vector>> basis;
  if (use_executor_path) {
    exec = make_executor(options_);
    A = exec->create_csr_matrix(make_core_csr(problem_, n));
    xf = exec->create_vector(n);
    av = exec->create_vector(n);
    xf->copy_from_host(f);
    basis.resize(static_cast<std::size_t>(maxiter));
  }

  struct Snapshot {
    VectorXd lam;
    VectorXd rel_res;
    MatrixXd Csel;
  };

  auto compute_snapshot = [&](FASTEIG_INT m, FASTEIG_DOUBLE beta_next, bool need_vectors) -> Snapshot {
    if (m <= 0) throw std::invalid_argument("LanczosSolver: snapshot m must be > 0.");
    const int m_i = static_cast<int>(m);
    const int nev_i = static_cast<int>(std::min<FASTEIG_INT>(nev, m));

    Eigen::SelfAdjointEigenSolver<MatrixXd> solver(T.topLeftCorner(m_i, m_i));
    const auto D = solver.eigenvalues();
    const auto C = solver.eigenvectors();

    std::vector<size_t> Ind(static_cast<size_t>(m), 0);
    if (which == "be") {
      int pos = 0;
      for (int i = 0; i < m_i / 2; ++i) {
        Ind[static_cast<size_t>(pos++)] = static_cast<size_t>(m_i - 1 - i);
        Ind[static_cast<size_t>(pos++)] = static_cast<size_t>(i);
      }
      if (m_i % 2 == 1) {
        Ind[static_cast<size_t>(pos)] = static_cast<size_t>(m_i / 2);
      }
    } else {
      Ind = sort_indices(D, which);
    }

    VectorXd lam(nev_i);
    VectorXd rel(nev_i);
    MatrixXd Csel;
    if (need_vectors) Csel.resize(m_i, nev_i);

    for (int j = 0; j < nev_i; ++j) {
      const int idx = static_cast<int>(Ind[static_cast<size_t>(j)]);
      const double lambda = D(idx);
      lam(j) = lambda;

      const double abs_res = std::abs(beta_next * C(m_i - 1, idx));
      rel(j) = abs_res / (1.0 + std::abs(lambda));

      if (need_vectors) Csel.col(j) = C.col(idx);
    }

    Snapshot out;
    out.lam = std::move(lam);
    out.rel_res = std::move(rel);
    out.Csel = std::move(Csel);
    return out;
  };

  auto emit_snapshot = [&](const Snapshot& s, FASTEIG_INT iter_label) {
    if (log) {
      problem_.logger().print_convergence_data(s.lam, s.rel_res, static_cast<int>(iter_label));
    }
    if (verbosity >= 2) {
      stdout_logger.print_convergence_data(s.lam, s.rel_res, static_cast<int>(iter_label));
    }
  };

  auto advance_executor = [&](FASTEIG_INT it) -> FASTEIG_DOUBLE {
    double beta = core::norm2(*xf);
    if (!std::isfinite(beta)) {
      throw std::runtime_error("LanczosSolver: non-finite residual norm encountered.");
    }

    if (beta <= std::numeric_limits<double>::epsilon()) {
      xf->copy_from_host(VectorXd::Random(static_cast<int>(n)));
      for (FASTEIG_INT j = 0; j < it; ++j) {
        const double h = core::dot(*basis[static_cast<std::size_t>(j)], *xf);
        core::axpy(-h, *basis[static_cast<std::size_t>(j)], *xf);
      }
      beta = core::norm2(*xf);
      if (beta <= std::numeric_limits<double>::epsilon()) {
        throw std::runtime_error("LanczosSolver: breakdown, unable to generate new basis vector.");
      }
    }

    auto v = exec->create_vector(n);
    core::copy(*xf, *v);
    core::scal(1.0 / beta, *v);
    basis[static_cast<std::size_t>(it)] = std::move(v);
    V.col(static_cast<int>(it)) = basis[static_cast<std::size_t>(it)]->copy_to_host();

    if (it != 0) {
      T(static_cast<int>(it), static_cast<int>(it - 1)) = beta;
      T(static_cast<int>(it - 1), static_cast<int>(it)) = beta;
    }

    core::spmv(*A, *basis[static_cast<std::size_t>(it)], *av);

    std::vector<double> h(static_cast<std::size_t>(it + 1), 0.0);
    core::copy(*av, *xf);
    for (FASTEIG_INT j = 0; j <= it; ++j) {
      h[static_cast<std::size_t>(j)] =
          core::dot(*basis[static_cast<std::size_t>(j)], *av);
      core::axpy(-h[static_cast<std::size_t>(j)],
                 *basis[static_cast<std::size_t>(j)],
                 *xf);
    }

    for (FASTEIG_INT j = 0; j <= it; ++j) {
      const double s = core::dot(*basis[static_cast<std::size_t>(j)], *xf);
      h[static_cast<std::size_t>(j)] += s;
      core::axpy(-s, *basis[static_cast<std::size_t>(j)], *xf);
    }

    T(static_cast<int>(it), static_cast<int>(it)) = h[static_cast<std::size_t>(it)];
    return core::norm2(*xf);
  };

  KrylovSubspace krylov(f, V, T);
  auto advance_krylov = [&](FASTEIG_INT it) -> FASTEIG_DOUBLE {
    const operations& ops_to_use = (which == "sai") ? opsSAI_ : opsA_;
    krylov.build(ops_to_use, static_cast<size_t>(it), static_cast<size_t>(it + 1));
    f = krylov.KrylovResidual();
    V = krylov.OrthoBasis();
    T = krylov.ProjectedMatrix();
    return f.norm();
  };

  bool converged = false;
  FASTEIG_INT it_stop = maxiter;
  FASTEIG_INT last_emitted_iter = 0;
  Snapshot final;
  FASTEIG_DOUBLE last_beta_next = 0.0;

  for (FASTEIG_INT it = 0; it < maxiter; ++it) {
    last_beta_next = use_executor_path ? advance_executor(it) : advance_krylov(it);

    const FASTEIG_INT m = it + 1;
    const bool do_progress = (print_every > 0) && ((m % print_every) == 0);
    const FASTEIG_INT check_every = (print_every > 0) ? print_every : 5;
    const bool do_check = (m >= nev) && ((m % check_every) == 0 || m == maxiter);

    if (!do_progress && !do_check) continue;

    Snapshot snap = compute_snapshot(m, last_beta_next, /*need_vectors=*/false);
    if (do_progress) {
      emit_snapshot(snap, m);
      last_emitted_iter = m;
    }

    if (do_check && snap.rel_res.size() > 0 && snap.rel_res.maxCoeff() < tol) {
      converged = true;
      it_stop = m;
      final = compute_snapshot(m, last_beta_next, /*need_vectors=*/true);
      if (!do_progress && print_every > 0) {
        emit_snapshot(final, m);
      }
      break;
    }
  }

  if (!converged) {
    final = compute_snapshot(maxiter, last_beta_next, /*need_vectors=*/true);
    if (print_every > 0 && last_emitted_iter != maxiter) {
      emit_snapshot(final, maxiter);
    }
  }

  if (which == "sai") {
    const FASTEIG_DOUBLE shift = options_.shift();
    std::vector<size_t> Ind(nev, 0);

    for (size_t j = 0; j < static_cast<size_t>(nev); ++j) {
      final.lam(static_cast<int>(j)) = 1.0 / final.lam(static_cast<int>(j)) + shift;
    }

    Ind = sort_indices(final.lam, "sr");
    for (size_t j = 0; j < static_cast<size_t>(nev); ++j) {
      eigvals_(static_cast<int>(j)) = final.lam(static_cast<int>(Ind[j]));
      eigvecs_.col(static_cast<int>(j)) =
          V.leftCols(static_cast<int>(it_stop)) * final.Csel.col(static_cast<int>(Ind[j]));
    }
  } else {
    eigvals_ = final.lam;
    eigvecs_.noalias() = V.leftCols(static_cast<int>(it_stop)) * final.Csel;
  }

  if (use_executor_path) {
    auto x = exec->create_vector(n);
    auto y = exec->create_vector(n);
    for (size_t j = 0; j < static_cast<size_t>(nev); ++j) {
      x->copy_from_host(eigvecs_.col(static_cast<int>(j)));
      core::spmv(*A, *x, *y);
      core::axpy(-eigvals_(static_cast<int>(j)), *x, *y);
      resnrms_(static_cast<int>(j)) = core::norm2(*y);
    }
  } else {
    VectorXd residual(n);
    for (size_t j = 0; j < static_cast<size_t>(nev); ++j) {
      opsA_.matvec(eigvecs_.col(static_cast<int>(j)), residual);
      residual -= eigvals_(static_cast<int>(j)) * eigvecs_.col(static_cast<int>(j));
      resnrms_(static_cast<int>(j)) = residual.norm();
    }
  }

  set_iterations(static_cast<long long>(it_stop));

  return status_;
}

}  // namespace fasteig

