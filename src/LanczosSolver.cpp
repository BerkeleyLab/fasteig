#include "LanczosSolver.hpp"

#include "fasteig_problem.hpp"
#include "fasteig_logger.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <string>
#include <vector>

namespace fasteig {

LanczosSolver::LanczosSolver(::fasteig_problem& problem,
                             const fasteig_options& options)
    : fasteig_solver(problem, options) {}

FASTEIG_INT LanczosSolver::solve() {
  const operations& ops = active_ops();
  const FASTEIG_INT n = eigvecs_.rows();
  const FASTEIG_INT nev = options_.nev();
  const FASTEIG_INT maxiter = options_.maxiter();
  std::string which = options_.which();
  const int verbosity = options_.verbosity();
  const FASTEIG_INT print_every = options_.print_every();
  const FASTEIG_DOUBLE tol = options_.tol();

  if (maxiter <= 0) throw std::invalid_argument("LanczosSolver: maxiter must be > 0.");
  if (nev <= 0) throw std::invalid_argument("LanczosSolver: nev must be > 0.");
  if (maxiter < nev) throw std::invalid_argument("LanczosSolver: maxiter must be >= nev.");
  if (maxiter > n) throw std::invalid_argument("LanczosSolver: maxiter must be <= n.");
  if (!ops.matvec) throw std::invalid_argument("LanczosSolver: ops.matvec is not set.");

  auto lower_copy = [](std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
  };

  which = lower_copy(which);
  
  if (which != "sr" && which != "lr" && which != "sm" && which != "lm" && which != "be" && which != "sai") {
    throw std::invalid_argument("LanczosSolver: invalid which option (use SR/LR/SM/LM/BE).");
  }

  std::ostream* log = (print_every > 0) ? problem_.log_stream() : nullptr;
  fasteig_logger stdout_logger;
  if (verbosity >= 2) stdout_logger.attach(&std::cout);

  // Use the first initial vector (options_.v0 is initialized by the driver).
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

  // Efficient Lanczos progress snapshot:
  // - Eigenpairs come from the small tridiagonal T_m.
  // - Residual norms use the standard estimate ||r|| = |beta_{m} * y_last|,
  //   where y_last is the last component of the eigenvector of T_m and
  //   beta_m = ||f|| is the norm of the next residual vector.
  struct Snapshot {
    VectorXd lam;      // selected Ritz values
    VectorXd rel_res;  // relative residual norms
    MatrixXd Csel;     // selected eigenvectors of T_m (m x nev_i), for final eigenvectors
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

  bool converged = false;
  FASTEIG_INT it_stop = maxiter;
  FASTEIG_INT last_emitted_iter = 0;
  Snapshot final;

  for (FASTEIG_INT it = 0; it < maxiter; ++it) {
    // Build the Krylov basis incrementally to preserve KrylovSubspace's expected
    // boundary behavior across algorithms (e.g., Krylov-Schur restarts).
    const operations& ops_to_use = (which == "sai") ? opsSAI_ : opsA_;
    KrylovSubspace(f, V, T, ops_to_use, static_cast<size_t>(it), static_cast<size_t>(it + 1));

    const FASTEIG_INT m = it + 1;
    const FASTEIG_DOUBLE beta_next = f.norm();

    const bool do_progress = (print_every > 0) && ((m % print_every) == 0);
    const FASTEIG_INT check_every = (print_every > 0) ? print_every : 5;
    const bool do_check = (m >= nev) && ((m % check_every) == 0 || m == maxiter);

    if (!do_progress && !do_check) continue;

    Snapshot snap = compute_snapshot(/*m=*/m, beta_next, /*need_vectors=*/false);
    if (do_progress) {
      emit_snapshot(snap, /*iter_label=*/m);
      last_emitted_iter = m;
    }

    if (do_check) {
      if (snap.rel_res.size() == 0) continue;
      if (snap.rel_res.maxCoeff() < tol) {
        converged = true;
        it_stop = m;
        final = compute_snapshot(/*m=*/m, beta_next, /*need_vectors=*/true);
        // If user didn't request progress printing at this iteration, still record the
        // final snapshot to file when logging is enabled.
        if (!do_progress && print_every > 0) {
          emit_snapshot(final, /*iter_label=*/m);
        }
        break;
      }
    }
  }

  if (!converged) {
    const FASTEIG_DOUBLE beta_next = f.norm();
    final = compute_snapshot(/*m=*/maxiter, beta_next, /*need_vectors=*/true);
    if (print_every > 0 && last_emitted_iter != maxiter) {
      emit_snapshot(final, /*iter_label=*/maxiter);
    }
  }

  VectorXd residual(nev);

  if (which == "sai") {
    FASTEIG_DOUBLE shift = options_.shift();
    std::vector<size_t> Ind(nev, 0);

    for (size_t j = 0; j < nev; ++j) {
        final.lam(static_cast<int>(j)) = 1.0 / final.lam(static_cast<int>(j)) + shift;   
    }

    Ind = sort_indices(final.lam, "sr"); // resort the transformed Ritz values for reporting convergence
    for (size_t j = 0; j < nev; ++j) {
      eigvals_(static_cast<int>(j)) = final.lam(static_cast<int>(Ind[j]));
      eigvecs_.col(static_cast<int>(j)) = V.leftCols(static_cast<int>(it_stop)) * final.Csel.col(static_cast<int>(Ind[j]));
    }
  }
  else {
    eigvals_ = final.lam;
    eigvecs_.noalias() = V.leftCols(static_cast<int>(it_stop)) * final.Csel;
  }

  for (size_t j = 0; j < nev; ++j) {
    opsA_.matvec(eigvecs_.col(static_cast<int>(j)), residual);
    residual -= eigvals_(static_cast<int>(j)) * eigvecs_.col(static_cast<int>(j));
    resnrms_(static_cast<int>(j)) = residual.norm();
  }

  
  set_iterations(static_cast<long long>(it_stop));

  return status_;
}

}  // namespace fasteig
