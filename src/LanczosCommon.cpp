#include "LanczosCommon.hpp"

#include "fasteig_solver.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace fasteig {

std::string lanczos_lower_copy(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

void validate_lanczos_which(const std::string& which,
                            bool allow_sai,
                            const char* solver_name) {
  if (which == "sr" || which == "lr" || which == "sm" || which == "lm" || which == "be") {
    return;
  }
  if (allow_sai && which == "sai") return;

  throw std::invalid_argument(std::string(solver_name) +
                              ": invalid which option (use SR/LR/SM/LM/BE).");
}

std::vector<size_t> lanczos_select_ritz_indices(const VectorXd& eigenvalues,
                                                const std::string& which) {
  const int m = static_cast<int>(eigenvalues.size());
  std::vector<size_t> indices(static_cast<size_t>(m), 0);
  if (which == "be") {
    int pos = 0;
    for (int i = 0; i < m / 2; ++i) {
      indices[static_cast<size_t>(pos++)] = static_cast<size_t>(m - 1 - i);
      indices[static_cast<size_t>(pos++)] = static_cast<size_t>(i);
    }
    if (m % 2 == 1) {
      indices[static_cast<size_t>(pos)] = static_cast<size_t>(m / 2);
    }
    return indices;
  }

  return sort_indices(eigenvalues, which);
}

LanczosSnapshot lanczos_make_snapshot_from_eigendecomp(const VectorXd& eigenvalues,
                                                       const MatrixXd& eigenvectors,
                                                       FASTEIG_INT nev,
                                                       FASTEIG_DOUBLE beta_next,
                                                       const std::string& which,
                                                       bool need_vectors) {
  const int m = static_cast<int>(eigenvalues.size());
  if (m <= 0) throw std::invalid_argument("lanczos_make_snapshot_from_eigendecomp: no eigenvalues.");
  if (eigenvectors.rows() != m || eigenvectors.cols() != m) {
    throw std::invalid_argument("lanczos_make_snapshot_from_eigendecomp: eigenvector dimension mismatch.");
  }

  const int nev_i = static_cast<int>(std::min<FASTEIG_INT>(nev, static_cast<FASTEIG_INT>(m)));
  const std::vector<size_t> indices = lanczos_select_ritz_indices(eigenvalues, which);

  VectorXd lam(nev_i);
  VectorXd rel(nev_i);
  MatrixXd Csel;
  if (need_vectors) Csel.resize(m, nev_i);

  for (int j = 0; j < nev_i; ++j) {
    const int idx = static_cast<int>(indices[static_cast<size_t>(j)]);
    const double lambda = eigenvalues(idx);
    lam(j) = lambda;

    const double abs_res = std::abs(beta_next * eigenvectors(m - 1, idx));
    rel(j) = abs_res / (1.0 + std::abs(lambda));

    if (need_vectors) Csel.col(j) = eigenvectors.col(idx);
  }

  LanczosSnapshot out;
  out.lam = std::move(lam);
  out.rel_res = std::move(rel);
  out.Csel = std::move(Csel);
  return out;
}

LanczosSnapshot lanczos_compute_cpu_snapshot(const MatrixXd& T,
                                             FASTEIG_INT m,
                                             FASTEIG_INT nev,
                                             FASTEIG_DOUBLE beta_next,
                                             const std::string& which,
                                             bool need_vectors,
                                             const char* solver_name) {
  if (m <= 0) {
    throw std::invalid_argument(std::string(solver_name) + ": snapshot m must be > 0.");
  }
  if (T.rows() < m || T.cols() < m) {
    throw std::invalid_argument(std::string(solver_name) + ": snapshot matrix is too small.");
  }

  const int m_i = static_cast<int>(m);
  Eigen::SelfAdjointEigenSolver<MatrixXd> solver(T.topLeftCorner(m_i, m_i));
  return lanczos_make_snapshot_from_eigendecomp(solver.eigenvalues(), solver.eigenvectors(),
                                               nev, beta_next, which, need_vectors);
}

LanczosConvergenceResult run_lanczos_convergence_loop(const LanczosConvergenceConfig& config,
                                                      const LanczosAdvanceFn& advance,
                                                      const LanczosSnapshotFn& snapshot,
                                                      const LanczosEmitFn& emit) {
  if (config.nev <= 0) {
    throw std::invalid_argument("run_lanczos_convergence_loop: nev must be > 0.");
  }
  if (config.maxiter <= 0) {
    throw std::invalid_argument("run_lanczos_convergence_loop: maxiter must be > 0.");
  }
  if (!advance) {
    throw std::invalid_argument("run_lanczos_convergence_loop: advance callback is empty.");
  }
  if (!snapshot) {
    throw std::invalid_argument("run_lanczos_convergence_loop: snapshot callback is empty.");
  }

  LanczosConvergenceResult result;
  result.it_stop = config.maxiter;
  FASTEIG_DOUBLE last_beta_next = 0.0;

  for (FASTEIG_INT it = 0; it < config.maxiter; ++it) {
    last_beta_next = advance(it);

    const FASTEIG_INT m = it + 1;
    const bool do_progress = (config.print_every > 0) && ((m % config.print_every) == 0);
    const bool do_silent_check =
        (config.print_every <= 0) &&
        (config.silent_check_interval > 0) &&
        ((m % config.silent_check_interval) == 0);
    const bool do_check = (m >= config.nev) && (do_progress || do_silent_check || m == config.maxiter);

    if (!do_progress && !do_check) continue;

    LanczosSnapshot snap = snapshot(m, last_beta_next, /*need_vectors=*/false);
    if (do_progress) {
      if (emit) emit(snap, m);
      result.last_emitted_iter = m;
    }

    if (do_check && snap.rel_res.size() > 0 && snap.rel_res.maxCoeff() < config.tol) {
      result.converged = true;
      result.it_stop = m;
      result.final = snapshot(m, last_beta_next, /*need_vectors=*/true);
      if (!do_progress && config.print_every > 0) {
        if (emit) emit(result.final, m);
      }
      return result;
    }
  }

  result.final = snapshot(config.maxiter, last_beta_next, /*need_vectors=*/true);
  if (config.print_every > 0 && result.last_emitted_iter != config.maxiter) {
    if (emit) emit(result.final, config.maxiter);
  }
  return result;
}

}  // namespace fasteig

