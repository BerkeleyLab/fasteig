#include "LanczosCommon.hpp"

#include "fasteig_logger.hpp"
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

LanczosSnapshot lanczos_make_snapshot_from_eigenvalues_last_row(const VectorXd& eigenvalues,
                                                               const VectorXd& eigenvector_last_row,
                                                               FASTEIG_INT nev,
                                                               FASTEIG_DOUBLE beta_next,
                                                               const std::string& which) {
  const int m = static_cast<int>(eigenvalues.size());
  if (m <= 0) {
    throw std::invalid_argument("lanczos_make_snapshot_from_eigenvalues_last_row: no eigenvalues.");
  }
  if (eigenvector_last_row.size() != m) {
    throw std::invalid_argument(
        "lanczos_make_snapshot_from_eigenvalues_last_row: last-row dimension mismatch.");
  }

  const int nev_i = static_cast<int>(std::min<FASTEIG_INT>(nev, static_cast<FASTEIG_INT>(m)));
  const std::vector<size_t> indices = lanczos_select_ritz_indices(eigenvalues, which);

  VectorXd lam(nev_i);
  VectorXd rel(nev_i);
  for (int j = 0; j < nev_i; ++j) {
    const int idx = static_cast<int>(indices[static_cast<std::size_t>(j)]);
    const double lambda = eigenvalues(idx);
    lam(j) = lambda;
    const double abs_res = std::abs(beta_next * eigenvector_last_row(idx));
    rel(j) = abs_res / (1.0 + std::abs(lambda));
  }

  LanczosSnapshot out;
  out.lam = std::move(lam);
  out.rel_res = std::move(rel);
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

LanczosSnapshot lanczos_prepare_final_snapshot(const LanczosSnapshot& snapshot,
                                               const std::string& which,
                                               FASTEIG_DOUBLE shift,
                                               const char* solver_name) {
  if (which != "sai") return snapshot;
  if (snapshot.Csel.cols() != snapshot.lam.size()) {
    throw std::invalid_argument(std::string(solver_name) +
                                ": SAI final snapshot has inconsistent dimensions.");
  }

  VectorXd transformed(snapshot.lam.size());
  for (int j = 0; j < snapshot.lam.size(); ++j) {
    if (snapshot.lam(j) == 0.0) {
      throw std::runtime_error(std::string(solver_name) +
                               ": SAI encountered a zero shift-invert Ritz value.");
    }
    transformed(j) = 1.0 / snapshot.lam(j) + shift;
  }

  const std::vector<size_t> indices = sort_indices(transformed, "sr");
  LanczosSnapshot out;
  out.lam.resize(transformed.size());
  out.rel_res.resize(snapshot.rel_res.size());
  out.Csel.resize(snapshot.Csel.rows(), snapshot.Csel.cols());

  for (int j = 0; j < transformed.size(); ++j) {
    const int idx = static_cast<int>(indices[static_cast<std::size_t>(j)]);
    out.lam(j) = transformed(idx);
    if (j < out.rel_res.size() && idx < snapshot.rel_res.size()) {
      out.rel_res(j) = snapshot.rel_res(idx);
    }
    out.Csel.col(j) = snapshot.Csel.col(idx);
  }
  return out;
}

void lanczos_emit_snapshot(const LanczosSnapshot& snapshot,
                           FASTEIG_INT iter_label,
                           fasteig_logger* log_logger,
                           fasteig_logger* stdout_logger) {
  if (log_logger) {
    log_logger->print_convergence_data(snapshot.lam, snapshot.rel_res,
                                       static_cast<int>(iter_label));
  }
  if (stdout_logger) {
    stdout_logger->print_convergence_data(snapshot.lam, snapshot.rel_res,
                                          static_cast<int>(iter_label));
  }
}

}  // namespace fasteig

