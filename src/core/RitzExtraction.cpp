#include "core/RitzExtraction.hpp"

#include "fasteig_logger.hpp"
#include "fasteig_solver.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace fasteig {
namespace core {

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

LanczosSnapshot block_lanczos_compute_cpu_snapshot(const MatrixXd& T,
                                                   FASTEIG_INT block_count,
                                                   FASTEIG_INT block_size,
                                                   FASTEIG_INT nev,
                                                   const MatrixXd& beta,
                                                   const std::string& which,
                                                   bool need_vectors,
                                                   const char* solver_name) {
  if (block_count <= 0) {
    throw std::invalid_argument(std::string(solver_name) + ": block snapshot count must be > 0.");
  }
  if (block_size <= 0) {
    throw std::invalid_argument(std::string(solver_name) + ": block size must be > 0.");
  }
  if (beta.rows() != block_size || beta.cols() != block_size) {
    throw std::invalid_argument(std::string(solver_name) + ": block snapshot beta shape mismatch.");
  }

  const FASTEIG_INT m = block_count * block_size;
  if (T.rows() < m || T.cols() < m) {
    throw std::invalid_argument(std::string(solver_name) + ": block snapshot matrix is too small.");
  }

  const int m_i = static_cast<int>(m);
  Eigen::SelfAdjointEigenSolver<MatrixXd> solver(T.topLeftCorner(m_i, m_i));
  return block_lanczos_make_snapshot_from_eigendecomp(
      solver.eigenvalues(), solver.eigenvectors(), block_size, nev, beta, which,
      need_vectors, solver_name);
}

LanczosSnapshot block_lanczos_make_snapshot_from_eigendecomp(const VectorXd& eigenvalues,
                                                             const MatrixXd& eigenvectors,
                                                             FASTEIG_INT block_size,
                                                             FASTEIG_INT nev,
                                                             const MatrixXd& beta,
                                                             const std::string& which,
                                                             bool need_vectors,
                                                             const char* solver_name) {
  const int m = static_cast<int>(eigenvalues.size());
  if (m <= 0) {
    throw std::invalid_argument(std::string(solver_name) + ": no block Ritz values.");
  }
  if (block_size <= 0 || m < block_size) {
    throw std::invalid_argument(std::string(solver_name) + ": invalid block size.");
  }
  if (eigenvectors.rows() != m || eigenvectors.cols() != m) {
    throw std::invalid_argument(std::string(solver_name) + ": block eigenvector dimension mismatch.");
  }
  if (beta.rows() != block_size || beta.cols() != block_size) {
    throw std::invalid_argument(std::string(solver_name) + ": block beta shape mismatch.");
  }

  MatrixXd last_block(static_cast<int>(block_size), m);
  last_block = eigenvectors.bottomRows(static_cast<int>(block_size));
  LanczosSnapshot out = block_lanczos_make_snapshot_from_eigenvalues_last_block(
      eigenvalues, last_block, block_size, nev, beta, which, solver_name);

  if (need_vectors) {
    out.Csel.resize(m, out.lam.size());
    const std::vector<size_t> indices = lanczos_select_ritz_indices(eigenvalues, which);
    for (int j = 0; j < out.lam.size(); ++j) {
      out.Csel.col(j) = eigenvectors.col(static_cast<int>(indices[static_cast<std::size_t>(j)]));
    }
  }
  return out;
}

LanczosSnapshot block_lanczos_make_snapshot_from_eigenvalues_last_block(
    const VectorXd& eigenvalues,
    const MatrixXd& eigenvector_last_block,
    FASTEIG_INT block_size,
    FASTEIG_INT nev,
    const MatrixXd& beta,
    const std::string& which,
    const char* solver_name) {
  const int m = static_cast<int>(eigenvalues.size());
  if (m <= 0) {
    throw std::invalid_argument(std::string(solver_name) + ": no block Ritz values.");
  }
  if (block_size <= 0) {
    throw std::invalid_argument(std::string(solver_name) + ": invalid block size.");
  }
  if (eigenvector_last_block.rows() != block_size || eigenvector_last_block.cols() != m) {
    throw std::invalid_argument(std::string(solver_name) + ": last block shape mismatch.");
  }
  if (beta.rows() != block_size || beta.cols() != block_size) {
    throw std::invalid_argument(std::string(solver_name) + ": block beta shape mismatch.");
  }

  const int nev_i = static_cast<int>(std::min<FASTEIG_INT>(nev, static_cast<FASTEIG_INT>(m)));
  if (nev_i <= 0) return {};

  const std::vector<size_t> indices = lanczos_select_ritz_indices(eigenvalues, which);
  VectorXd lam(nev_i);
  VectorXd rel_res(nev_i);

  for (int j = 0; j < nev_i; ++j) {
    const int idx = static_cast<int>(indices[static_cast<std::size_t>(j)]);
    const double lambda = eigenvalues(idx);
    lam(j) = lambda;

    const VectorXd y_last = eigenvector_last_block.col(idx);
    const double abs_res = (beta * y_last).norm();
    rel_res(j) = abs_res / (1.0 + std::abs(lambda));
  }

  LanczosSnapshot out;
  out.lam = std::move(lam);
  out.rel_res = std::move(rel_res);
  return out;
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

LanczosSnapshot lanczos_snapshot_head(const LanczosSnapshot& snapshot,
                                      FASTEIG_INT count,
                                      const char* solver_name) {
  if (count < 0 || count > snapshot.lam.size() || count > snapshot.rel_res.size() ||
      (snapshot.Csel.size() > 0 && count > snapshot.Csel.cols())) {
    throw std::invalid_argument(std::string(solver_name) +
                                ": invalid snapshot truncation.");
  }

  LanczosSnapshot out;
  out.lam = snapshot.lam.head(static_cast<int>(count));
  out.rel_res = snapshot.rel_res.head(static_cast<int>(count));
  if (snapshot.Csel.size() > 0) {
    out.Csel = snapshot.Csel.leftCols(static_cast<int>(count));
  }
  return out;
}

VectorXd lanczos_absolute_residuals(const LanczosSnapshot& snapshot) {
  if (snapshot.lam.size() != snapshot.rel_res.size()) {
    throw std::invalid_argument(
        "lanczos_absolute_residuals: Ritz-value/residual size mismatch.");
  }
  return (snapshot.rel_res.array() * (1.0 + snapshot.lam.array().abs())).matrix();
}

LanczosSnapshot krylov_schur_progress_snapshot(
    const LanczosSnapshot& snapshot,
    FASTEIG_INT nev,
    const std::string& which,
    FASTEIG_DOUBLE shift,
    const char* solver_name) {
  LanczosSnapshot out = lanczos_snapshot_head(snapshot, nev, solver_name);
  out.rel_res = lanczos_absolute_residuals(out);
  if (which == "sai") {
    for (int j = 0; j < out.lam.size(); ++j) {
      if (out.lam(j) == 0.0) {
        throw std::runtime_error(std::string(solver_name) +
                                 ": zero shift-invert Ritz value.");
      }
      out.lam(j) = shift + 1.0 / out.lam(j);
    }
  }
  return out;
}

MatrixXd krylov_schur_restart_projection(const LanczosSnapshot& snapshot,
                                         FASTEIG_INT retained_cols,
                                         FASTEIG_DOUBLE beta_next,
                                         const char* solver_name) {
  if (retained_cols <= 0 || retained_cols > snapshot.lam.size()) {
    throw std::invalid_argument(std::string(solver_name) +
                                ": invalid retained Ritz-vector count.");
  }
  if (snapshot.Csel.rows() <= retained_cols ||
      snapshot.Csel.cols() < retained_cols) {
    throw std::invalid_argument(std::string(solver_name) +
                                ": restart coefficient matrix is too small.");
  }

  MatrixXd projected = MatrixXd::Zero(static_cast<int>(retained_cols + 1),
                                      static_cast<int>(retained_cols + 1));
  for (FASTEIG_INT j = 0; j < retained_cols; ++j) {
    projected(static_cast<int>(j), static_cast<int>(j)) = snapshot.lam(static_cast<int>(j));
    const double coupling =
        beta_next * snapshot.Csel(snapshot.Csel.rows() - 1, static_cast<int>(j));
    projected(static_cast<int>(retained_cols), static_cast<int>(j)) = coupling;
    projected(static_cast<int>(j), static_cast<int>(retained_cols)) = coupling;
  }
  return projected;
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

}  // namespace core
}  // namespace fasteig

