#pragma once

#include "fasteig.hpp"

#include <string>
#include <vector>

class fasteig_logger;

namespace fasteig {
namespace core {

class MultiVector;

struct LanczosSnapshot {
  VectorXd lam;
  VectorXd rel_res;
  MatrixXd Csel;
};

struct ProjectedRitzPairs {
  VectorXd values;
  VectorXd relative_residuals;
  MatrixXd coefficients;
};

ProjectedRitzPairs extract_projected_ritz_pairs(
    const MatrixXd& projected_matrix,
    const MultiVector& basis,
    const MultiVector& operator_basis,
    FASTEIG_INT active_cols,
    FASTEIG_INT requested,
    const std::string& which,
    MultiVector& ritz_vectors,
    MultiVector& operator_ritz_vectors,
    MultiVector& residuals,
    const char* solver_name);

std::string lanczos_lower_copy(std::string s);

void validate_lanczos_which(const std::string& which,
                            bool allow_sai,
                            const char* solver_name);

std::vector<size_t> lanczos_select_ritz_indices(const VectorXd& eigenvalues,
                                                const std::string& which);

LanczosSnapshot lanczos_make_snapshot_from_eigendecomp(const VectorXd& eigenvalues,
                                                       const MatrixXd& eigenvectors,
                                                       FASTEIG_INT nev,
                                                       FASTEIG_DOUBLE beta_next,
                                                       const std::string& which,
                                                       bool need_vectors);

LanczosSnapshot lanczos_make_snapshot_from_eigenvalues_last_row(const VectorXd& eigenvalues,
                                                               const VectorXd& eigenvector_last_row,
                                                               FASTEIG_INT nev,
                                                               FASTEIG_DOUBLE beta_next,
                                                               const std::string& which);

LanczosSnapshot lanczos_compute_cpu_snapshot(const MatrixXd& T,
                                             FASTEIG_INT m,
                                             FASTEIG_INT nev,
                                             FASTEIG_DOUBLE beta_next,
                                             const std::string& which,
                                             bool need_vectors,
                                             const char* solver_name);

LanczosSnapshot block_lanczos_compute_cpu_snapshot(const MatrixXd& T,
                                                   FASTEIG_INT block_count,
                                                   FASTEIG_INT block_size,
                                                   FASTEIG_INT nev,
                                                   const MatrixXd& beta,
                                                   const std::string& which,
                                                   bool need_vectors,
                                                   const char* solver_name);

LanczosSnapshot block_lanczos_make_snapshot_from_eigendecomp(const VectorXd& eigenvalues,
                                                             const MatrixXd& eigenvectors,
                                                             FASTEIG_INT block_size,
                                                             FASTEIG_INT nev,
                                                             const MatrixXd& beta,
                                                             const std::string& which,
                                                             bool need_vectors,
                                                             const char* solver_name);

LanczosSnapshot block_lanczos_make_snapshot_from_eigenvalues_last_block(
    const VectorXd& eigenvalues,
    const MatrixXd& eigenvector_last_block,
    FASTEIG_INT block_size,
    FASTEIG_INT nev,
    const MatrixXd& beta,
    const std::string& which,
    const char* solver_name);

LanczosSnapshot lanczos_prepare_final_snapshot(const LanczosSnapshot& snapshot,
                                               const std::string& which,
                                               FASTEIG_DOUBLE shift,
                                               const char* solver_name);

LanczosSnapshot lanczos_snapshot_head(const LanczosSnapshot& snapshot,
                                      FASTEIG_INT count,
                                      const char* solver_name);

VectorXd lanczos_absolute_residuals(const LanczosSnapshot& snapshot);

LanczosSnapshot krylov_schur_progress_snapshot(
    const LanczosSnapshot& snapshot,
    FASTEIG_INT nev,
    const std::string& which,
    FASTEIG_DOUBLE shift,
    const char* solver_name);

MatrixXd krylov_schur_restart_projection(const LanczosSnapshot& snapshot,
                                         FASTEIG_INT retained_cols,
                                         FASTEIG_DOUBLE beta_next,
                                         const char* solver_name);

void lanczos_emit_snapshot(const LanczosSnapshot& snapshot,
                           FASTEIG_INT iter_label,
                           fasteig_logger* log_logger,
                           fasteig_logger* stdout_logger);

}  // namespace core
}  // namespace fasteig

