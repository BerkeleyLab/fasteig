#pragma once

#include "fasteig.hpp"

#include <string>
#include <vector>

class fasteig_logger;

namespace fasteig {

struct LanczosSnapshot {
  VectorXd lam;
  VectorXd rel_res;
  MatrixXd Csel;
};

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

void lanczos_emit_snapshot(const LanczosSnapshot& snapshot,
                           FASTEIG_INT iter_label,
                           fasteig_logger* log_logger,
                           fasteig_logger* stdout_logger);

}  // namespace fasteig

