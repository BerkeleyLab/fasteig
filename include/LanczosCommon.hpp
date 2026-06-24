#pragma once

#include "fasteig.hpp"

#include <functional>
#include <string>
#include <vector>

namespace fasteig {

struct LanczosSnapshot {
  VectorXd lam;
  VectorXd rel_res;
  MatrixXd Csel;
};

struct LanczosConvergenceConfig {
  FASTEIG_INT nev = 0;
  FASTEIG_INT maxiter = 0;
  FASTEIG_INT print_every = 0;
  FASTEIG_INT silent_check_interval = 0;
  FASTEIG_DOUBLE tol = 0.0;
};

struct LanczosConvergenceResult {
  bool converged = false;
  FASTEIG_INT it_stop = 0;
  FASTEIG_INT last_emitted_iter = 0;
  LanczosSnapshot final;
};

using LanczosAdvanceFn = std::function<FASTEIG_DOUBLE(FASTEIG_INT it)>;
using LanczosSnapshotFn = std::function<LanczosSnapshot(FASTEIG_INT m,
                                                        FASTEIG_DOUBLE beta_next,
                                                        bool need_vectors)>;
using LanczosEmitFn = std::function<void(const LanczosSnapshot& snapshot,
                                         FASTEIG_INT iter_label)>;

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

LanczosSnapshot lanczos_compute_cpu_snapshot(const MatrixXd& T,
                                             FASTEIG_INT m,
                                             FASTEIG_INT nev,
                                             FASTEIG_DOUBLE beta_next,
                                             const std::string& which,
                                             bool need_vectors,
                                             const char* solver_name);

LanczosConvergenceResult run_lanczos_convergence_loop(const LanczosConvergenceConfig& config,
                                                      const LanczosAdvanceFn& advance,
                                                      const LanczosSnapshotFn& snapshot,
                                                      const LanczosEmitFn& emit);

}  // namespace fasteig

