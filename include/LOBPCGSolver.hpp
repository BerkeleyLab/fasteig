#pragma once

#include "fasteig_solver.hpp"

class fasteig_problem;

namespace fasteig {

class LOBPCGSolver final : public fasteig_solver {
public:
  LOBPCGSolver(::fasteig_problem& problem,
               const fasteig_options& options);
  FASTEIG_INT solve() override;

private:
  void reset_state(double ctol,
                   double otol,
                   int vb,
                   int pe,
                   std::ostream* logstream);
  void initialize_XP_subspaces();
  void lock_eigenvectors();
  bool modified_rayleigh_ritz();

  int iter = -1;                 // Iteration counter
  operations ops_;               // Active operator (may be transformed for which=LR/LM)
  double conv_tol = 1e-14;       // Convergence tolerance
  double orth_tol = 1e-14;       // Orthogonalization tolerance
  int verbose = 1;               // Verbosity level
  int print_every = 0;           // Print convergence every N iterations (0 => never)

  VectorXd& lam;                 // Eigenvalue
  MatrixXd& V;                   // eigenvector arrays

  size_t n = 0;                  // row size
  size_t b = 0;                  // block size

  // Subspace arrays XP = [X, P] and W
  MatrixXd XP;
  MatrixXd AXP;
  MatrixXd W;
  MatrixXd AW;

  // Arrays for the Rayleigh-Ritz-step
  MatrixXd StAS;                 // Overlap matrix S'AS, size 3b x 3b
  MatrixXd StS;                  // Overlap matrix S'S, size 3b x 3b
  MatrixXd C;                    // Ritz Coefficients, size 3b x 2b

  // resnormual norms
  VectorXd resnorms;

  // Eigensolver for orthogonalization and RR step.
  Eigen::SelfAdjointEigenSolver<MatrixXd> eigsolver;

  // Number of converged eigenpairs
  int b_conv = 0;

  // Flag. Sets whether to orthogonalize preconditioned resnormuals W w.r.t. [X, P]
  bool use_orth = false;

  // Convergence logging
  bool logging = false;
  std::ostream* log = nullptr;   // non-owning; opened/managed by the caller
};

}  // namespace fasteig

void ortho(MatrixXd& U,
           const MatrixXd& V,
           double tol,
           Eigen::SelfAdjointEigenSolver<MatrixXd>& solver,
           int maxiter1 = 3,
           int maxiter2 = 3,
           double tau = 1e-10);

void svqb(MatrixXd& U,
          double tau,
          char method,
          Eigen::SelfAdjointEigenSolver<MatrixXd>& solver);

