#pragma once

#include "fasteig.hpp"

class KrylovSubspace {
public:
  KrylovSubspace() = default;
  KrylovSubspace(VectorXd& KrylovResidual,
                 MatrixXd& OrthoBasis,
                 MatrixXd& ProjectedMatrix);
  KrylovSubspace(const MatrixXd& V0,
                 MatrixXd& OrthoBasis,
                 MatrixXd& ProjectedMatrix,
                 MatrixXd& beta);

  static void getevbounds(FASTEIG_INT n,
                          const operations& ops,
                          FASTEIG_DOUBLE& evlb,
                          FASTEIG_DOUBLE& evub);

  void build(const operations& ops,
             FASTEIG_SIZET begin,
             FASTEIG_SIZET end);

  void build_block(const operations& ops,
                   FASTEIG_SIZET begin,
                   FASTEIG_SIZET end);

  VectorXd& KrylovResidual() noexcept { return *krylov_residual_; }
  const VectorXd& KrylovResidual() const noexcept { return *krylov_residual_; }
  const MatrixXd& V0() const noexcept { return *v0_; }
  MatrixXd& OrthoBasis() noexcept { return *ortho_basis_; }
  const MatrixXd& OrthoBasis() const noexcept { return *ortho_basis_; }
  MatrixXd& ProjectedMatrix() noexcept { return *projected_matrix_; }
  const MatrixXd& ProjectedMatrix() const noexcept { return *projected_matrix_; }
  MatrixXd& beta() noexcept { return *beta_; }
  const MatrixXd& beta() const noexcept { return *beta_; }
  FASTEIG_DOUBLE beta_scalar() const noexcept { return beta_scalar_; }

  // Compatibility wrappers for existing callers.
  static void build(VectorXd& KrylovResidual,
                    MatrixXd& OrthoBasis,
                    MatrixXd& ProjectedMatrix,
                    const operations& ops,
                    FASTEIG_SIZET begin,
                    FASTEIG_SIZET end);

  static void build(const MatrixXd& V0,
                    MatrixXd& OrthoBasis,
                    MatrixXd& ProjectedMatrix,
                    const operations& ops,
                    FASTEIG_SIZET begin,
                    FASTEIG_SIZET end,
                    MatrixXd& beta);

private:
  VectorXd* krylov_residual_ = nullptr;
  const MatrixXd* v0_ = nullptr;
  MatrixXd* ortho_basis_ = nullptr;
  MatrixXd* projected_matrix_ = nullptr;
  MatrixXd* beta_ = nullptr;
  FASTEIG_DOUBLE beta_scalar_ = 0.0;
};

