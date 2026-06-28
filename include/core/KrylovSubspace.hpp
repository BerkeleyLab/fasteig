#pragma once

#include "fasteig.hpp"

namespace fasteig {
namespace core {

class LanczosWorkspace;
class LinearOperator;
class MultiVector;
class Vector;

class KrylovSubspace {
public:
  KrylovSubspace(const LinearOperator& op,
                 Vector& residual,
                 Vector& operator_result,
                 MultiVector& basis,
                 LanczosWorkspace& projection,
                 unsigned int restart_seed);
  KrylovSubspace(const LinearOperator& op,
                 Vector& residual,
                 Vector& operator_result,
                 MultiVector& basis,
                 MatrixXd& projected_matrix,
                 unsigned int restart_seed);

  void build(FASTEIG_INT begin, FASTEIG_INT end);

  Vector& residual() noexcept { return residual_; }
  const Vector& residual() const noexcept { return residual_; }
  MultiVector& basis() noexcept { return basis_; }
  const MultiVector& basis() const noexcept { return basis_; }

private:
  void build_one(FASTEIG_INT it, FASTEIG_INT begin, FASTEIG_INT end);
  double normalize_or_restart(FASTEIG_INT it);
  double orthogonalize_active_columns(FASTEIG_INT active_cols);
  void record_step(FASTEIG_INT it,
                   FASTEIG_INT begin,
                   FASTEIG_INT end,
                   double beta,
                   double alpha,
                   double beta_next);
  void restart_residual(FASTEIG_INT it);

  const LinearOperator& op_;
  Vector& residual_;
  Vector& operator_result_;
  MultiVector& basis_;
  LanczosWorkspace* lanczos_projection_ = nullptr;
  MatrixXd* projected_matrix_ = nullptr;
  unsigned int restart_seed_ = 1u;
};

class BlockKrylovSubspace {
public:
  BlockKrylovSubspace(const LinearOperator& op,
                      const MatrixXd& initial_block,
                      MultiVector& basis,
                      MatrixXd& projected_matrix,
                      MatrixXd& beta);

  void build(FASTEIG_INT begin, FASTEIG_INT end);
  const MatrixXd& host_basis() const noexcept { return host_basis_; }

private:
  void apply_block(FASTEIG_INT first_col, FASTEIG_INT block_size, MatrixXd& output) const;
  void copy_host_block_to_basis(FASTEIG_INT first_col, const MatrixXd& block);

  const LinearOperator& op_;
  const MatrixXd& initial_block_;
  MultiVector& basis_;
  MatrixXd& projected_matrix_;
  MatrixXd& beta_;
  MatrixXd host_basis_;
  bool initialized_ = false;
};

}  // namespace core
}  // namespace fasteig

