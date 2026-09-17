#pragma once

#include "fasteig.hpp"

#include <memory>

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
                 unsigned int restart_seed,
                 const LinearOperator* metric = nullptr);
  ~KrylovSubspace();
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
  double normalize_residual_into_basis(FASTEIG_INT basis_col);
  void apply_operator_to_basis_column(FASTEIG_INT basis_col);
  double orthogonalize_operator_result(FASTEIG_INT active_cols);
  double residual_norm() const;
  void rotate_basis(FASTEIG_INT active_cols,
                    const Eigen::Ref<const MatrixXd>& coefficients);

  Vector& residual() noexcept { return residual_; }
  const Vector& residual() const noexcept { return residual_; }
  MultiVector& basis() noexcept { return basis_; }
  const MultiVector& basis() const noexcept { return basis_; }

private:
  void build_one(FASTEIG_INT it, FASTEIG_INT begin, FASTEIG_INT end);
  double normalize_or_restart(FASTEIG_INT it);
  void record_step(FASTEIG_INT it,
                   FASTEIG_INT begin,
                   FASTEIG_INT end,
                   double beta,
                   double alpha,
                   double beta_next);
  void restart_residual(FASTEIG_INT it);

  const LinearOperator& op_;
  const LinearOperator* metric_ = nullptr;
  std::unique_ptr<MultiVector> metric_basis_;
  std::unique_ptr<Vector> metric_residual_;
  Vector& residual_;
  Vector& operator_result_;
  MultiVector& basis_;
  LanczosWorkspace* lanczos_projection_ = nullptr;
  MatrixXd* projected_matrix_ = nullptr;
  unsigned int restart_seed_ = 1u;
  bool restarted_column_ = false;
};

class BlockKrylovSubspace {
public:
  BlockKrylovSubspace(const LinearOperator& op,
                      const MultiVector& initial_block,
                      MultiVector& basis,
                      const LinearOperator* metric = nullptr,
                      unsigned int restart_seed = 1u);

  void initialize_basis_block();
  void apply_operator_to_basis_block(FASTEIG_INT block_index);
  void orthogonalize_operator_block(FASTEIG_INT active_cols);
  void qr_residual_block(FASTEIG_INT next_block_index);
  // Retain any positive whole number of blocks, then append the old residual
  // block. Rotate the cached B images with the same coefficients.
  void restart_with_ritz_block(
      FASTEIG_INT active_cols,
      const Eigen::Ref<const MatrixXd>& coefficients,
      FASTEIG_INT residual_block_index);
  bool replenished_directions() const noexcept { return replenished_directions_; }
  const MatrixXd& diagonal_block() const noexcept { return diagonal_block_; }
  const MatrixXd& coupling_block() const noexcept { return coupling_block_; }

private:
  void copy_block_to_basis(FASTEIG_INT first_col, const MultiVector& block);

  const LinearOperator& op_;
  const LinearOperator* metric_ = nullptr;
  unsigned int restart_seed_ = 1u;
  std::unique_ptr<MultiVector> metric_basis_;
  std::unique_ptr<MultiVector> metric_block_;
  std::unique_ptr<MultiVector> qr_block_;
  std::unique_ptr<MultiVector> restart_metric_ritz_block_;
  std::unique_ptr<MultiVector> restart_metric_residual_block_;
  const MultiVector& initial_block_;
  MultiVector& basis_;
  std::unique_ptr<MultiVector> operator_block_;
  std::unique_ptr<MultiVector> residual_block_;
  std::unique_ptr<MultiVector> restart_ritz_block_;
  std::unique_ptr<MultiVector> restart_residual_block_;
  MatrixXd diagonal_block_;
  MatrixXd coupling_block_;
  bool initialized_ = false;
  bool replenished_directions_ = false;
};

}  // namespace core
}  // namespace fasteig

