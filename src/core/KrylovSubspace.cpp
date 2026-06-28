#include "core/KrylovSubspace.hpp"

#include "core/Executor.hpp"
#include "core/LanczosWorkspace.hpp"
#include "core/LinearOperator.hpp"
#include "core/MultiVector.hpp"
#include "core/Operations.hpp"
#include "core/Vector.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace fasteig {
namespace core {

KrylovSubspace::KrylovSubspace(const LinearOperator& op,
                               Vector& residual,
                               Vector& operator_result,
                               MultiVector& basis,
                               LanczosWorkspace& projection,
                               unsigned int restart_seed)
    : op_(op),
      residual_(residual),
      operator_result_(operator_result),
      basis_(basis),
      lanczos_projection_(&projection),
      projected_matrix_(nullptr),
      restart_seed_(restart_seed) {
  if (&op_.executor() != &residual_.executor() ||
      &op_.executor() != &operator_result_.executor() ||
      &op_.executor() != &basis_.executor()) {
    throw std::invalid_argument("core::KrylovSubspace: executor mismatch.");
  }
  if (op_.rows() != operator_result_.size() || op_.cols() != residual_.size()) {
    throw std::invalid_argument("core::KrylovSubspace: operator/vector dimension mismatch.");
  }
  if (basis_.rows() != residual_.size()) {
    throw std::invalid_argument("core::KrylovSubspace: basis row count mismatch.");
  }
}

KrylovSubspace::KrylovSubspace(const LinearOperator& op,
                               Vector& residual,
                               Vector& operator_result,
                               MultiVector& basis,
                               MatrixXd& projected_matrix,
                               unsigned int restart_seed)
    : op_(op),
      residual_(residual),
      operator_result_(operator_result),
      basis_(basis),
      lanczos_projection_(nullptr),
      projected_matrix_(&projected_matrix),
      restart_seed_(restart_seed) {
  if (&op_.executor() != &residual_.executor() ||
      &op_.executor() != &operator_result_.executor() ||
      &op_.executor() != &basis_.executor()) {
    throw std::invalid_argument("core::KrylovSubspace: executor mismatch.");
  }
  if (op_.rows() != operator_result_.size() || op_.cols() != residual_.size()) {
    throw std::invalid_argument("core::KrylovSubspace: operator/vector dimension mismatch.");
  }
  if (basis_.rows() != residual_.size()) {
    throw std::invalid_argument("core::KrylovSubspace: basis row count mismatch.");
  }
  if (projected_matrix.rows() < basis_.cols() || projected_matrix.cols() < basis_.cols()) {
    throw std::invalid_argument("core::KrylovSubspace: projected matrix is too small.");
  }
}

void KrylovSubspace::build(FASTEIG_INT begin, FASTEIG_INT end) {
  if (begin < 0 || end < begin) {
    throw std::invalid_argument("core::KrylovSubspace::build: invalid range.");
  }
  if (end > basis_.cols()) {
    throw std::invalid_argument("core::KrylovSubspace::build: range exceeds basis columns.");
  }

  for (FASTEIG_INT it = begin; it < end; ++it) {
    build_one(it, begin, end);
  }
}

void KrylovSubspace::build_one(FASTEIG_INT it, FASTEIG_INT begin, FASTEIG_INT end) {
  const double beta = normalize_or_restart(it);

  scal(1.0 / beta, residual_);
  copy_column_from(basis_, it, residual_);
  apply_column(op_, basis_, it, operator_result_);

  const double alpha = orthogonalize_active_columns(it + 1);
  const double beta_next = norm2(residual_);
  if (!std::isfinite(beta_next)) {
    throw std::runtime_error("core::KrylovSubspace: non-finite next residual norm encountered.");
  }

  record_step(it, begin, end, beta, alpha, beta_next);
}

double KrylovSubspace::orthogonalize_active_columns(FASTEIG_INT active_cols) {
  return fasteig::core::orthogonalize_active_columns(
      basis_, active_cols, operator_result_, residual_);
}

void KrylovSubspace::record_step(FASTEIG_INT it,
                                 FASTEIG_INT begin,
                                 FASTEIG_INT end,
                                 double beta,
                                 double alpha,
                                 double beta_next) {
  if (lanczos_projection_) {
    lanczos_projection_->record_step(it, beta, alpha, beta_next);
    return;
  }

  if (!projected_matrix_) {
    throw std::logic_error("core::KrylovSubspace: no projection target configured.");
  }

  MatrixXd& H = *projected_matrix_;
  if (it < 0 || it >= H.rows() || it >= H.cols()) {
    throw std::invalid_argument("core::KrylovSubspace: projected matrix index out of range.");
  }

  // Krylov-Schur restarts seed a dense coupling row before expansion.  Preserve the
  // legacy helper's convention: do not overwrite the first coupling of a multi-step
  // restarted expansion, but do write it for single-step expansions.
  if (it != begin || (begin != 0 && end - begin == 1)) {
    H(static_cast<int>(it), static_cast<int>(it - 1)) = beta;
    H(static_cast<int>(it - 1), static_cast<int>(it)) = beta;
  }
  H(static_cast<int>(it), static_cast<int>(it)) = alpha;
  (void)beta_next;
}

double KrylovSubspace::normalize_or_restart(FASTEIG_INT it) {
  double beta = norm2(residual_);
  if (!std::isfinite(beta)) {
    throw std::runtime_error("core::KrylovSubspace: non-finite residual norm encountered.");
  }
  if (beta > std::numeric_limits<double>::epsilon()) {
    return beta;
  }

  restart_residual(it);
  beta = norm2(residual_);
  if (beta <= std::numeric_limits<double>::epsilon()) {
    throw std::runtime_error("core::KrylovSubspace: breakdown, unable to generate new basis vector.");
  }
  return beta;
}

void KrylovSubspace::restart_residual(FASTEIG_INT it) {
  auto candidate =
      residual_.executor().create_random_vector(residual_.size(),
                                                restart_seed_ + static_cast<unsigned int>(it + 1));
  copy(*candidate, residual_);

  for (FASTEIG_INT j = 0; j < it; ++j) {
    const double h = column_dot(basis_, j, residual_);
    column_axpy(-h, basis_, j, residual_);
  }
}

BlockKrylovSubspace::BlockKrylovSubspace(const LinearOperator& op,
                                         const MatrixXd& initial_block,
                                         MultiVector& basis,
                                         MatrixXd& projected_matrix,
                                         MatrixXd& beta)
    : op_(op),
      initial_block_(initial_block),
      basis_(basis),
      projected_matrix_(projected_matrix),
      beta_(beta),
      host_basis_(static_cast<int>(basis.rows()), static_cast<int>(basis.cols())) {
  if (&op_.executor() != &basis_.executor()) {
    throw std::invalid_argument("core::BlockKrylovSubspace: executor mismatch.");
  }
  if (initial_block.rows() != basis.rows()) {
    throw std::invalid_argument("core::BlockKrylovSubspace: initial block row count mismatch.");
  }
  if (initial_block.cols() <= 0) {
    throw std::invalid_argument("core::BlockKrylovSubspace: block size must be positive.");
  }
  if (projected_matrix.rows() < basis.cols() || projected_matrix.cols() < basis.cols()) {
    throw std::invalid_argument("core::BlockKrylovSubspace: projected matrix is too small.");
  }
  if (beta.rows() != initial_block.cols() || beta.cols() != initial_block.cols()) {
    throw std::invalid_argument("core::BlockKrylovSubspace: beta shape mismatch.");
  }
  host_basis_.setZero();
}

void BlockKrylovSubspace::build(FASTEIG_INT begin, FASTEIG_INT end) {
  if (begin < 0 || end < begin) {
    throw std::invalid_argument("core::BlockKrylovSubspace::build: invalid range.");
  }

  const FASTEIG_INT n = basis_.rows();
  const FASTEIG_INT b = static_cast<FASTEIG_INT>(initial_block_.cols());
  const FASTEIG_INT required_cols = (end + 1) * b;
  if (required_cols > basis_.cols()) {
    throw std::invalid_argument("core::BlockKrylovSubspace::build: range exceeds basis columns.");
  }

  if (begin == 0 && !initialized_) {
    Eigen::HouseholderQR<MatrixXd> qr_init(initial_block_);
    MatrixXd q0 = MatrixXd::Identity(static_cast<int>(n), static_cast<int>(b));
    q0 = qr_init.householderQ() * q0;
    host_basis_.block(0, 0, static_cast<int>(n), static_cast<int>(b)) = q0;
    copy_host_block_to_basis(0, q0);
    initialized_ = true;
  }

  MatrixXd AV(static_cast<int>(n), static_cast<int>(b));
  MatrixXd residual(static_cast<int>(n), static_cast<int>(b));

  for (FASTEIG_INT i = begin + 1; i <= end; ++i) {
    apply_block((i - 1) * b, b, AV);

    const FASTEIG_INT active_cols = i * b;
    MatrixXd H = host_basis_.leftCols(static_cast<int>(active_cols)).transpose() * AV;
    residual.noalias() = AV - host_basis_.leftCols(static_cast<int>(active_cols)) * H;

    MatrixXd S = host_basis_.leftCols(static_cast<int>(active_cols)).transpose() * residual;
    H.noalias() += S;
    residual.noalias() -= host_basis_.leftCols(static_cast<int>(active_cols)) * S;

    Eigen::HouseholderQR<MatrixXd> qr_residual(residual);
    MatrixXd q_new = MatrixXd::Identity(static_cast<int>(n), static_cast<int>(b));
    MatrixXd r_top =
        qr_residual.matrixQR().topLeftCorner(static_cast<int>(b), static_cast<int>(b))
            .template triangularView<Eigen::Upper>();
    q_new = qr_residual.householderQ() * q_new;
    host_basis_.block(0,
                      static_cast<int>(i * b),
                      static_cast<int>(n),
                      static_cast<int>(b)) = q_new;
    copy_host_block_to_basis(i * b, q_new);

    beta_ = r_top;
    projected_matrix_.block(static_cast<int>((i - 1) * b),
                            static_cast<int>((i - 1) * b),
                            static_cast<int>(b),
                            static_cast<int>(b)) = H.bottomRows(static_cast<int>(b));
    projected_matrix_.block(static_cast<int>(i * b),
                            static_cast<int>((i - 1) * b),
                            static_cast<int>(b),
                            static_cast<int>(b)) = beta_;
    projected_matrix_.block(static_cast<int>((i - 1) * b),
                            static_cast<int>(i * b),
                            static_cast<int>(b),
                            static_cast<int>(b)) = beta_.transpose();
  }
}

void BlockKrylovSubspace::apply_block(FASTEIG_INT first_col,
                                      FASTEIG_INT block_size,
                                      MatrixXd& output) const {
  if (first_col < 0 || block_size <= 0 || first_col + block_size > basis_.cols()) {
    throw std::invalid_argument("core::BlockKrylovSubspace::apply_block: invalid column range.");
  }
  if (output.rows() != op_.rows() || output.cols() != block_size) {
    output.resize(static_cast<int>(op_.rows()), static_cast<int>(block_size));
  }

  auto block_result = op_.executor().create_multivector(op_.rows(), block_size);
  fasteig::core::apply_block(op_, basis_, first_col, block_size, *block_result);
  output = block_result->copy_to_host();
}

void BlockKrylovSubspace::copy_host_block_to_basis(FASTEIG_INT first_col,
                                                   const MatrixXd& block) {
  if (first_col < 0 || first_col + block.cols() > basis_.cols() ||
      block.rows() != basis_.rows()) {
    throw std::invalid_argument("core::BlockKrylovSubspace::copy_host_block_to_basis: dimension mismatch.");
  }

  for (FASTEIG_INT j = 0; j < block.cols(); ++j) {
    auto column = basis_.executor().create_vector(block.col(static_cast<int>(j)));
    copy_column_from(basis_, first_col + j, *column);
  }
}

}  // namespace core
}  // namespace fasteig

