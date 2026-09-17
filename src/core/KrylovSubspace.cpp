#include "core/KrylovSubspace.hpp"

#include "core/Executor.hpp"
#include "core/LanczosWorkspace.hpp"
#include "core/LinearOperator.hpp"
#include "core/MultiVector.hpp"
#include "core/Operations.hpp"
#include "core/Orthogonalization.hpp"
#include "core/Vector.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace fasteig {
namespace core {

KrylovSubspace::KrylovSubspace(const LinearOperator& op,
                               Vector& residual,
                               Vector& operator_result,
                               MultiVector& basis,
                               unsigned int restart_seed,
                               const LinearOperator* metric)
    : op_(op),
      metric_(metric),
      residual_(residual),
      operator_result_(operator_result),
      basis_(basis),
      lanczos_projection_(nullptr),
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
  if (metric_) {
    if (&metric_->executor() != &op_.executor() || metric_->rows() != residual_.size() ||
        metric_->cols() != residual_.size()) {
      throw std::invalid_argument("core::KrylovSubspace: metric dimension/executor mismatch.");
    }
    metric_basis_ = basis_.executor().create_multivector(basis_.rows(), basis_.cols());
    metric_residual_ = residual_.executor().create_vector(residual_.size());
  }
}

KrylovSubspace::~KrylovSubspace() = default;

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
  const double beta = normalize_residual_into_basis(it);
  apply_operator_to_basis_column(it);
  const double alpha = orthogonalize_operator_result(it + 1);
  const double beta_next = residual_norm();
  record_step(it, begin, end, beta, alpha, beta_next);
}

double KrylovSubspace::normalize_residual_into_basis(FASTEIG_INT basis_col) {
  if (basis_col < 0 || basis_col >= basis_.cols()) {
    throw std::invalid_argument("core::KrylovSubspace::normalize_residual_into_basis: column out of range.");
  }
  const double beta = normalize_or_restart(basis_col);
  scal(1.0 / beta, residual_);
  copy_column_from(basis_, basis_col, residual_);
  if (metric_) {
    // normalize_or_restart refreshed B*f; scale it with f and cache B*q.
    scal(1.0 / beta, *metric_residual_);
    copy_column_from(*metric_basis_, basis_col, *metric_residual_);
  }
  // An invariant subspace has zero coupling to a replacement random vector.
  // Its normalization factor must not be recorded as a Lanczos off-diagonal.
  return restarted_column_ ? 0.0 : beta;
}

void KrylovSubspace::apply_operator_to_basis_column(FASTEIG_INT basis_col) {
  if (basis_col < 0 || basis_col >= basis_.cols()) {
    throw std::invalid_argument("core::KrylovSubspace::apply_operator_to_basis_column: column out of range.");
  }
  apply_column(op_, basis_, basis_col, operator_result_);
}

double KrylovSubspace::orthogonalize_operator_result(FASTEIG_INT active_cols) {
  return fasteig::core::orthogonalize_active_columns(
      basis_, active_cols, operator_result_, residual_, metric_basis_.get());
}

double KrylovSubspace::residual_norm() const {
  double beta_next = 0.0;
  if (metric_) {
    apply(*metric_, residual_, *metric_residual_);
    const double squared = dot(residual_, *metric_residual_);
    if (!std::isfinite(squared) || squared < 0.0) {
      throw std::runtime_error("core::KrylovSubspace: B must be positive definite (invalid B-norm).");
    }
    beta_next = std::sqrt(squared);
  } else {
    beta_next = norm2(residual_);
  }
  if (!std::isfinite(beta_next)) {
    throw std::runtime_error("core::KrylovSubspace: non-finite next residual norm encountered.");
  }
  return beta_next;
}

void KrylovSubspace::rotate_basis(
    FASTEIG_INT active_cols,
    const Eigen::Ref<const MatrixXd>& coefficients) {
  if (active_cols <= 0 || active_cols > basis_.cols()) {
    throw std::invalid_argument("core::KrylovSubspace::rotate_basis: invalid active column count.");
  }
  if (coefficients.rows() != static_cast<int>(active_cols) ||
      coefficients.cols() <= 0 || coefficients.cols() > basis_.cols()) {
    throw std::invalid_argument("core::KrylovSubspace::rotate_basis: coefficient dimension mismatch.");
  }

  const FASTEIG_INT retained_cols = static_cast<FASTEIG_INT>(coefficients.cols());
  auto rotated = basis_.executor().create_multivector(basis_.rows(), retained_cols);
  basis_.right_multiply_active_columns(active_cols, coefficients, *rotated);
  basis_.copy_columns_from(0, *rotated, 0, retained_cols);
  if (metric_basis_) {
    metric_basis_->right_multiply_active_columns(active_cols, coefficients, *rotated);
    metric_basis_->copy_columns_from(0, *rotated, 0, retained_cols);
  }
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
  restarted_column_ = false;
  double beta = residual_norm();
  if (!std::isfinite(beta)) {
    throw std::runtime_error("core::KrylovSubspace: non-finite residual norm encountered.");
  }
  if (beta > std::numeric_limits<double>::epsilon()) {
    return beta;
  }

  restart_residual(it);
  restarted_column_ = true;
  beta = residual_norm();
  if (beta <= std::numeric_limits<double>::epsilon()) {
    throw std::runtime_error("core::KrylovSubspace: breakdown, unable to generate new basis vector.");
  }
  return beta;
}

void KrylovSubspace::restart_residual(FASTEIG_INT it) {
  auto candidate =
      residual_.executor().create_random_vector(residual_.size(),
                                                restart_seed_ + static_cast<unsigned int>(it + 1));
  // Two-pass projection also handles random replacement after breakdown.
  // With a metric, coefficients use (B*Q)^T while subtraction uses Q.
  fasteig::core::orthogonalize_active_columns(
      basis_, it, *candidate, residual_, metric_basis_.get());
}

BlockKrylovSubspace::BlockKrylovSubspace(const LinearOperator& op,
                                         const MultiVector& initial_block,
                                         MultiVector& basis,
                                         const LinearOperator* metric,
                                         unsigned int restart_seed)
    : op_(op),
      metric_(metric),
      restart_seed_(restart_seed),
      initial_block_(initial_block),
      basis_(basis) {
  if (&op_.executor() != &basis_.executor() ||
      &op_.executor() != &initial_block_.executor()) {
    throw std::invalid_argument("core::BlockKrylovSubspace: executor mismatch.");
  }
  if (initial_block.rows() != basis.rows()) {
    throw std::invalid_argument("core::BlockKrylovSubspace: initial block row count mismatch.");
  }
  if (initial_block.cols() <= 0) {
    throw std::invalid_argument("core::BlockKrylovSubspace: block size must be positive.");
  }
  if (metric_) {
    if (&metric_->executor() != &basis_.executor() || metric_->rows() != basis_.rows() ||
        metric_->cols() != basis_.rows()) {
      throw std::invalid_argument("core::BlockKrylovSubspace: metric dimension/executor mismatch.");
    }
    metric_basis_ = basis_.executor().create_multivector(basis_.rows(), basis_.cols());
    metric_block_ = basis_.executor().create_multivector(basis_.rows(), initial_block_.cols());
  }
  qr_block_ = basis_.executor().create_multivector(basis_.rows(), initial_block_.cols());
  operator_block_ = op_.executor().create_multivector(op_.rows(), initial_block_.cols());
  residual_block_ = op_.executor().create_multivector(op_.rows(), initial_block_.cols());
}

void BlockKrylovSubspace::initialize_basis_block() {
  if (initialized_) return;

  const FASTEIG_INT n = basis_.rows();
  const FASTEIG_INT b = initial_block_.cols();
  if (initial_block_.cols() != b || initial_block_.rows() != n) {
    throw std::invalid_argument("core::BlockKrylovSubspace::initialize_basis_block: initial block mismatch.");
  }

  MatrixXd R;
  complete_block_qr(basis_, 0, initial_block_, *residual_block_, R,
                    metric_, metric_basis_.get(), restart_seed_, 0.0);
  copy_block_to_basis(0, *residual_block_);
  initialized_ = true;
}

void BlockKrylovSubspace::apply_operator_to_basis_block(FASTEIG_INT block_index) {
  const FASTEIG_INT b = initial_block_.cols();
  if (!initialized_) {
    throw std::logic_error("core::BlockKrylovSubspace::apply_operator_to_basis_block: basis not initialized.");
  }
  if (block_index < 0 || (block_index + 1) * b > basis_.cols()) {
    throw std::invalid_argument("core::BlockKrylovSubspace::apply_operator_to_basis_block: block out of range.");
  }
  fasteig::core::apply_block(op_, basis_, block_index * b, b, *operator_block_);
}

void BlockKrylovSubspace::orthogonalize_operator_block(FASTEIG_INT active_cols) {
  const FASTEIG_INT b = initial_block_.cols();
  if (active_cols <= 0 || active_cols > basis_.cols() || (active_cols % b != 0 && active_cols != basis_.rows())) {
    throw std::invalid_argument("core::BlockKrylovSubspace::orthogonalize_operator_block: invalid active column count.");
  }

  MatrixXd H;
  const FASTEIG_INT info = block_gram_schmidt(basis_, active_cols, *operator_block_,
                                              *residual_block_, H, metric_basis_.get());
  if (info != 0) {
    throw std::runtime_error("core::BlockKrylovSubspace::orthogonalize_operator_block: block Gram-Schmidt failed.");
  }
  const FASTEIG_INT width = 1 + (active_cols - 1) % b;
  diagonal_block_.setZero(b, b);
  diagonal_block_.topLeftCorner(width, width) = H.bottomRows(width).leftCols(width);
}

void BlockKrylovSubspace::qr_residual_block(FASTEIG_INT next_block_index) {
  const FASTEIG_INT b = initial_block_.cols();
  if (next_block_index < 0 || (next_block_index + 1) * b > basis_.cols()) {
    throw std::invalid_argument("core::BlockKrylovSubspace::qr_residual_block: block out of range.");
  }

  double scale = 0.0;
  if (metric_) {
    metric_->apply_block(*operator_block_, 0, b, *metric_block_);
    const MatrixXd gram = operator_block_->transpose_multiply(b, *metric_block_, b);
    scale = std::sqrt(std::max(0.0, gram.trace()));
  } else {
    scale = operator_block_->column_norms().norm();
  }
  // Reuse the rank-aware QR for either inner product. This also handles a
  // partial final block or an invariant subspace without normalizing roundoff.
  const bool replenished = complete_block_qr(
      basis_, std::min(next_block_index * b, basis_.rows()), *residual_block_, *qr_block_,
      coupling_block_, metric_, metric_basis_.get(),
      restart_seed_ + static_cast<unsigned int>(next_block_index * b), scale);
  replenished_directions_ = replenished_directions_ || replenished;
  residual_block_->copy_from(*qr_block_);
  copy_block_to_basis(next_block_index * b, *residual_block_);
}

void BlockKrylovSubspace::restart_with_ritz_block(
    FASTEIG_INT active_cols,
    const Eigen::Ref<const MatrixXd>& coefficients,
    FASTEIG_INT residual_block_index) {
  const FASTEIG_INT b = initial_block_.cols();
  const FASTEIG_INT retained = coefficients.cols();
  if (!initialized_) {
    throw std::logic_error(
        "core::BlockKrylovSubspace::restart_with_ritz_block: basis not initialized.");
  }
  if (active_cols <= 0 || active_cols > basis_.cols() ||
      coefficients.rows() != static_cast<int>(active_cols) ||
      retained <= 0 || retained % b != 0 || retained >= active_cols ||
      retained + b > basis_.cols()) {
    throw std::invalid_argument(
        "core::BlockKrylovSubspace::restart_with_ritz_block: coefficient mismatch.");
  }
  if (residual_block_index < 0 ||
      (residual_block_index + 1) * b > basis_.cols()) {
    throw std::invalid_argument(
        "core::BlockKrylovSubspace::restart_with_ritz_block: residual block out of range.");
  }

  if (!restart_ritz_block_ || restart_ritz_block_->cols() != retained) {
    restart_ritz_block_ = basis_.executor().create_multivector(basis_.rows(), retained);
    restart_residual_block_ = basis_.executor().create_multivector(basis_.rows(), b);
  }

  // Form V*C and preserve the trailing residual basis block before clearing V.
  basis_.right_multiply_active_columns(active_cols, coefficients,
                                       *restart_ritz_block_);
  restart_residual_block_->copy_columns_from(
      0, basis_, residual_block_index * b, b);

  if (metric_) {
    if (!restart_metric_ritz_block_ || restart_metric_ritz_block_->cols() != retained) {
      restart_metric_ritz_block_ = basis_.executor().create_multivector(basis_.rows(), retained);
      restart_metric_residual_block_ = basis_.executor().create_multivector(basis_.rows(), b);
    }
    metric_basis_->right_multiply_active_columns(active_cols, coefficients,
                                                 *restart_metric_ritz_block_);
    restart_metric_residual_block_->copy_columns_from(
        0, *metric_basis_, residual_block_index * b, b);
  }
  basis_.set_zero();
  basis_.copy_columns_from(0, *restart_ritz_block_, 0, retained);
  basis_.copy_columns_from(retained, *restart_residual_block_, 0, b);
  if (metric_) {
    // Rotate Q and BQ together; preserve the trailing metric residual block.
    metric_basis_->set_zero();
    metric_basis_->copy_columns_from(0, *restart_metric_ritz_block_, 0, retained);
    metric_basis_->copy_columns_from(retained, *restart_metric_residual_block_, 0, b);
  }
}

void BlockKrylovSubspace::copy_block_to_basis(FASTEIG_INT first_col,
                                              const MultiVector& block) {
  if (&basis_.executor() != &block.executor()) {
    throw std::invalid_argument("core::BlockKrylovSubspace::copy_block_to_basis: executor mismatch.");
  }
  if (first_col < 0 || first_col + block.cols() > basis_.cols() ||
      block.rows() != basis_.rows()) {
    throw std::invalid_argument("core::BlockKrylovSubspace::copy_block_to_basis: dimension mismatch.");
  }

  basis_.copy_columns_from(first_col, block, 0, block.cols());
  if (metric_) {
    metric_->apply_block(block, 0, block.cols(), *metric_block_);
    metric_basis_->copy_columns_from(first_col, *metric_block_, 0, block.cols());
  }
}

}  // namespace core
}  // namespace fasteig

