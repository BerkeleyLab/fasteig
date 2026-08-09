#include "core/KrylovSubspace.hpp"

#include "core/Executor.hpp"
#include "core/LanczosWorkspace.hpp"
#include "core/LinearOperator.hpp"
#include "core/MultiVector.hpp"
#include "core/Operations.hpp"
#include "core/Orthogonalization.hpp"
#include "core/Vector.hpp"

#ifdef USE_CUDA
#include "gpu/CudaCheck.hpp"
#include "gpu/CudaContext.hpp"
#include "gpu/CudaMultiVector.hpp"
#include "gpu/CudaUtils.hpp"

#include <cuda_runtime.h>
#endif

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
                               unsigned int restart_seed)
    : op_(op),
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
}

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
  return beta;
}

void KrylovSubspace::apply_operator_to_basis_column(FASTEIG_INT basis_col) {
  if (basis_col < 0 || basis_col >= basis_.cols()) {
    throw std::invalid_argument("core::KrylovSubspace::apply_operator_to_basis_column: column out of range.");
  }
  apply_column(op_, basis_, basis_col, operator_result_);
}

double KrylovSubspace::orthogonalize_operator_result(FASTEIG_INT active_cols) {
  return fasteig::core::orthogonalize_active_columns(
      basis_, active_cols, operator_result_, residual_);
}

double KrylovSubspace::residual_norm() const {
  const double beta_next = norm2(residual_);
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
                                         const MultiVector& initial_block,
                                         MultiVector& basis)
    : op_(op),
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
  operator_block_ = op_.executor().create_multivector(op_.rows(), initial_block_.cols());
  residual_block_ = op_.executor().create_multivector(op_.rows(), initial_block_.cols());
}

BlockKrylovSubspace::BlockKrylovSubspace(const LinearOperator& op,
                                         const MultiVector& initial_block,
                                         MultiVector& basis,
                                         LanczosWorkspace& projection)
    : BlockKrylovSubspace(op, initial_block, basis) {
  projection_ = &projection;
}

void BlockKrylovSubspace::build(FASTEIG_INT begin, FASTEIG_INT end) {
  if (begin < 0 || end < begin) {
    throw std::invalid_argument("core::BlockKrylovSubspace::build: invalid range.");
  }

  const FASTEIG_INT b = initial_block_.cols();
  const FASTEIG_INT required_cols = (end + 1) * b;
  if (required_cols > basis_.cols()) {
    throw std::invalid_argument("core::BlockKrylovSubspace::build: range exceeds basis columns.");
  }

  if (begin == 0) {
    initialize_basis_block();
  }

  for (FASTEIG_INT i = begin + 1; i <= end; ++i) {
    apply_operator_to_basis_block(i - 1);
    orthogonalize_operator_block(i * b);
    qr_residual_block(i);
    if (!projection_) {
      throw std::logic_error("core::BlockKrylovSubspace::build: no projection target configured.");
    }
    projection_->record_block_step(i - 1, b, diagonal_block_, coupling_block_);
  }
}

void BlockKrylovSubspace::initialize_basis_block() {
  if (initialized_) return;

  const FASTEIG_INT n = basis_.rows();
  const FASTEIG_INT b = initial_block_.cols();
  if (initial_block_.cols() != b || initial_block_.rows() != n) {
    throw std::invalid_argument("core::BlockKrylovSubspace::initialize_basis_block: initial block mismatch.");
  }

  residual_block_->copy_from(initial_block_);
  MatrixXd R;
  const FASTEIG_INT info = thin_qr_in_place(*residual_block_, R, QrMethod::Cholesky);
  if (info != 0) {
    throw std::runtime_error("core::BlockKrylovSubspace::initialize_basis_block: initial QR failed.");
  }
  copy_block_to_basis(0, *residual_block_);
  initialized_ = true;
}

void BlockKrylovSubspace::set_basis_block(FASTEIG_INT block_index,
                                          const MultiVector& block) {
  const FASTEIG_INT b = initial_block_.cols();
  if (block_index < 0 || block.cols() != b || block.rows() != basis_.rows()) {
    throw std::invalid_argument("core::BlockKrylovSubspace::set_basis_block: block mismatch.");
  }
  copy_block_to_basis(block_index * b, block);
  if (block_index == 0) initialized_ = true;
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
  if (active_cols <= 0 || active_cols > basis_.cols() || active_cols % b != 0) {
    throw std::invalid_argument("core::BlockKrylovSubspace::orthogonalize_operator_block: invalid active column count.");
  }

  MatrixXd H;
  const FASTEIG_INT info = block_gram_schmidt(basis_, active_cols, *operator_block_,
                                              *residual_block_, H);
  if (info != 0) {
    throw std::runtime_error("core::BlockKrylovSubspace::orthogonalize_operator_block: block Gram-Schmidt failed.");
  }
  diagonal_block_ = H.bottomRows(static_cast<int>(b));
}

void BlockKrylovSubspace::qr_residual_block(FASTEIG_INT next_block_index) {
  const FASTEIG_INT b = initial_block_.cols();
  if (next_block_index < 0 || (next_block_index + 1) * b > basis_.cols()) {
    throw std::invalid_argument("core::BlockKrylovSubspace::qr_residual_block: block out of range.");
  }

  FASTEIG_INT info = thin_qr_in_place(*residual_block_, coupling_block_, QrMethod::Cholesky);
  if (info != 0) {
    throw std::runtime_error("core::BlockKrylovSubspace::qr_residual_block: block QR failed.");
  }
  copy_block_to_basis(next_block_index * b, *residual_block_);
}

MatrixXd BlockKrylovSubspace::ritz_vectors(
    FASTEIG_INT active_cols,
    const Eigen::Ref<const MatrixXd>& coefficients) const {
  if (active_cols < 0 || active_cols > basis_.cols()) {
    throw std::invalid_argument("core::BlockKrylovSubspace::ritz_vectors: invalid active column count.");
  }
  if (coefficients.rows() != static_cast<int>(active_cols)) {
    throw std::invalid_argument("core::BlockKrylovSubspace::ritz_vectors: coefficient row mismatch.");
  }
  return basis_.multiply_active_columns(active_cols, coefficients);
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

  if (auto* cpu_basis = dynamic_cast<CpuMultiVector*>(&basis_)) {
    const auto* cpu_block = dynamic_cast<const CpuMultiVector*>(&block);
    if (!cpu_block) {
      throw std::invalid_argument("core::BlockKrylovSubspace::copy_block_to_basis: CPU object mismatch.");
    }
    cpu_basis->values().block(0,
                              static_cast<int>(first_col),
                              static_cast<int>(block.rows()),
                              static_cast<int>(block.cols())) = cpu_block->values();
    return;
  }

#ifdef USE_CUDA
  if (auto* cuda_basis = dynamic_cast<CudaMultiVector*>(&basis_)) {
    const auto* cuda_block = dynamic_cast<const CudaMultiVector*>(&block);
    if (!cuda_block) {
      throw std::invalid_argument("core::BlockKrylovSubspace::copy_block_to_basis: CUDA object mismatch.");
    }
    const std::size_t offset =
        static_cast<std::size_t>(first_col) * static_cast<std::size_t>(basis_.rows());
    const std::size_t count =
        static_cast<std::size_t>(block.rows()) * static_cast<std::size_t>(block.cols());
    auto& context = cuda_executor_from(basis_.executor(),
                                       "core::BlockKrylovSubspace::copy_block_to_basis").context();
    FASTEIG_CUDA_CHECK(cudaMemcpyAsync(cuda_basis->data() + offset,
                                       cuda_block->data(),
                                       count * sizeof(double),
                                       cudaMemcpyDeviceToDevice,
                                       context.stream()));
    return;
  }
#endif

  throw std::invalid_argument("core::BlockKrylovSubspace::copy_block_to_basis: unsupported backend.");
}

}  // namespace core
}  // namespace fasteig

