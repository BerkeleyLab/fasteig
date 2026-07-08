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
                                         const MultiVector& initial_block,
                                         MultiVector& basis,
                                         LanczosWorkspace& projection)
    : op_(op),
      initial_block_(initial_block),
      basis_(basis),
      projection_(projection) {
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
}

void BlockKrylovSubspace::build(FASTEIG_INT begin, FASTEIG_INT end) {
  if (begin < 0 || end < begin) {
    throw std::invalid_argument("core::BlockKrylovSubspace::build: invalid range.");
  }

  const FASTEIG_INT n = basis_.rows();
  const FASTEIG_INT b = initial_block_.cols();
  const FASTEIG_INT required_cols = (end + 1) * b;
  if (required_cols > basis_.cols()) {
    throw std::invalid_argument("core::BlockKrylovSubspace::build: range exceeds basis columns.");
  }

  if (begin == 0) {
    initialize_basis_block(n, b);
  }

  for (FASTEIG_INT i = begin + 1; i <= end; ++i) {
    auto block_result = op_.executor().create_multivector(op_.rows(), b);
    auto residual = op_.executor().create_multivector(op_.rows(), b);

    fasteig::core::apply_block(op_, basis_, (i - 1) * b, b, *block_result);

    const FASTEIG_INT active_cols = i * b;
    MatrixXd H;
    MatrixXd beta;
    FASTEIG_INT info = block_gram_schmidt(basis_, active_cols, *block_result,
                                          *residual, H);
    if (info != 0) {
      throw std::runtime_error("core::BlockKrylovSubspace::build: block Gram-Schmidt failed.");
    }

    info = thin_qr_in_place(*residual, beta, QrMethod::Cholesky2);
    if (info != 0) {
      throw std::runtime_error("core::BlockKrylovSubspace::build: block QR failed.");
    }

    copy_block_to_basis(i * b, *residual);

    const MatrixXd diagonal_block = H.bottomRows(static_cast<int>(b));
    projection_.record_block_step(i - 1, b, diagonal_block, beta);
  }
}

void BlockKrylovSubspace::initialize_basis_block(FASTEIG_INT n, FASTEIG_INT b) {
  if (initialized_) return;

  if (initial_block_.cols() != b || initial_block_.rows() != n) {
    throw std::invalid_argument("core::BlockKrylovSubspace::initialize_basis_block: initial block mismatch.");
  }

  auto initial_q = op_.executor().create_multivector(n, b);
  initial_q->copy_from(initial_block_);
  MatrixXd R;
  const FASTEIG_INT info = thin_qr_in_place(*initial_q, R, QrMethod::Cholesky2);
  if (info != 0) {
    throw std::runtime_error("core::BlockKrylovSubspace::initialize_basis_block: initial QR failed.");
  }
  copy_block_to_basis(0, *initial_q);
  initialized_ = true;
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

