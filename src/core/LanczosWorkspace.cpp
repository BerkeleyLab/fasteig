#include "core/LanczosWorkspace.hpp"

#include "core/Executor.hpp"
#include "core/LinearOperator.hpp"
#include "core/MultiVector.hpp"
#include "core/Operations.hpp"
#include "core/Vector.hpp"
#ifdef USE_CUDA
#include "gpu/CudaLanczosWorkspace.hpp"
#endif

#include <stdexcept>

namespace fasteig {
namespace core {

CpuLanczosWorkspace::CpuLanczosWorkspace(FASTEIG_INT maxiter)
    : T_(static_cast<int>(maxiter), static_cast<int>(maxiter)) {
  if (maxiter <= 0) {
    throw std::invalid_argument("CpuLanczosWorkspace: maxiter must be positive.");
  }
  T_.setZero();
}

void CpuLanczosWorkspace::record_step(FASTEIG_INT it,
                                      double beta,
                                      double alpha,
                                      double beta_next) {
  if (it < 0 || it >= T_.rows() || it >= T_.cols()) {
    throw std::invalid_argument("CpuLanczosWorkspace::record_step: iteration out of range.");
  }

  if (it != 0) {
    T_(static_cast<int>(it), static_cast<int>(it - 1)) = beta;
    T_(static_cast<int>(it - 1), static_cast<int>(it)) = beta;
  }
  T_(static_cast<int>(it), static_cast<int>(it)) = alpha;
  beta_next_ = beta_next;
}

void CpuLanczosWorkspace::record_block_step(FASTEIG_INT block_index,
                                            FASTEIG_INT block_size,
                                            const MatrixXd& diagonal_block,
                                            const MatrixXd& beta_block) {
  if (block_index < 0 || block_size <= 0) {
    throw std::invalid_argument("CpuLanczosWorkspace::record_block_step: invalid block index/size.");
  }
  if (diagonal_block.rows() != block_size || diagonal_block.cols() != block_size ||
      beta_block.rows() != block_size || beta_block.cols() != block_size) {
    throw std::invalid_argument("CpuLanczosWorkspace::record_block_step: block shape mismatch.");
  }

  const FASTEIG_INT row = block_index * block_size;
  const FASTEIG_INT next_row = row + block_size;
  if (next_row + block_size > T_.rows() || next_row + block_size > T_.cols()) {
    throw std::invalid_argument("CpuLanczosWorkspace::record_block_step: projected matrix too small.");
  }

  T_.block(static_cast<int>(row),
           static_cast<int>(row),
           static_cast<int>(block_size),
           static_cast<int>(block_size)) = diagonal_block;
  T_.block(static_cast<int>(next_row),
           static_cast<int>(row),
           static_cast<int>(block_size),
           static_cast<int>(block_size)) = beta_block;
  T_.block(static_cast<int>(row),
           static_cast<int>(next_row),
           static_cast<int>(block_size),
           static_cast<int>(block_size)) = beta_block.transpose();
  beta_block_ = beta_block;
}

LanczosSnapshot CpuLanczosWorkspace::compute_snapshot(FASTEIG_INT m,
                                                      FASTEIG_INT nev,
                                                      const std::string& which,
                                                      bool need_vectors,
                                                      const char* solver_name) {
  return lanczos_compute_cpu_snapshot(T_, m, nev, beta_next_, which, need_vectors, solver_name);
}

LanczosSnapshot CpuLanczosWorkspace::compute_block_snapshot(FASTEIG_INT block_count,
                                                            FASTEIG_INT block_size,
                                                            FASTEIG_INT nev,
                                                            const std::string& which,
                                                            bool need_vectors,
                                                            const char* solver_name) {
  return block_lanczos_compute_cpu_snapshot(T_, block_count, block_size, nev, beta_block_,
                                            which, need_vectors, solver_name);
}

void CpuLanczosWorkspace::finalize_eigenpairs(const LinearOperator& op,
                                              const MultiVector& basis,
                                              FASTEIG_INT active_cols,
                                              const LanczosSnapshot& snapshot,
                                              const std::string& which,
                                              FASTEIG_DOUBLE shift,
                                              VectorXd& eigvals,
                                              MatrixXd& eigvecs,
                                              VectorXd& residual_norms) {
  MatrixXd host_basis = basis.copy_to_host();
  LanczosSnapshot final =
      lanczos_prepare_final_snapshot(snapshot, which, shift, "CpuLanczosWorkspace");

  if (active_cols < 0 || active_cols > host_basis.cols()) {
    throw std::invalid_argument("CpuLanczosWorkspace::finalize_eigenpairs: invalid active column count.");
  }
  if (final.Csel.rows() != active_cols) {
    throw std::invalid_argument("CpuLanczosWorkspace::finalize_eigenpairs: Csel row mismatch.");
  }

  eigvals = final.lam;
  eigvecs.noalias() = host_basis.leftCols(static_cast<int>(active_cols)) * final.Csel;
  residual_norms.resize(eigvals.size());
  if (which == "sai") {
    residual_norms.setZero();
    return;
  }

  auto x = basis.executor().create_vector(op.cols());
  auto y = basis.executor().create_vector(op.rows());
  for (int j = 0; j < eigvals.size(); ++j) {
    x->copy_from_host(eigvecs.col(j));
    apply(op, *x, *y);
    axpy(-eigvals(j), *x, *y);
    residual_norms(j) = norm2(*y);
  }
}

std::unique_ptr<LanczosWorkspace> make_lanczos_workspace(const Executor& exec,
                                                         FASTEIG_INT n,
                                                         FASTEIG_INT maxiter,
                                                         FASTEIG_INT nev) {
  if (exec.kind() == ExecutorKind::Cpu) {
    (void)n;
    (void)nev;
    return std::make_unique<CpuLanczosWorkspace>(maxiter);
  }

#ifdef USE_CUDA
  if (exec.kind() == ExecutorKind::Cuda) {
    return std::make_unique<CudaLanczosWorkspace>(exec, n, maxiter, nev);
  }
#endif

  throw std::invalid_argument("make_lanczos_workspace: unsupported executor.");
}

}  // namespace core
}  // namespace fasteig

