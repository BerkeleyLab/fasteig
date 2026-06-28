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

LanczosSnapshot CpuLanczosWorkspace::compute_snapshot(FASTEIG_INT m,
                                                      FASTEIG_INT nev,
                                                      const std::string& which,
                                                      bool need_vectors,
                                                      const char* solver_name) {
  return lanczos_compute_cpu_snapshot(T_, m, nev, beta_next_, which, need_vectors, solver_name);
}

void CpuLanczosWorkspace::finalize_eigenpairs(const LinearOperator& op,
                                              const MultiVector& basis,
                                              const MatrixXd& host_basis,
                                              FASTEIG_INT active_cols,
                                              const LanczosSnapshot& snapshot,
                                              VectorXd& eigvals,
                                              MatrixXd& eigvecs,
                                              VectorXd& residual_norms) {
  if (active_cols < 0 || active_cols > host_basis.cols()) {
    throw std::invalid_argument("CpuLanczosWorkspace::finalize_eigenpairs: invalid active column count.");
  }
  if (snapshot.Csel.rows() != active_cols) {
    throw std::invalid_argument("CpuLanczosWorkspace::finalize_eigenpairs: Csel row mismatch.");
  }

  eigvals = snapshot.lam;
  eigvecs.noalias() = host_basis.leftCols(static_cast<int>(active_cols)) * snapshot.Csel;
  residual_norms.resize(eigvals.size());

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

