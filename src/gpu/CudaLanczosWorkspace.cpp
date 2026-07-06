#include "gpu/CudaLanczosWorkspace.hpp"

#ifdef USE_CUDA

#include "core/LinearOperator.hpp"
#include "gpu/CudaCheck.hpp"
#include "gpu/CudaContext.hpp"
#include "gpu/CudaExecutor.hpp"
#include "gpu/CudaLanczosKernels.hpp"
#include "gpu/CudaMultiVector.hpp"
#include "gpu/CudaUtils.hpp"
#include "gpu/CudaVector.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>

namespace fasteig {
namespace core {

CudaLanczosWorkspace::CudaLanczosWorkspace(const Executor& exec,
                                           FASTEIG_INT n,
                                           FASTEIG_INT maxiter,
                                           FASTEIG_INT nev)
    : exec_(exec), n_(n), maxiter_(maxiter), nev_(nev) {
  if (n <= 0 || maxiter <= 0 || nev <= 0) {
    throw std::invalid_argument("CudaLanczosWorkspace: dimensions must be positive.");
  }

  auto& context = cuda_executor_from(exec_, "CudaLanczosWorkspace").context();
  const auto maxiter_sz = static_cast<std::size_t>(maxiter_);
  const auto nev_sz = static_cast<std::size_t>(nev_);
  const auto n_sz = static_cast<std::size_t>(n_);

  FASTEIG_CUDA_CHECK(cudaMalloc(&d_T_, maxiter_sz * maxiter_sz * sizeof(double)));
  FASTEIG_CUDA_CHECK(cudaMalloc(&d_snapshot_T_, maxiter_sz * maxiter_sz * sizeof(double)));
  FASTEIG_CUDA_CHECK(cudaMalloc(&d_W_, maxiter_sz * sizeof(double)));
  FASTEIG_CUDA_CHECK(cudaMalloc(&d_dev_info_, sizeof(int)));
  FASTEIG_CUDA_CHECK(cudaMalloc(&d_Csel_, maxiter_sz * nev_sz * sizeof(double)));
  FASTEIG_CUDA_CHECK(cudaMalloc(&d_ritz_, n_sz * nev_sz * sizeof(double)));
  FASTEIG_CUDA_CHECK(cudaMemsetAsync(d_T_, 0, maxiter_sz * maxiter_sz * sizeof(double),
                                     context.stream()));

  FASTEIG_CUSOLVER_CHECK(cusolverDnCreate(&cusolver_handle_));
  FASTEIG_CUSOLVER_CHECK(cusolverDnSetStream(cusolver_handle_, context.stream()));

  int lwork = 0;
  FASTEIG_CUSOLVER_CHECK(cusolverDnDsyevd_bufferSize(
      cusolver_handle_, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
      static_cast<int>(maxiter_), d_snapshot_T_, static_cast<int>(maxiter_), d_W_, &lwork));
  lwork_ = lwork;
  FASTEIG_CUDA_CHECK(cudaMalloc(&d_work_, static_cast<std::size_t>(lwork_) * sizeof(double)));
  FASTEIG_CUDA_CHECK(cudaStreamSynchronize(context.stream()));
}

CudaLanczosWorkspace::~CudaLanczosWorkspace() {
  if (d_T_) cudaFree(d_T_);
  if (d_snapshot_T_) cudaFree(d_snapshot_T_);
  if (d_W_) cudaFree(d_W_);
  if (d_work_) cudaFree(d_work_);
  if (d_Csel_) cudaFree(d_Csel_);
  if (d_ritz_) cudaFree(d_ritz_);
  if (d_dev_info_) cudaFree(d_dev_info_);
  if (cusolver_handle_) cusolverDnDestroy(cusolver_handle_);
}

void CudaLanczosWorkspace::record_step(FASTEIG_INT it,
                                       double beta,
                                       double alpha,
                                       double beta_next) {
  if (it < 0 || it >= maxiter_) {
    throw std::invalid_argument("CudaLanczosWorkspace::record_step: iteration out of range.");
  }

  auto& context = cuda_executor_from(exec_, "CudaLanczosWorkspace::record_step").context();
  gpu::launch_set_lanczos_projected_matrix_entry(
      d_T_, static_cast<int>(maxiter_), static_cast<int>(it), beta, alpha, context.stream());
  beta_next_ = beta_next;
}

LanczosSnapshot CudaLanczosWorkspace::compute_snapshot(FASTEIG_INT m,
                                                       FASTEIG_INT nev,
                                                       const std::string& which,
                                                       bool need_vectors,
                                                       const char*) {
  if (m <= 0 || m > maxiter_) {
    throw std::invalid_argument("CudaLanczosWorkspace::compute_snapshot: invalid snapshot size.");
  }

  auto& context = cuda_executor_from(exec_, "CudaLanczosWorkspace::compute_snapshot").context();
  const int m_i = static_cast<int>(m);

  FASTEIG_CUDA_CHECK(cudaMemcpy2DAsync(
      d_snapshot_T_, static_cast<std::size_t>(m_i) * sizeof(double),
      d_T_, static_cast<std::size_t>(maxiter_) * sizeof(double),
      static_cast<std::size_t>(m_i) * sizeof(double),
      static_cast<std::size_t>(m_i),
      cudaMemcpyDeviceToDevice,
      context.stream()));

  FASTEIG_CUSOLVER_CHECK(cusolverDnDsyevd(
      cusolver_handle_, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
      m_i, d_snapshot_T_, m_i, d_W_, d_work_, lwork_, d_dev_info_));

  int dev_info = 0;
  VectorXd D(m_i);
  FASTEIG_CUDA_CHECK(cudaMemcpyAsync(&dev_info, d_dev_info_, sizeof(int),
                                     cudaMemcpyDeviceToHost, context.stream()));
  FASTEIG_CUDA_CHECK(cudaMemcpyAsync(D.data(), d_W_,
                                     static_cast<std::size_t>(m_i) * sizeof(double),
                                     cudaMemcpyDeviceToHost, context.stream()));

  MatrixXd C;
  VectorXd last_row;
  if (need_vectors) {
    C.resize(m_i, m_i);
    FASTEIG_CUDA_CHECK(cudaMemcpyAsync(C.data(), d_snapshot_T_,
                                       static_cast<std::size_t>(m_i) *
                                           static_cast<std::size_t>(m_i) * sizeof(double),
                                       cudaMemcpyDeviceToHost, context.stream()));
  } else {
    last_row.resize(m_i);
    FASTEIG_CUDA_CHECK(cudaMemcpy2DAsync(
        last_row.data(), sizeof(double),
        d_snapshot_T_ + (m_i - 1), static_cast<std::size_t>(m_i) * sizeof(double),
        sizeof(double), static_cast<std::size_t>(m_i),
        cudaMemcpyDeviceToHost,
        context.stream()));
  }
  FASTEIG_CUDA_CHECK(cudaStreamSynchronize(context.stream()));

  if (dev_info != 0) {
    throw std::runtime_error("CudaLanczosWorkspace::compute_snapshot: cuSOLVER syevd failed.");
  }

  if (!need_vectors) {
    return lanczos_make_snapshot_from_eigenvalues_last_row(D, last_row, nev, beta_next_, which);
  }
  return lanczos_make_snapshot_from_eigendecomp(D, C, nev, beta_next_, which, need_vectors);
}

void CudaLanczosWorkspace::finalize_eigenpairs(const LinearOperator& op,
                                               const MultiVector& basis,
                                               FASTEIG_INT active_cols,
                                               const LanczosSnapshot& snapshot,
                                               const std::string& which,
                                               FASTEIG_DOUBLE shift,
                                               VectorXd& eigvals,
                                               MatrixXd& eigvecs,
                                               VectorXd& residual_norms) {
  LanczosSnapshot final =
      lanczos_prepare_final_snapshot(snapshot, which, shift, "CudaLanczosWorkspace");

  if (active_cols <= 0 || active_cols > maxiter_) {
    throw std::invalid_argument("CudaLanczosWorkspace::finalize_eigenpairs: invalid active column count.");
  }
  if (final.Csel.rows() != active_cols) {
    throw std::invalid_argument("CudaLanczosWorkspace::finalize_eigenpairs: Csel row mismatch.");
  }
  if (final.Csel.cols() > nev_) {
    throw std::invalid_argument("CudaLanczosWorkspace::finalize_eigenpairs: Csel has too many columns.");
  }

  const auto* cuda_basis = dynamic_cast<const CudaMultiVector*>(&basis);
  if (!cuda_basis) {
    throw std::invalid_argument("CudaLanczosWorkspace::finalize_eigenpairs: basis must be a CudaMultiVector.");
  }

  auto& context = cuda_executor_from(exec_, "CudaLanczosWorkspace::finalize_eigenpairs").context();
  const int final_nev = static_cast<int>(final.Csel.cols());
  eigvals = final.lam;
  eigvecs.resize(static_cast<int>(n_), final_nev);
  residual_norms.resize(final_nev);
  if (final_nev == 0) return;

  FASTEIG_CUDA_CHECK(cudaMemcpyAsync(
      d_Csel_, final.Csel.data(),
      static_cast<std::size_t>(active_cols) * static_cast<std::size_t>(final_nev) * sizeof(double),
      cudaMemcpyHostToDevice, context.stream()));

  const double one = 1.0;
  const double zero = 0.0;
  FASTEIG_CUBLAS_CHECK(cublasDgemm(context.cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N,
                                   static_cast<int>(n_), final_nev, static_cast<int>(active_cols),
                                   &one, cuda_basis->data(), static_cast<int>(n_),
                                   d_Csel_, static_cast<int>(active_cols),
                                   &zero, d_ritz_, static_cast<int>(n_)));

  if (which == "sai") {
    residual_norms.setZero();
    FASTEIG_CUDA_CHECK(cudaMemcpyAsync(
        eigvecs.data(), d_ritz_,
        static_cast<std::size_t>(n_) * static_cast<std::size_t>(final_nev) * sizeof(double),
        cudaMemcpyDeviceToHost, context.stream()));
    FASTEIG_CUDA_CHECK(cudaStreamSynchronize(context.stream()));
    return;
  }

  auto x = op.executor().create_vector(n_);
  auto y = op.executor().create_vector(n_);
  auto* cuda_x = dynamic_cast<CudaVector*>(x.get());
  auto* cuda_y = dynamic_cast<CudaVector*>(y.get());
  if (!cuda_x || !cuda_y) {
    throw std::invalid_argument(
        "CudaLanczosWorkspace::finalize_eigenpairs: residual workspace must be CUDA vectors.");
  }

  for (int j = 0; j < final_nev; ++j) {
    const double minus_lambda = -eigvals(j);
    const double* d_ritz_col = d_ritz_ + static_cast<std::size_t>(j) * static_cast<std::size_t>(n_);
    FASTEIG_CUBLAS_CHECK(cublasDcopy(context.cublas_handle(), static_cast<int>(n_),
                                     d_ritz_col, 1, cuda_x->data(), 1));
    op.apply(*cuda_x, *cuda_y);
    FASTEIG_CUBLAS_CHECK(cublasDaxpy(context.cublas_handle(), static_cast<int>(n_),
                                     &minus_lambda, cuda_x->data(), 1, cuda_y->data(), 1));
    FASTEIG_CUBLAS_CHECK(cublasDnrm2(context.cublas_handle(), static_cast<int>(n_),
                                     cuda_y->data(), 1, &residual_norms(j)));
  }

  FASTEIG_CUDA_CHECK(cudaMemcpyAsync(
      eigvecs.data(), d_ritz_,
      static_cast<std::size_t>(n_) * static_cast<std::size_t>(final_nev) * sizeof(double),
      cudaMemcpyDeviceToHost, context.stream()));
  FASTEIG_CUDA_CHECK(cudaStreamSynchronize(context.stream()));
}

}  // namespace core
}  // namespace fasteig

#endif  // USE_CUDA

