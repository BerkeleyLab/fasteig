#pragma once

#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cusparse.h>

namespace fasteig::gpu {

class CudaContext {
public:
  CudaContext();
  ~CudaContext();

  CudaContext(const CudaContext&) = delete;
  CudaContext& operator=(const CudaContext&) = delete;

  cudaStream_t stream() const noexcept { return stream_; }
  cublasHandle_t cublas_handle() const noexcept { return cublas_handle_; }
  cusolverDnHandle_t cusolver_handle() const noexcept { return cusolver_handle_; }
  cusparseHandle_t cusparse_handle() const noexcept { return cusparse_handle_; }

private:
  cudaStream_t stream_ = nullptr;
  cublasHandle_t cublas_handle_ = nullptr;
  cusolverDnHandle_t cusolver_handle_ = nullptr;
  cusparseHandle_t cusparse_handle_ = nullptr;
};

}  // namespace fasteig::gpu

#endif  // USE_CUDA

