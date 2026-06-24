#include "gpu/CudaContext.hpp"

#ifdef USE_CUDA

#include "gpu/CudaCheck.hpp"

namespace fasteig::gpu {

CudaContext::CudaContext() {
  FASTEIG_CUDA_CHECK(cudaStreamCreate(&stream_));
  FASTEIG_CUBLAS_CHECK(cublasCreate(&cublas_handle_));
  FASTEIG_CUBLAS_CHECK(cublasSetStream(cublas_handle_, stream_));
  FASTEIG_CUSPARSE_CHECK(cusparseCreate(&cusparse_handle_));
  FASTEIG_CUSPARSE_CHECK(cusparseSetStream(cusparse_handle_, stream_));
}

CudaContext::~CudaContext() {
  if (cusparse_handle_) cusparseDestroy(cusparse_handle_);
  if (cublas_handle_) cublasDestroy(cublas_handle_);
  if (stream_) cudaStreamDestroy(stream_);
}

}  // namespace fasteig::gpu

#endif  // USE_CUDA

