#pragma once

#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cusparse.h>

#include <stdexcept>
#include <string>

namespace fasteig::gpu {

inline void check_cuda(cudaError_t status, const char* call, const char* file, int line) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(call) + " failed at " + file + ":" +
                             std::to_string(line) + ": " + cudaGetErrorString(status));
  }
}

inline void check_cusparse(cusparseStatus_t status, const char* call, const char* file, int line) {
  if (status != CUSPARSE_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(call) + " failed at " + file + ":" +
                             std::to_string(line) + ": cuSPARSE status " +
                             std::to_string(static_cast<int>(status)));
  }
}

inline void check_cublas(cublasStatus_t status, const char* call, const char* file, int line) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(call) + " failed at " + file + ":" +
                             std::to_string(line) + ": cuBLAS status " +
                             std::to_string(static_cast<int>(status)));
  }
}

inline void check_cusolver(cusolverStatus_t status, const char* call, const char* file, int line) {
  if (status != CUSOLVER_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(call) + " failed at " + file + ":" +
                             std::to_string(line) + ": cuSOLVER status " +
                             std::to_string(static_cast<int>(status)));
  }
}

}  // namespace fasteig::gpu

#define FASTEIG_CUDA_CHECK(call) ::fasteig::gpu::check_cuda((call), #call, __FILE__, __LINE__)
#define FASTEIG_CUSPARSE_CHECK(call) ::fasteig::gpu::check_cusparse((call), #call, __FILE__, __LINE__)
#define FASTEIG_CUBLAS_CHECK(call) ::fasteig::gpu::check_cublas((call), #call, __FILE__, __LINE__)
#define FASTEIG_CUSOLVER_CHECK(call) ::fasteig::gpu::check_cusolver((call), #call, __FILE__, __LINE__)

#endif  // USE_CUDA

