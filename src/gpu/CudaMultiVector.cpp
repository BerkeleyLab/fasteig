#include "gpu/CudaMultiVector.hpp"

#ifdef USE_CUDA

#include "gpu/CudaExecutor.hpp"
#include "gpu/CudaContext.hpp"
#include "gpu/CudaCheck.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace fasteig {
namespace core {
namespace {

const CudaExecutor& cuda_executor_from(const Executor& exec, const char* where) {
  const auto* cuda_exec = dynamic_cast<const CudaExecutor*>(&exec);
  if (!cuda_exec) {
    throw std::invalid_argument(std::string(where) + ": executor is not a CudaExecutor.");
  }
  return *cuda_exec;
}

}  // namespace

CudaMultiVector::CudaMultiVector(const Executor& exec, FASTEIG_INT rows, FASTEIG_INT cols)
    : exec_(exec), rows_(rows), cols_(cols) {
  if (rows < 0 || cols < 0) {
    throw std::invalid_argument("CudaMultiVector: dimensions must be nonnegative.");
  }
  auto& context = cuda_executor_from(exec_, "CudaMultiVector").context();
  const size_t count = static_cast<size_t>(rows_) * static_cast<size_t>(cols_);
  if (count > 0) {
    FASTEIG_CUDA_CHECK(cudaMalloc(&d_values_, count * sizeof(double)));
    FASTEIG_CUDA_CHECK(cudaMemsetAsync(d_values_, 0, count * sizeof(double), context.stream()));
    FASTEIG_CUDA_CHECK(cudaStreamSynchronize(context.stream()));
  }
}

CudaMultiVector::~CudaMultiVector() {
  if (d_values_) cudaFree(d_values_);
}

void CudaMultiVector::copy_from_host(const Eigen::Ref<const MatrixXd>& src) {
  if (src.rows() != rows_ || src.cols() != cols_) {
    throw std::invalid_argument("CudaMultiVector::copy_from_host: dimension mismatch.");
  }
  auto& context = cuda_executor_from(exec_, "CudaMultiVector::copy_from_host").context();
  const size_t count = static_cast<size_t>(rows_) * static_cast<size_t>(cols_);
  if (count > 0) {
    FASTEIG_CUDA_CHECK(cudaMemcpyAsync(d_values_, src.data(), count * sizeof(double),
                                       cudaMemcpyHostToDevice, context.stream()));
    FASTEIG_CUDA_CHECK(cudaStreamSynchronize(context.stream()));
  }
}

MatrixXd CudaMultiVector::copy_to_host() const {
  MatrixXd out(static_cast<int>(rows_), static_cast<int>(cols_));
  auto& context = cuda_executor_from(exec_, "CudaMultiVector::copy_to_host").context();
  const size_t count = static_cast<size_t>(rows_) * static_cast<size_t>(cols_);
  if (count > 0) {
    FASTEIG_CUDA_CHECK(cudaMemcpyAsync(out.data(), d_values_, count * sizeof(double),
                                       cudaMemcpyDeviceToHost, context.stream()));
    FASTEIG_CUDA_CHECK(cudaStreamSynchronize(context.stream()));
  }
  return out;
}

void CudaMultiVector::copy_from(const MultiVector& src) {
  if (src.rows() != rows_ || src.cols() != cols_) {
    throw std::invalid_argument("CudaMultiVector::copy_from: dimension mismatch.");
  }
  const auto* cuda_src = dynamic_cast<const CudaMultiVector*>(&src);
  if (!cuda_src) {
    throw std::invalid_argument("CudaMultiVector::copy_from: source is not a CudaMultiVector.");
  }
  auto& context = cuda_executor_from(exec_, "CudaMultiVector::copy_from").context();
  const size_t count = static_cast<size_t>(rows_) * static_cast<size_t>(cols_);
  if (count > 0) {
    FASTEIG_CUBLAS_CHECK(cublasDcopy(context.cublas_handle(), static_cast<int>(count),
                                     cuda_src->d_values_, 1, d_values_, 1));
    FASTEIG_CUDA_CHECK(cudaStreamSynchronize(context.stream()));
  }
}

void CudaMultiVector::set_zero() {
  auto& context = cuda_executor_from(exec_, "CudaMultiVector::set_zero").context();
  const size_t count = static_cast<size_t>(rows_) * static_cast<size_t>(cols_);
  if (count > 0) {
    FASTEIG_CUDA_CHECK(cudaMemsetAsync(d_values_, 0, count * sizeof(double), context.stream()));
    FASTEIG_CUDA_CHECK(cudaStreamSynchronize(context.stream()));
  }
}

void CudaMultiVector::scale(double alpha) {
  auto& context = cuda_executor_from(exec_, "CudaMultiVector::scale").context();
  const size_t count = static_cast<size_t>(rows_) * static_cast<size_t>(cols_);
  if (count > 0) {
    FASTEIG_CUBLAS_CHECK(cublasDscal(context.cublas_handle(), static_cast<int>(count),
                                     &alpha, d_values_, 1));
    FASTEIG_CUDA_CHECK(cudaStreamSynchronize(context.stream()));
  }
}

}  // namespace core
}  // namespace fasteig

#endif  // USE_CUDA

