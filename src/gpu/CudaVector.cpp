#include "gpu/CudaVector.hpp"

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

CudaVector::CudaVector(const Executor& exec, FASTEIG_INT size)
    : exec_(exec), size_(size) {
  if (size < 0) {
    throw std::invalid_argument("CudaVector: size must be nonnegative.");
  }
  auto& context = cuda_executor_from(exec_, "CudaVector").context();
  if (size_ > 0) {
    FASTEIG_CUDA_CHECK(cudaMalloc(&d_values_, static_cast<size_t>(size_) * sizeof(double)));
    FASTEIG_CUDA_CHECK(cudaMemsetAsync(d_values_, 0, static_cast<size_t>(size_) * sizeof(double),
                                       context.stream()));
    FASTEIG_CUDA_CHECK(cudaStreamSynchronize(context.stream()));
  }
}

CudaVector::~CudaVector() {
  if (d_values_) cudaFree(d_values_);
}

void CudaVector::copy_from_host(const Eigen::Ref<const VectorXd>& src) {
  if (src.size() != size_) {
    throw std::invalid_argument("CudaVector::copy_from_host: size mismatch.");
  }
  auto& context = cuda_executor_from(exec_, "CudaVector::copy_from_host").context();
  if (size_ > 0) {
    FASTEIG_CUDA_CHECK(cudaMemcpyAsync(d_values_, src.data(),
                                       static_cast<size_t>(size_) * sizeof(double),
                                       cudaMemcpyHostToDevice, context.stream()));
    FASTEIG_CUDA_CHECK(cudaStreamSynchronize(context.stream()));
  }
}

VectorXd CudaVector::copy_to_host() const {
  VectorXd out(static_cast<int>(size_));
  auto& context = cuda_executor_from(exec_, "CudaVector::copy_to_host").context();
  if (size_ > 0) {
    FASTEIG_CUDA_CHECK(cudaMemcpyAsync(out.data(), d_values_,
                                       static_cast<size_t>(size_) * sizeof(double),
                                       cudaMemcpyDeviceToHost, context.stream()));
    FASTEIG_CUDA_CHECK(cudaStreamSynchronize(context.stream()));
  }
  return out;
}

void CudaVector::copy_from(const Vector& src) {
  if (src.size() != size_) {
    throw std::invalid_argument("CudaVector::copy_from: size mismatch.");
  }
  const auto* cuda_src = dynamic_cast<const CudaVector*>(&src);
  if (!cuda_src) {
    throw std::invalid_argument("CudaVector::copy_from: source is not a CudaVector.");
  }
  auto& context = cuda_executor_from(exec_, "CudaVector::copy_from").context();
  if (size_ > 0) {
    FASTEIG_CUBLAS_CHECK(cublasDcopy(context.cublas_handle(), static_cast<int>(size_),
                                     cuda_src->d_values_, 1, d_values_, 1));
    FASTEIG_CUDA_CHECK(cudaStreamSynchronize(context.stream()));
  }
}

void CudaVector::set_zero() {
  auto& context = cuda_executor_from(exec_, "CudaVector::set_zero").context();
  if (size_ > 0) {
    FASTEIG_CUDA_CHECK(cudaMemsetAsync(d_values_, 0, static_cast<size_t>(size_) * sizeof(double),
                                       context.stream()));
    FASTEIG_CUDA_CHECK(cudaStreamSynchronize(context.stream()));
  }
}

void CudaVector::scale(double alpha) {
  auto& context = cuda_executor_from(exec_, "CudaVector::scale").context();
  if (size_ > 0) {
    FASTEIG_CUBLAS_CHECK(cublasDscal(context.cublas_handle(), static_cast<int>(size_),
                                     &alpha, d_values_, 1));
    FASTEIG_CUDA_CHECK(cudaStreamSynchronize(context.stream()));
  }
}

void CudaVector::axpy(double alpha, const Vector& x) {
  if (x.size() != size_) {
    throw std::invalid_argument("CudaVector::axpy: size mismatch.");
  }
  const auto* cuda_x = dynamic_cast<const CudaVector*>(&x);
  if (!cuda_x) {
    throw std::invalid_argument("CudaVector::axpy: x is not a CudaVector.");
  }
  auto& context = cuda_executor_from(exec_, "CudaVector::axpy").context();
  if (size_ > 0) {
    FASTEIG_CUBLAS_CHECK(cublasDaxpy(context.cublas_handle(), static_cast<int>(size_),
                                     &alpha, cuda_x->d_values_, 1, d_values_, 1));
    FASTEIG_CUDA_CHECK(cudaStreamSynchronize(context.stream()));
  }
}

double CudaVector::dot(const Vector& x) const {
  if (x.size() != size_) {
    throw std::invalid_argument("CudaVector::dot: size mismatch.");
  }
  const auto* cuda_x = dynamic_cast<const CudaVector*>(&x);
  if (!cuda_x) {
    throw std::invalid_argument("CudaVector::dot: x is not a CudaVector.");
  }
  auto& context = cuda_executor_from(exec_, "CudaVector::dot").context();
  double result = 0.0;
  if (size_ > 0) {
    FASTEIG_CUBLAS_CHECK(cublasDdot(context.cublas_handle(), static_cast<int>(size_),
                                    d_values_, 1, cuda_x->d_values_, 1, &result));
    FASTEIG_CUDA_CHECK(cudaStreamSynchronize(context.stream()));
  }
  return result;
}

double CudaVector::norm2() const {
  auto& context = cuda_executor_from(exec_, "CudaVector::norm2").context();
  double result = 0.0;
  if (size_ > 0) {
    FASTEIG_CUBLAS_CHECK(cublasDnrm2(context.cublas_handle(), static_cast<int>(size_),
                                     d_values_, 1, &result));
    FASTEIG_CUDA_CHECK(cudaStreamSynchronize(context.stream()));
  }
  return result;
}

}  // namespace core
}  // namespace fasteig

#endif  // USE_CUDA

