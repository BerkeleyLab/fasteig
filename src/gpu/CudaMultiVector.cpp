#include "gpu/CudaMultiVector.hpp"

#ifdef USE_CUDA

#include "gpu/CudaExecutor.hpp"
#include "gpu/CudaContext.hpp"
#include "gpu/CudaCheck.hpp"
#include "gpu/CudaUtils.hpp"
#include "gpu/CudaVector.hpp"

#include <cuda_runtime.h>

#include <stdexcept>

namespace fasteig {
namespace core {

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
  if (d_h_) cudaFree(d_h_);
  if (d_s_) cudaFree(d_s_);
}

void CudaMultiVector::ensure_orthog_workspace(FASTEIG_INT active_cols) const {
  if (active_cols <= orthog_workspace_cols_) return;

  if (d_h_) {
    FASTEIG_CUDA_CHECK(cudaFree(d_h_));
    d_h_ = nullptr;
  }
  if (d_s_) {
    FASTEIG_CUDA_CHECK(cudaFree(d_s_));
    d_s_ = nullptr;
  }

  if (active_cols > 0) {
    FASTEIG_CUDA_CHECK(cudaMalloc(&d_h_, static_cast<size_t>(active_cols) * sizeof(double)));
    FASTEIG_CUDA_CHECK(cudaMalloc(&d_s_, static_cast<size_t>(active_cols) * sizeof(double)));
  }
  orthog_workspace_cols_ = active_cols;
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
  }
}

void CudaMultiVector::set_zero() {
  auto& context = cuda_executor_from(exec_, "CudaMultiVector::set_zero").context();
  const size_t count = static_cast<size_t>(rows_) * static_cast<size_t>(cols_);
  if (count > 0) {
    FASTEIG_CUDA_CHECK(cudaMemsetAsync(d_values_, 0, count * sizeof(double), context.stream()));
  }
}

void CudaMultiVector::scale(double alpha) {
  auto& context = cuda_executor_from(exec_, "CudaMultiVector::scale").context();
  const size_t count = static_cast<size_t>(rows_) * static_cast<size_t>(cols_);
  if (count > 0) {
    FASTEIG_CUBLAS_CHECK(cublasDscal(context.cublas_handle(), static_cast<int>(count),
                                     &alpha, d_values_, 1));
  }
}

void CudaMultiVector::copy_column_from(FASTEIG_INT col, const Vector& src) {
  if (col < 0 || col >= cols_) {
    throw std::invalid_argument("CudaMultiVector::copy_column_from: column out of range.");
  }
  if (src.size() != rows_) {
    throw std::invalid_argument("CudaMultiVector::copy_column_from: size mismatch.");
  }
  const auto* cuda_src = dynamic_cast<const CudaVector*>(&src);
  if (!cuda_src) {
    throw std::invalid_argument("CudaMultiVector::copy_column_from: source is not a CudaVector.");
  }
  auto& context = cuda_executor_from(exec_, "CudaMultiVector::copy_column_from").context();
  if (rows_ > 0) {
    double* d_col = d_values_ + static_cast<size_t>(col) * static_cast<size_t>(rows_);
    FASTEIG_CUBLAS_CHECK(cublasDcopy(context.cublas_handle(), static_cast<int>(rows_),
                                     cuda_src->data(), 1, d_col, 1));
  }
}

VectorXd CudaMultiVector::copy_column_to_host(FASTEIG_INT col) const {
  if (col < 0 || col >= cols_) {
    throw std::invalid_argument("CudaMultiVector::copy_column_to_host: column out of range.");
  }
  VectorXd out(static_cast<int>(rows_));
  auto& context = cuda_executor_from(exec_, "CudaMultiVector::copy_column_to_host").context();
  if (rows_ > 0) {
    const double* d_col = d_values_ + static_cast<size_t>(col) * static_cast<size_t>(rows_);
    FASTEIG_CUDA_CHECK(cudaMemcpyAsync(out.data(), d_col,
                                       static_cast<size_t>(rows_) * sizeof(double),
                                       cudaMemcpyDeviceToHost, context.stream()));
    FASTEIG_CUDA_CHECK(cudaStreamSynchronize(context.stream()));
  }
  return out;
}

double CudaMultiVector::column_dot(FASTEIG_INT col, const Vector& x) const {
  if (col < 0 || col >= cols_) {
    throw std::invalid_argument("CudaMultiVector::column_dot: column out of range.");
  }
  if (x.size() != rows_) {
    throw std::invalid_argument("CudaMultiVector::column_dot: size mismatch.");
  }
  const auto* cuda_x = dynamic_cast<const CudaVector*>(&x);
  if (!cuda_x) {
    throw std::invalid_argument("CudaMultiVector::column_dot: x is not a CudaVector.");
  }
  auto& context = cuda_executor_from(exec_, "CudaMultiVector::column_dot").context();
  double result = 0.0;
  if (rows_ > 0) {
    const double* d_col = d_values_ + static_cast<size_t>(col) * static_cast<size_t>(rows_);
    FASTEIG_CUBLAS_CHECK(cublasDdot(context.cublas_handle(), static_cast<int>(rows_),
                                    d_col, 1, cuda_x->data(), 1, &result));
    FASTEIG_CUDA_CHECK(cudaStreamSynchronize(context.stream()));
  }
  return result;
}

void CudaMultiVector::column_axpy(double alpha, FASTEIG_INT col, Vector& y) const {
  if (col < 0 || col >= cols_) {
    throw std::invalid_argument("CudaMultiVector::column_axpy: column out of range.");
  }
  if (y.size() != rows_) {
    throw std::invalid_argument("CudaMultiVector::column_axpy: size mismatch.");
  }
  auto* cuda_y = dynamic_cast<CudaVector*>(&y);
  if (!cuda_y) {
    throw std::invalid_argument("CudaMultiVector::column_axpy: y is not a CudaVector.");
  }
  auto& context = cuda_executor_from(exec_, "CudaMultiVector::column_axpy").context();
  if (rows_ > 0) {
    const double* d_col = d_values_ + static_cast<size_t>(col) * static_cast<size_t>(rows_);
    FASTEIG_CUBLAS_CHECK(cublasDaxpy(context.cublas_handle(), static_cast<int>(rows_),
                                     &alpha, d_col, 1, cuda_y->data(), 1));
  }
}

void CudaMultiVector::orthogonalize_active_columns(FASTEIG_INT active_cols,
                                                   const Vector& source,
                                                   Vector& residual,
                                                   std::vector<double>& coeffs) const {
  if (active_cols < 0 || active_cols > cols_) {
    throw std::invalid_argument("CudaMultiVector::orthogonalize_active_columns: invalid active column count.");
  }
  if (source.size() != rows_ || residual.size() != rows_) {
    throw std::invalid_argument("CudaMultiVector::orthogonalize_active_columns: size mismatch.");
  }
  const auto* cuda_source = dynamic_cast<const CudaVector*>(&source);
  auto* cuda_residual = dynamic_cast<CudaVector*>(&residual);
  if (!cuda_source || !cuda_residual) {
    throw std::invalid_argument(
        "CudaMultiVector::orthogonalize_active_columns: source and residual must be CudaVector objects.");
  }

  auto& context = cuda_executor_from(exec_, "CudaMultiVector::orthogonalize_active_columns").context();
  coeffs.assign(static_cast<std::size_t>(active_cols), 0.0);
  ensure_orthog_workspace(active_cols);

  orthogonalize_active_columns_device_coeffs(active_cols, source, residual, d_h_, d_s_);

  if (active_cols > 0) {
    FASTEIG_CUDA_CHECK(cudaMemcpyAsync(coeffs.data(), d_h_,
                                       static_cast<std::size_t>(active_cols) * sizeof(double),
                                       cudaMemcpyDeviceToHost, context.stream()));
    FASTEIG_CUDA_CHECK(cudaStreamSynchronize(context.stream()));
  }
}

double CudaMultiVector::orthogonalize_active_columns(FASTEIG_INT active_cols,
                                                     const Vector& source,
                                                     Vector& residual) const {
  if (active_cols < 0 || active_cols > cols_) {
    throw std::invalid_argument("CudaMultiVector::orthogonalize_active_columns: invalid active column count.");
  }
  if (source.size() != rows_ || residual.size() != rows_) {
    throw std::invalid_argument("CudaMultiVector::orthogonalize_active_columns: size mismatch.");
  }
  if (active_cols == 0) {
    residual.copy_from(source);
    return 0.0;
  }

  auto& context = cuda_executor_from(exec_, "CudaMultiVector::orthogonalize_active_columns").context();
  ensure_orthog_workspace(active_cols);

  orthogonalize_active_columns_device_coeffs(active_cols, source, residual, d_h_, d_s_);

  double alpha = 0.0;
  FASTEIG_CUDA_CHECK(cudaMemcpyAsync(&alpha, d_h_ + static_cast<std::size_t>(active_cols - 1),
                                     sizeof(double), cudaMemcpyDeviceToHost, context.stream()));
  FASTEIG_CUDA_CHECK(cudaStreamSynchronize(context.stream()));
  return alpha;
}

MatrixXd CudaMultiVector::multiply_active_columns(FASTEIG_INT active_cols,
                                                  const Eigen::Ref<const MatrixXd>& coeffs) const {
  if (active_cols < 0 || active_cols > cols_) {
    throw std::invalid_argument("CudaMultiVector::multiply_active_columns: invalid active column count.");
  }
  if (coeffs.rows() != static_cast<int>(active_cols)) {
    throw std::invalid_argument("CudaMultiVector::multiply_active_columns: coefficient row mismatch.");
  }

  const FASTEIG_INT out_cols = static_cast<FASTEIG_INT>(coeffs.cols());
  MatrixXd out(static_cast<int>(rows_), static_cast<int>(out_cols));
  out.setZero();
  if (rows_ == 0 || out_cols == 0 || active_cols == 0) return out;

  struct DeviceBuffer {
    double* ptr = nullptr;
    ~DeviceBuffer() {
      if (ptr) cudaFree(ptr);
    }
  };

  auto& context = cuda_executor_from(exec_, "CudaMultiVector::multiply_active_columns").context();
  DeviceBuffer d_coeffs;
  DeviceBuffer d_out;
  const std::size_t coeff_count =
      static_cast<std::size_t>(active_cols) * static_cast<std::size_t>(out_cols);
  const std::size_t out_count =
      static_cast<std::size_t>(rows_) * static_cast<std::size_t>(out_cols);

  FASTEIG_CUDA_CHECK(cudaMalloc(&d_coeffs.ptr, coeff_count * sizeof(double)));
  FASTEIG_CUDA_CHECK(cudaMalloc(&d_out.ptr, out_count * sizeof(double)));
  FASTEIG_CUDA_CHECK(cudaMemcpyAsync(d_coeffs.ptr, coeffs.data(), coeff_count * sizeof(double),
                                     cudaMemcpyHostToDevice, context.stream()));

  const double one = 1.0;
  const double zero = 0.0;
  FASTEIG_CUBLAS_CHECK(cublasDgemm(context.cublas_handle(),
                                   CUBLAS_OP_N,
                                   CUBLAS_OP_N,
                                   static_cast<int>(rows_),
                                   static_cast<int>(out_cols),
                                   static_cast<int>(active_cols),
                                   &one,
                                   d_values_,
                                   static_cast<int>(rows_),
                                   d_coeffs.ptr,
                                   static_cast<int>(active_cols),
                                   &zero,
                                   d_out.ptr,
                                   static_cast<int>(rows_)));
  FASTEIG_CUDA_CHECK(cudaMemcpyAsync(out.data(), d_out.ptr, out_count * sizeof(double),
                                     cudaMemcpyDeviceToHost, context.stream()));
  FASTEIG_CUDA_CHECK(cudaStreamSynchronize(context.stream()));

  return out;
}

void CudaMultiVector::orthogonalize_active_columns_device_coeffs(FASTEIG_INT active_cols,
                                                                 const Vector& source,
                                                                 Vector& residual,
                                                                 double* d_coeffs,
                                                                 double* d_correction) const {
  if (active_cols < 0 || active_cols > cols_) {
    throw std::invalid_argument(
        "CudaMultiVector::orthogonalize_active_columns_device_coeffs: invalid active column count.");
  }
  if (source.size() != rows_ || residual.size() != rows_) {
    throw std::invalid_argument(
        "CudaMultiVector::orthogonalize_active_columns_device_coeffs: size mismatch.");
  }
  if (active_cols > 0 && (!d_coeffs || !d_correction)) {
    throw std::invalid_argument(
        "CudaMultiVector::orthogonalize_active_columns_device_coeffs: null workspace.");
  }
  const auto* cuda_source = dynamic_cast<const CudaVector*>(&source);
  auto* cuda_residual = dynamic_cast<CudaVector*>(&residual);
  if (!cuda_source || !cuda_residual) {
    throw std::invalid_argument(
        "CudaMultiVector::orthogonalize_active_columns_device_coeffs: source and residual must be CudaVector objects.");
  }

  auto& context =
      cuda_executor_from(exec_, "CudaMultiVector::orthogonalize_active_columns_device_coeffs").context();
  if (rows_ <= 0) return;

  FASTEIG_CUBLAS_CHECK(cublasDcopy(context.cublas_handle(), static_cast<int>(rows_),
                                   cuda_source->data(), 1, cuda_residual->data(), 1));
  if (active_cols == 0) return;

  const double one = 1.0;
  const double zero = 0.0;
  const double minus_one = -1.0;
  const int nrows = static_cast<int>(rows_);
  const int ncols = static_cast<int>(active_cols);

  FASTEIG_CUBLAS_CHECK(cublasDgemv(context.cublas_handle(), CUBLAS_OP_T,
                                   nrows, ncols,
                                   &one, d_values_, nrows,
                                   cuda_source->data(), 1,
                                   &zero, d_coeffs, 1));
  FASTEIG_CUBLAS_CHECK(cublasDgemv(context.cublas_handle(), CUBLAS_OP_N,
                                   nrows, ncols,
                                   &minus_one, d_values_, nrows,
                                   d_coeffs, 1,
                                   &one, cuda_residual->data(), 1));
  FASTEIG_CUBLAS_CHECK(cublasDgemv(context.cublas_handle(), CUBLAS_OP_T,
                                   nrows, ncols,
                                   &one, d_values_, nrows,
                                   cuda_residual->data(), 1,
                                   &zero, d_correction, 1));
  FASTEIG_CUBLAS_CHECK(cublasDaxpy(context.cublas_handle(), ncols,
                                   &one, d_correction, 1, d_coeffs, 1));
  FASTEIG_CUBLAS_CHECK(cublasDgemv(context.cublas_handle(), CUBLAS_OP_N,
                                   nrows, ncols,
                                   &minus_one, d_values_, nrows,
                                   d_correction, 1,
                                   &one, cuda_residual->data(), 1));
}

}  // namespace core
}  // namespace fasteig

#endif  // USE_CUDA

