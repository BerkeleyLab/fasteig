#include "gpu/CudaCsrMatrix.hpp"

#ifdef USE_CUDA

#include "gpu/CudaExecutor.hpp"
#include "gpu/CudaMultiVector.hpp"
#include "gpu/CudaVector.hpp"
#include "gpu/CudaContext.hpp"
#include "gpu/CudaCheck.hpp"
#include "gpu/CudaUtils.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <utility>

namespace fasteig {
namespace core {
CudaCsrMatrix::CudaCsrMatrix(const Executor& exec, CsrMatrixData data)
    : exec_(exec), rows_(data.rows), cols_(data.cols), nnz_(data.nonzeros()) {
  validate_csr_matrix_data(data);

  auto& context = cuda_executor_from(exec_, "CudaCsrMatrix").context();
  FASTEIG_CUDA_CHECK(cudaMalloc(&d_rowptr_, data.rowptr.size() * sizeof(int)));
  if (!data.colind.empty()) {
    FASTEIG_CUDA_CHECK(cudaMalloc(&d_colind_, data.colind.size() * sizeof(int)));
    FASTEIG_CUDA_CHECK(cudaMalloc(&d_values_, data.values.size() * sizeof(double)));
  }

  FASTEIG_CUDA_CHECK(cudaMemcpyAsync(d_rowptr_, data.rowptr.data(),
                                     data.rowptr.size() * sizeof(int),
                                     cudaMemcpyHostToDevice, context.stream()));
  if (nnz_ > 0) {
    FASTEIG_CUDA_CHECK(cudaMemcpyAsync(d_colind_, data.colind.data(),
                                       data.colind.size() * sizeof(int),
                                       cudaMemcpyHostToDevice, context.stream()));
    FASTEIG_CUDA_CHECK(cudaMemcpyAsync(d_values_, data.values.data(),
                                       data.values.size() * sizeof(double),
                                       cudaMemcpyHostToDevice, context.stream()));
  }
  FASTEIG_CUDA_CHECK(cudaStreamSynchronize(context.stream()));

  FASTEIG_CUSPARSE_CHECK(cusparseCreateCsr(&mat_descr_,
                                           static_cast<int64_t>(rows_),
                                           static_cast<int64_t>(cols_),
                                           static_cast<int64_t>(nnz_),
                                           d_rowptr_, d_colind_, d_values_,
                                           CUSPARSE_INDEX_32I,
                                           CUSPARSE_INDEX_32I,
                                           CUSPARSE_INDEX_BASE_ZERO,
                                           CUDA_R_64F));
}

CudaCsrMatrix::~CudaCsrMatrix() {
  if (x_descr_) cusparseDestroyDnVec(x_descr_);
  if (y_descr_) cusparseDestroyDnVec(y_descr_);
  if (mat_descr_) cusparseDestroySpMat(mat_descr_);
  if (d_spmv_buffer_) cudaFree(d_spmv_buffer_);
  if (d_rowptr_) cudaFree(d_rowptr_);
  if (d_colind_) cudaFree(d_colind_);
  if (d_values_) cudaFree(d_values_);
}

CsrMatrixData CudaCsrMatrix::copy_to_host() const {
  CsrMatrixData out;
  out.rows = rows_;
  out.cols = cols_;
  out.rowptr.resize(static_cast<std::size_t>(rows_ + 1));
  out.colind.resize(static_cast<std::size_t>(nnz_));
  out.values.resize(static_cast<std::size_t>(nnz_));

  auto& context = cuda_executor_from(exec_, "CudaCsrMatrix::copy_to_host").context();
  FASTEIG_CUDA_CHECK(cudaMemcpyAsync(out.rowptr.data(), d_rowptr_,
                                     out.rowptr.size() * sizeof(int),
                                     cudaMemcpyDeviceToHost, context.stream()));
  if (nnz_ > 0) {
    FASTEIG_CUDA_CHECK(cudaMemcpyAsync(out.colind.data(), d_colind_,
                                       out.colind.size() * sizeof(int),
                                       cudaMemcpyDeviceToHost, context.stream()));
    FASTEIG_CUDA_CHECK(cudaMemcpyAsync(out.values.data(), d_values_,
                                       out.values.size() * sizeof(double),
                                       cudaMemcpyDeviceToHost, context.stream()));
  }
  FASTEIG_CUDA_CHECK(cudaStreamSynchronize(context.stream()));
  return out;
}

void CudaCsrMatrix::apply(const Vector& x, Vector& y, double alpha, double beta) const {
  if (x.size() != cols_ || y.size() != rows_) {
    throw std::invalid_argument("CudaCsrMatrix::apply: dimension mismatch.");
  }
  const auto* cuda_x = dynamic_cast<const CudaVector*>(&x);
  auto* cuda_y = dynamic_cast<CudaVector*>(&y);
  if (!cuda_x || !cuda_y) {
    throw std::invalid_argument("CudaCsrMatrix::apply: x and y must be CudaVector objects.");
  }

  auto& context = cuda_executor_from(exec_, "CudaCsrMatrix::apply").context();
  if (!x_descr_) {
    FASTEIG_CUSPARSE_CHECK(cusparseCreateDnVec(&x_descr_,
                                               static_cast<int64_t>(cols_),
                                               const_cast<double*>(cuda_x->data()),
                                               CUDA_R_64F));
  } else {
    FASTEIG_CUSPARSE_CHECK(cusparseDnVecSetValues(x_descr_,
                                                  const_cast<double*>(cuda_x->data())));
  }

  if (!y_descr_) {
    FASTEIG_CUSPARSE_CHECK(cusparseCreateDnVec(&y_descr_,
                                               static_cast<int64_t>(rows_),
                                               cuda_y->data(),
                                               CUDA_R_64F));
  } else {
    FASTEIG_CUSPARSE_CHECK(cusparseDnVecSetValues(y_descr_, cuda_y->data()));
  }

  if (!d_spmv_buffer_) {
    FASTEIG_CUSPARSE_CHECK(cusparseSpMV_bufferSize(context.cusparse_handle(),
                                                   CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                   &alpha, mat_descr_, x_descr_,
                                                   &beta, y_descr_, CUDA_R_64F,
                                                   CUSPARSE_SPMV_ALG_DEFAULT,
                                                   &spmv_buffer_size_));
    if (spmv_buffer_size_ > 0) {
      FASTEIG_CUDA_CHECK(cudaMalloc(&d_spmv_buffer_, spmv_buffer_size_));
    }
  }

  FASTEIG_CUSPARSE_CHECK(cusparseSpMV(context.cusparse_handle(),
                                      CUSPARSE_OPERATION_NON_TRANSPOSE,
                                      &alpha, mat_descr_, x_descr_,
                                      &beta, y_descr_, CUDA_R_64F,
                                      CUSPARSE_SPMV_ALG_DEFAULT,
                                      d_spmv_buffer_));
}

void CudaCsrMatrix::apply_column(const MultiVector& X,
                                 FASTEIG_INT col,
                                 Vector& y,
                                 double alpha,
                                 double beta) const {
  if (col < 0 || col >= X.cols()) {
    throw std::invalid_argument("CudaCsrMatrix::apply_column: column out of range.");
  }
  if (X.rows() != cols_ || y.size() != rows_) {
    throw std::invalid_argument("CudaCsrMatrix::apply_column: dimension mismatch.");
  }
  const auto* cuda_X = dynamic_cast<const CudaMultiVector*>(&X);
  auto* cuda_y = dynamic_cast<CudaVector*>(&y);
  if (!cuda_X || !cuda_y) {
    throw std::invalid_argument("CudaCsrMatrix::apply_column: X and y must be CUDA objects.");
  }

  const double* d_x = cuda_X->data() + static_cast<size_t>(col) * static_cast<size_t>(X.rows());
  auto& context = cuda_executor_from(exec_, "CudaCsrMatrix::apply_column").context();
  if (!x_descr_) {
    FASTEIG_CUSPARSE_CHECK(cusparseCreateDnVec(&x_descr_,
                                               static_cast<int64_t>(cols_),
                                               const_cast<double*>(d_x),
                                               CUDA_R_64F));
  } else {
    FASTEIG_CUSPARSE_CHECK(cusparseDnVecSetValues(x_descr_, const_cast<double*>(d_x)));
  }

  if (!y_descr_) {
    FASTEIG_CUSPARSE_CHECK(cusparseCreateDnVec(&y_descr_,
                                               static_cast<int64_t>(rows_),
                                               cuda_y->data(),
                                               CUDA_R_64F));
  } else {
    FASTEIG_CUSPARSE_CHECK(cusparseDnVecSetValues(y_descr_, cuda_y->data()));
  }

  if (!d_spmv_buffer_) {
    FASTEIG_CUSPARSE_CHECK(cusparseSpMV_bufferSize(context.cusparse_handle(),
                                                   CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                   &alpha, mat_descr_, x_descr_,
                                                   &beta, y_descr_, CUDA_R_64F,
                                                   CUSPARSE_SPMV_ALG_DEFAULT,
                                                   &spmv_buffer_size_));
    if (spmv_buffer_size_ > 0) {
      FASTEIG_CUDA_CHECK(cudaMalloc(&d_spmv_buffer_, spmv_buffer_size_));
    }
  }

  FASTEIG_CUSPARSE_CHECK(cusparseSpMV(context.cusparse_handle(),
                                      CUSPARSE_OPERATION_NON_TRANSPOSE,
                                      &alpha, mat_descr_, x_descr_,
                                      &beta, y_descr_, CUDA_R_64F,
                                      CUSPARSE_SPMV_ALG_DEFAULT,
                                      d_spmv_buffer_));
}

void CudaCsrMatrix::apply_block(const MultiVector& X,
                                FASTEIG_INT first_col,
                                FASTEIG_INT block_size,
                                MultiVector& Y,
                                double alpha,
                                double beta) const {
  if (first_col < 0 || block_size <= 0 || first_col + block_size > X.cols()) {
    throw std::invalid_argument("CudaCsrMatrix::apply_block: invalid column range.");
  }
  if (X.rows() != cols_ || Y.rows() != rows_ || Y.cols() < block_size) {
    throw std::invalid_argument("CudaCsrMatrix::apply_block: dimension mismatch.");
  }
  const auto* cuda_X = dynamic_cast<const CudaMultiVector*>(&X);
  auto* cuda_Y = dynamic_cast<CudaMultiVector*>(&Y);
  if (!cuda_X || !cuda_Y) {
    throw std::invalid_argument("CudaCsrMatrix::apply_block: X and Y must be CUDA objects.");
  }

  auto y = exec_.create_vector(rows_);
  for (FASTEIG_INT j = 0; j < block_size; ++j) {
    if (beta != 0.0) {
      y->copy_from_host(Y.copy_column_to_host(j));
    }
    apply_column(X, first_col + j, *y, alpha, beta);
    cuda_Y->copy_column_from(j, *y);
  }
}

}  // namespace core
}  // namespace fasteig

#endif  // USE_CUDA

