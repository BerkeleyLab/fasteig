#include "gpu/CudaCsrMatrix.hpp"

#ifdef USE_CUDA

#include "gpu/CudaExecutor.hpp"
#include "gpu/CudaVector.hpp"
#include "gpu/CudaContext.hpp"
#include "gpu/CudaCheck.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

namespace fasteig {
namespace core {
namespace {

void validate_csr_data(const CsrMatrixData& data) {
  if (data.rows < 0 || data.cols < 0) {
    throw std::invalid_argument("CsrMatrixData: dimensions must be nonnegative.");
  }
  if (data.rowptr.size() != static_cast<std::size_t>(data.rows + 1)) {
    throw std::invalid_argument("CsrMatrixData: rowptr size must be rows + 1.");
  }
  if (data.colind.size() != data.values.size()) {
    throw std::invalid_argument("CsrMatrixData: colind and values sizes differ.");
  }
  if (!data.rowptr.empty()) {
    if (data.rowptr.front() != 0) {
      throw std::invalid_argument("CsrMatrixData: rowptr must start at zero.");
    }
    if (data.rowptr.back() != static_cast<int>(data.values.size())) {
      throw std::invalid_argument("CsrMatrixData: rowptr end must equal nnz.");
    }
  }
  for (std::size_t i = 1; i < data.rowptr.size(); ++i) {
    if (data.rowptr[i] < data.rowptr[i - 1]) {
      throw std::invalid_argument("CsrMatrixData: rowptr must be nondecreasing.");
    }
  }
  for (int col : data.colind) {
    if (col < 0 || col >= data.cols) {
      throw std::invalid_argument("CsrMatrixData: column index out of range.");
    }
  }
}

const CudaExecutor& cuda_executor_from(const Executor& exec, const char* where) {
  const auto* cuda_exec = dynamic_cast<const CudaExecutor*>(&exec);
  if (!cuda_exec) {
    throw std::invalid_argument(std::string(where) + ": executor is not a CudaExecutor.");
  }
  return *cuda_exec;
}

}  // namespace

CudaCsrMatrix::CudaCsrMatrix(const Executor& exec, CsrMatrixData data)
    : exec_(exec), rows_(data.rows), cols_(data.cols), nnz_(data.nonzeros()) {
  validate_csr_data(data);

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

void CudaCsrMatrix::spmv(const Vector& x, Vector& y, double alpha, double beta) const {
  if (x.size() != cols_ || y.size() != rows_) {
    throw std::invalid_argument("CudaCsrMatrix::spmv: dimension mismatch.");
  }
  const auto* cuda_x = dynamic_cast<const CudaVector*>(&x);
  auto* cuda_y = dynamic_cast<CudaVector*>(&y);
  if (!cuda_x || !cuda_y) {
    throw std::invalid_argument("CudaCsrMatrix::spmv: x and y must be CudaVector objects.");
  }

  auto& context = cuda_executor_from(exec_, "CudaCsrMatrix::spmv").context();
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
  FASTEIG_CUDA_CHECK(cudaStreamSynchronize(context.stream()));
}

}  // namespace core
}  // namespace fasteig

#endif  // USE_CUDA

