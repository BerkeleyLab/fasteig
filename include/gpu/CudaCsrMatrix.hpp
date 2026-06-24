#pragma once

#ifdef USE_CUDA

#include "core/CsrMatrix.hpp"

#include <cusparse.h>

namespace fasteig {
namespace core {

class CudaCsrMatrix final : public CsrMatrix {
public:
  CudaCsrMatrix(const Executor& exec, CsrMatrixData data);
  ~CudaCsrMatrix() override;

  CudaCsrMatrix(const CudaCsrMatrix&) = delete;
  CudaCsrMatrix& operator=(const CudaCsrMatrix&) = delete;

  const Executor& executor() const noexcept override { return exec_; }
  FASTEIG_INT rows() const noexcept override { return rows_; }
  FASTEIG_INT cols() const noexcept override { return cols_; }
  FASTEIG_INT nonzeros() const noexcept override { return nnz_; }

  CsrMatrixData copy_to_host() const override;
  void spmv(const Vector& x, Vector& y, double alpha = 1.0, double beta = 0.0) const override;

private:
  const Executor& exec_;
  FASTEIG_INT rows_ = 0;
  FASTEIG_INT cols_ = 0;
  FASTEIG_INT nnz_ = 0;
  int* d_rowptr_ = nullptr;
  int* d_colind_ = nullptr;
  double* d_values_ = nullptr;
  mutable void* d_spmv_buffer_ = nullptr;
  mutable size_t spmv_buffer_size_ = 0;
  mutable cusparseSpMatDescr_t mat_descr_ = nullptr;
  mutable cusparseDnVecDescr_t x_descr_ = nullptr;
  mutable cusparseDnVecDescr_t y_descr_ = nullptr;
};

}  // namespace core
}  // namespace fasteig

#endif  // USE_CUDA

