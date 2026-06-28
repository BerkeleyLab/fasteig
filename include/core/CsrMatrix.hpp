#pragma once

#include "core/Executor.hpp"
#include "core/LinearOperator.hpp"
#include "fasteig.hpp"

#include <vector>

namespace fasteig {
namespace core {

class MultiVector;
class Vector;

struct CsrMatrixData {
  FASTEIG_INT rows = 0;
  FASTEIG_INT cols = 0;
  std::vector<int> rowptr;
  std::vector<int> colind;
  std::vector<double> values;

  FASTEIG_INT nonzeros() const noexcept {
    return static_cast<FASTEIG_INT>(values.size());
  }
};

void validate_csr_matrix_data(const CsrMatrixData& data);

class CsrMatrix : public LinearOperator {
public:
  virtual ~CsrMatrix() = default;

  virtual FASTEIG_INT nonzeros() const noexcept = 0;

  virtual CsrMatrixData copy_to_host() const = 0;

  void spmv(const Vector& x, Vector& y, double alpha = 1.0, double beta = 0.0) const {
    apply(x, y, alpha, beta);
  }

  void spmv_column(const MultiVector& X, FASTEIG_INT col, Vector& y,
                   double alpha = 1.0, double beta = 0.0) const {
    apply_column(X, col, y, alpha, beta);
  }

  void spmv_block(const MultiVector& X, FASTEIG_INT first_col, FASTEIG_INT block_size,
                  MultiVector& Y, double alpha = 1.0, double beta = 0.0) const {
    apply_block(X, first_col, block_size, Y, alpha, beta);
  }
};

class CpuCsrMatrix final : public CsrMatrix {
public:
  CpuCsrMatrix(const Executor& exec, CsrMatrixData data);

  const Executor& executor() const noexcept override { return exec_; }
  FASTEIG_INT rows() const noexcept override { return data_.rows; }
  FASTEIG_INT cols() const noexcept override { return data_.cols; }
  FASTEIG_INT nonzeros() const noexcept override { return data_.nonzeros(); }

  CsrMatrixData copy_to_host() const override { return data_; }
  void apply(const Vector& x, Vector& y, double alpha = 1.0, double beta = 0.0) const override;
  void apply_column(const MultiVector& X, FASTEIG_INT col, Vector& y,
                    double alpha = 1.0, double beta = 0.0) const override;
  void apply_block(const MultiVector& X, FASTEIG_INT first_col, FASTEIG_INT block_size,
                   MultiVector& Y, double alpha = 1.0, double beta = 0.0) const override;

  const CsrMatrixData& data() const noexcept { return data_; }

private:
  const Executor& exec_;
  CsrMatrixData data_;
};

}  // namespace core
}  // namespace fasteig

