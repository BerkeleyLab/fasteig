#pragma once

#include "core/Executor.hpp"
#include "fasteig.hpp"

#include <vector>

namespace fasteig {
namespace core {

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

class CsrMatrix {
public:
  virtual ~CsrMatrix() = default;

  virtual const Executor& executor() const noexcept = 0;
  virtual FASTEIG_INT rows() const noexcept = 0;
  virtual FASTEIG_INT cols() const noexcept = 0;
  virtual FASTEIG_INT nonzeros() const noexcept = 0;

  virtual CsrMatrixData copy_to_host() const = 0;
  virtual void spmv(const Vector& x, Vector& y, double alpha = 1.0, double beta = 0.0) const = 0;
};

class CpuCsrMatrix final : public CsrMatrix {
public:
  CpuCsrMatrix(const Executor& exec, CsrMatrixData data);

  const Executor& executor() const noexcept override { return exec_; }
  FASTEIG_INT rows() const noexcept override { return data_.rows; }
  FASTEIG_INT cols() const noexcept override { return data_.cols; }
  FASTEIG_INT nonzeros() const noexcept override { return data_.nonzeros(); }

  CsrMatrixData copy_to_host() const override { return data_; }
  void spmv(const Vector& x, Vector& y, double alpha = 1.0, double beta = 0.0) const override;

  const CsrMatrixData& data() const noexcept { return data_; }

private:
  const Executor& exec_;
  CsrMatrixData data_;
};

}  // namespace core
}  // namespace fasteig
