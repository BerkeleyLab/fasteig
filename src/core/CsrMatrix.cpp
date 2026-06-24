#include "core/CsrMatrix.hpp"

#include "core/Vector.hpp"

#include <stdexcept>
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

}  // namespace

CpuCsrMatrix::CpuCsrMatrix(const Executor& exec, CsrMatrixData data)
    : exec_(exec), data_(std::move(data)) {
  validate_csr_data(data_);
}

void CpuCsrMatrix::spmv(const Vector& x, Vector& y, double alpha, double beta) const {
  if (x.size() != cols() || y.size() != rows()) {
    throw std::invalid_argument("CpuCsrMatrix::spmv: dimension mismatch.");
  }
  const auto* cpu_x = dynamic_cast<const CpuVector*>(&x);
  auto* cpu_y = dynamic_cast<CpuVector*>(&y);
  if (!cpu_x || !cpu_y) {
    throw std::invalid_argument("CpuCsrMatrix::spmv: x and y must be CpuVector objects.");
  }

  if (beta == 0.0) {
    cpu_y->values().setZero();
  } else {
    cpu_y->values() *= beta;
  }

  const VectorXd& xv = cpu_x->values();
  VectorXd& yv = cpu_y->values();
  for (FASTEIG_INT row = 0; row < data_.rows; ++row) {
    double sum = 0.0;
    const int begin = data_.rowptr[static_cast<std::size_t>(row)];
    const int end = data_.rowptr[static_cast<std::size_t>(row + 1)];
    for (int k = begin; k < end; ++k) {
      sum += data_.values[static_cast<std::size_t>(k)] *
             xv(data_.colind[static_cast<std::size_t>(k)]);
    }
    yv(static_cast<int>(row)) += alpha * sum;
  }
}

}  // namespace core
}  // namespace fasteig
