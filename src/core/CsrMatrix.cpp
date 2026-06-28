#include "core/CsrMatrix.hpp"

#include "core/MultiVector.hpp"
#include "core/Vector.hpp"

#include <stdexcept>
#include <utility>

namespace fasteig {
namespace core {
void validate_csr_matrix_data(const CsrMatrixData& data) {
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

CpuCsrMatrix::CpuCsrMatrix(const Executor& exec, CsrMatrixData data)
    : exec_(exec), data_(std::move(data)) {
  validate_csr_matrix_data(data_);
}

void CpuCsrMatrix::apply(const Vector& x, Vector& y, double alpha, double beta) const {
  if (x.size() != cols() || y.size() != rows()) {
    throw std::invalid_argument("CpuCsrMatrix::apply: dimension mismatch.");
  }
  const auto* cpu_x = dynamic_cast<const CpuVector*>(&x);
  auto* cpu_y = dynamic_cast<CpuVector*>(&y);
  if (!cpu_x || !cpu_y) {
    throw std::invalid_argument("CpuCsrMatrix::apply: x and y must be CpuVector objects.");
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

void CpuCsrMatrix::apply_column(const MultiVector& X,
                                FASTEIG_INT col,
                                Vector& y,
                                double alpha,
                                double beta) const {
  if (col < 0 || col >= X.cols()) {
    throw std::invalid_argument("CpuCsrMatrix::apply_column: column out of range.");
  }
  if (X.rows() != cols() || y.size() != rows()) {
    throw std::invalid_argument("CpuCsrMatrix::apply_column: dimension mismatch.");
  }
  const auto* cpu_X = dynamic_cast<const CpuMultiVector*>(&X);
  auto* cpu_y = dynamic_cast<CpuVector*>(&y);
  if (!cpu_X || !cpu_y) {
    throw std::invalid_argument("CpuCsrMatrix::apply_column: X and y must be CPU objects.");
  }

  if (beta == 0.0) {
    cpu_y->values().setZero();
  } else {
    cpu_y->values() *= beta;
  }

  const auto xv = cpu_X->values().col(static_cast<int>(col));
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

void CpuCsrMatrix::apply_block(const MultiVector& X,
                               FASTEIG_INT first_col,
                               FASTEIG_INT block_size,
                               MultiVector& Y,
                               double alpha,
                               double beta) const {
  if (first_col < 0 || block_size <= 0 || first_col + block_size > X.cols()) {
    throw std::invalid_argument("CpuCsrMatrix::apply_block: invalid column range.");
  }
  if (X.rows() != cols() || Y.rows() != rows() || Y.cols() < block_size) {
    throw std::invalid_argument("CpuCsrMatrix::apply_block: dimension mismatch.");
  }
  const auto* cpu_X = dynamic_cast<const CpuMultiVector*>(&X);
  auto* cpu_Y = dynamic_cast<CpuMultiVector*>(&Y);
  if (!cpu_X || !cpu_Y) {
    throw std::invalid_argument("CpuCsrMatrix::apply_block: X and Y must be CPU objects.");
  }

  auto Y_block = cpu_Y->values().leftCols(static_cast<int>(block_size));
  if (beta == 0.0) {
    Y_block.setZero();
  } else {
    Y_block *= beta;
  }

  const MatrixXd& Xv = cpu_X->values();
  for (FASTEIG_INT row = 0; row < data_.rows; ++row) {
    const int begin = data_.rowptr[static_cast<std::size_t>(row)];
    const int end = data_.rowptr[static_cast<std::size_t>(row + 1)];
    for (int k = begin; k < end; ++k) {
      const int col = data_.colind[static_cast<std::size_t>(k)];
      const double a = alpha * data_.values[static_cast<std::size_t>(k)];
      Y_block.row(static_cast<int>(row)).noalias() +=
          a * Xv.row(col).segment(static_cast<int>(first_col), static_cast<int>(block_size));
    }
  }
}

}  // namespace core
}  // namespace fasteig

