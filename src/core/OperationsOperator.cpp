#include "core/OperationsOperator.hpp"

#include "core/MultiVector.hpp"
#include "core/Vector.hpp"

#include <stdexcept>

namespace fasteig {
namespace core {

CpuOperationsOperator::CpuOperationsOperator(const Executor& exec,
                                             FASTEIG_INT n,
                                             const operations& ops)
    : exec_(exec), n_(n), ops_(ops) {
  if (n <= 0) {
    throw std::invalid_argument("CpuOperationsOperator: dimension must be positive.");
  }
  if (!ops_.matvec) {
    throw std::invalid_argument("CpuOperationsOperator: ops.matvec is not set.");
  }
}

void CpuOperationsOperator::apply(const Vector& x,
                                  Vector& y,
                                  double alpha,
                                  double beta) const {
  if (x.size() != n_ || y.size() != n_) {
    throw std::invalid_argument("CpuOperationsOperator::apply: dimension mismatch.");
  }
  const auto* cpu_x = dynamic_cast<const CpuVector*>(&x);
  auto* cpu_y = dynamic_cast<CpuVector*>(&y);
  if (!cpu_x || !cpu_y) {
    throw std::invalid_argument("CpuOperationsOperator::apply: x and y must be CpuVector objects.");
  }

  VectorXd tmp(static_cast<int>(n_));
  ops_.matvec(cpu_x->values(), tmp);

  if (beta == 0.0) {
    cpu_y->values().noalias() = alpha * tmp;
  } else {
    cpu_y->values() = beta * cpu_y->values() + alpha * tmp;
  }
}

void CpuOperationsOperator::apply_column(const MultiVector& X,
                                         FASTEIG_INT col,
                                         Vector& y,
                                         double alpha,
                                         double beta) const {
  if (col < 0 || col >= X.cols()) {
    throw std::invalid_argument("CpuOperationsOperator::apply_column: column out of range.");
  }
  if (X.rows() != n_ || y.size() != n_) {
    throw std::invalid_argument("CpuOperationsOperator::apply_column: dimension mismatch.");
  }
  const auto* cpu_X = dynamic_cast<const CpuMultiVector*>(&X);
  auto* cpu_y = dynamic_cast<CpuVector*>(&y);
  if (!cpu_X || !cpu_y) {
    throw std::invalid_argument(
        "CpuOperationsOperator::apply_column: X and y must be CPU objects.");
  }

  VectorXd tmp(static_cast<int>(n_));
  ops_.matvec(cpu_X->values().col(static_cast<int>(col)), tmp);

  if (beta == 0.0) {
    cpu_y->values().noalias() = alpha * tmp;
  } else {
    cpu_y->values() = beta * cpu_y->values() + alpha * tmp;
  }
}

void CpuOperationsOperator::apply_block(const MultiVector& X,
                                        FASTEIG_INT first_col,
                                        FASTEIG_INT block_size,
                                        MultiVector& Y,
                                        double alpha,
                                        double beta) const {
  if (first_col < 0 || block_size <= 0 || first_col + block_size > X.cols()) {
    throw std::invalid_argument("CpuOperationsOperator::apply_block: invalid column range.");
  }
  if (X.rows() != n_ || Y.rows() != n_ || Y.cols() < block_size) {
    throw std::invalid_argument("CpuOperationsOperator::apply_block: dimension mismatch.");
  }
  const auto* cpu_X = dynamic_cast<const CpuMultiVector*>(&X);
  auto* cpu_Y = dynamic_cast<CpuMultiVector*>(&Y);
  if (!cpu_X || !cpu_Y) {
    throw std::invalid_argument(
        "CpuOperationsOperator::apply_block: X and Y must be CPU objects.");
  }

  MatrixXd tmp(static_cast<int>(n_), static_cast<int>(block_size));
  if (ops_.block_matvec) {
    MatrixXd X_block =
        cpu_X->values().middleCols(static_cast<int>(first_col), static_cast<int>(block_size));
    ops_.block_matvec(X_block, tmp);
  } else {
    VectorXd tmp_col(static_cast<int>(n_));
    for (FASTEIG_INT j = 0; j < block_size; ++j) {
      ops_.matvec(cpu_X->values().col(static_cast<int>(first_col + j)),
                  tmp_col);
      tmp.col(static_cast<int>(j)) = tmp_col;
    }
  }

  auto Y_block = cpu_Y->values().leftCols(static_cast<int>(block_size));
  if (beta == 0.0) {
    Y_block.noalias() = alpha * tmp;
  } else {
    Y_block = beta * Y_block + alpha * tmp;
  }
}

}  // namespace core
}  // namespace fasteig

