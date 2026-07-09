#include "core/Operations.hpp"

#include <stdexcept>
#include <string>

namespace fasteig {
namespace core {

namespace {

void require_same_executor(const Executor& a, const Executor& b, const char* where) {
  if (&a != &b) {
    throw std::invalid_argument(std::string(where) + ": executor mismatch.");
  }
}

}  // namespace

void copy(const Vector& x, Vector& y) {
  require_same_executor(x.executor(), y.executor(), "core::copy(Vector)");
  y.copy_from(x);
}

void set_zero(Vector& x) {
  x.set_zero();
}

void scal(double alpha, Vector& x) {
  x.scale(alpha);
}

void axpy(double alpha, const Vector& x, Vector& y) {
  require_same_executor(x.executor(), y.executor(), "core::axpy");
  y.axpy(alpha, x);
}

double dot(const Vector& x, const Vector& y) {
  require_same_executor(x.executor(), y.executor(), "core::dot");
  return x.dot(y);
}

double norm2(const Vector& x) {
  return x.norm2();
}

void copy(const MultiVector& x, MultiVector& y) {
  require_same_executor(x.executor(), y.executor(), "core::copy(MultiVector)");
  y.copy_from(x);
}

void set_zero(MultiVector& x) {
  x.set_zero();
}

void scal(double alpha, MultiVector& x) {
  x.scale(alpha);
}

void copy_column_from(MultiVector& X, FASTEIG_INT col, const Vector& x) {
  require_same_executor(X.executor(), x.executor(), "core::copy_column_from");
  X.copy_column_from(col, x);
}

VectorXd copy_column_to_host(const MultiVector& X, FASTEIG_INT col) {
  return X.copy_column_to_host(col);
}

double column_dot(const MultiVector& X, FASTEIG_INT col, const Vector& x) {
  require_same_executor(X.executor(), x.executor(), "core::column_dot");
  return X.column_dot(col, x);
}

void column_axpy(double alpha, const MultiVector& X, FASTEIG_INT col, Vector& y) {
  require_same_executor(X.executor(), y.executor(), "core::column_axpy");
  X.column_axpy(alpha, col, y);
}

void orthogonalize_active_columns(const MultiVector& X,
                                  FASTEIG_INT active_cols,
                                  const Vector& source,
                                  Vector& residual,
                                  std::vector<double>& coeffs) {
  require_same_executor(X.executor(), source.executor(), "core::orthogonalize_active_columns(X,source)");
  require_same_executor(X.executor(), residual.executor(), "core::orthogonalize_active_columns(X,residual)");
  X.orthogonalize_active_columns(active_cols, source, residual, coeffs);
}

double orthogonalize_active_columns(const MultiVector& X,
                                    FASTEIG_INT active_cols,
                                    const Vector& source,
                                    Vector& residual) {
  require_same_executor(X.executor(), source.executor(), "core::orthogonalize_active_columns(X,source)");
  require_same_executor(X.executor(), residual.executor(), "core::orthogonalize_active_columns(X,residual)");
  return X.orthogonalize_active_columns(active_cols, source, residual);
}

void spmv(const CsrMatrix& A, const Vector& x, Vector& y, double alpha, double beta) {
  apply(A, x, y, alpha, beta);
}

void spmv_column(const CsrMatrix& A,
                 const MultiVector& X,
                 FASTEIG_INT col,
                 Vector& y,
                 double alpha,
                 double beta) {
  apply_column(A, X, col, y, alpha, beta);
}

void spmv_block(const CsrMatrix& A,
                const MultiVector& X,
                FASTEIG_INT first_col,
                FASTEIG_INT block_size,
                MultiVector& Y,
                double alpha,
                double beta) {
  apply_block(A, X, first_col, block_size, Y, alpha, beta);
}

void apply(const LinearOperator& op, const Vector& x, Vector& y, double alpha, double beta) {
  require_same_executor(op.executor(), x.executor(), "core::apply(op,x)");
  require_same_executor(op.executor(), y.executor(), "core::apply(op,y)");
  op.apply(x, y, alpha, beta);
}

void apply_column(const LinearOperator& op,
                  const MultiVector& X,
                  FASTEIG_INT col,
                  Vector& y,
                  double alpha,
                  double beta) {
  require_same_executor(op.executor(), X.executor(), "core::apply_column(op,X)");
  require_same_executor(op.executor(), y.executor(), "core::apply_column(op,y)");
  op.apply_column(X, col, y, alpha, beta);
}

void apply_block(const LinearOperator& op,
                 const MultiVector& X,
                 FASTEIG_INT first_col,
                 FASTEIG_INT block_size,
                 MultiVector& Y,
                 double alpha,
                 double beta) {
  require_same_executor(op.executor(), X.executor(), "core::apply_block(op,X)");
  require_same_executor(op.executor(), Y.executor(), "core::apply_block(op,Y)");
  op.apply_block(X, first_col, block_size, Y, alpha, beta);
}

MatrixXd apply_to_host(const LinearOperator& op,
                       const Eigen::Ref<const MatrixXd>& X,
                       double alpha,
                       double beta) {
  if (X.rows() != op.cols()) {
    throw std::invalid_argument("core::apply_to_host: input row count does not match operator columns.");
  }
  if (X.cols() == 0) {
    return MatrixXd(static_cast<int>(op.rows()), 0);
  }

  auto x_exec = op.executor().create_multivector(X);
  auto y_exec = op.executor().create_multivector(op.rows(), static_cast<FASTEIG_INT>(X.cols()));
  if (beta != 0.0) {
    throw std::invalid_argument("core::apply_to_host: beta != 0 requires an existing destination.");
  }
  apply_block(op, *x_exec, 0, static_cast<FASTEIG_INT>(X.cols()), *y_exec, alpha, beta);
  op.executor().synchronize();
  return y_exec->copy_to_host();
}

}  // namespace core
}  // namespace fasteig

