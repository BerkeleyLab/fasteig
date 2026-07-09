#pragma once

#include "core/CsrMatrix.hpp"
#include "core/LinearOperator.hpp"
#include "core/MultiVector.hpp"
#include "core/Vector.hpp"

#include <vector>

namespace fasteig {
namespace core {

void copy(const Vector& x, Vector& y);
void set_zero(Vector& x);
void scal(double alpha, Vector& x);
void axpy(double alpha, const Vector& x, Vector& y);
double dot(const Vector& x, const Vector& y);
double norm2(const Vector& x);

void copy(const MultiVector& x, MultiVector& y);
void set_zero(MultiVector& x);
void scal(double alpha, MultiVector& x);
void copy_column_from(MultiVector& X, FASTEIG_INT col, const Vector& x);
VectorXd copy_column_to_host(const MultiVector& X, FASTEIG_INT col);
double column_dot(const MultiVector& X, FASTEIG_INT col, const Vector& x);
void column_axpy(double alpha, const MultiVector& X, FASTEIG_INT col, Vector& y);
void orthogonalize_active_columns(const MultiVector& X,
                                  FASTEIG_INT active_cols,
                                  const Vector& source,
                                  Vector& residual,
                                  std::vector<double>& coeffs);
double orthogonalize_active_columns(const MultiVector& X,
                                    FASTEIG_INT active_cols,
                                    const Vector& source,
                                    Vector& residual);

void spmv(const CsrMatrix& A,
          const Vector& x,
          Vector& y,
          double alpha = 1.0,
          double beta = 0.0);
void spmv_column(const CsrMatrix& A,
                 const MultiVector& X,
                 FASTEIG_INT col,
                 Vector& y,
                 double alpha = 1.0,
                 double beta = 0.0);
void spmv_block(const CsrMatrix& A,
                const MultiVector& X,
                FASTEIG_INT first_col,
                FASTEIG_INT block_size,
                MultiVector& Y,
                double alpha = 1.0,
                double beta = 0.0);

void apply(const LinearOperator& op,
           const Vector& x,
           Vector& y,
           double alpha = 1.0,
           double beta = 0.0);
void apply_column(const LinearOperator& op,
                  const MultiVector& X,
                  FASTEIG_INT col,
                  Vector& y,
                  double alpha = 1.0,
                  double beta = 0.0);
void apply_block(const LinearOperator& op,
                 const MultiVector& X,
                 FASTEIG_INT first_col,
                 FASTEIG_INT block_size,
                 MultiVector& Y,
                 double alpha = 1.0,
                 double beta = 0.0);

MatrixXd apply_to_host(const LinearOperator& op,
                       const Eigen::Ref<const MatrixXd>& X,
                       double alpha = 1.0,
                       double beta = 0.0);

}  // namespace core
}  // namespace fasteig

