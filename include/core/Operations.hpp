#pragma once

#include "core/CsrMatrix.hpp"
#include "core/MultiVector.hpp"
#include "core/Vector.hpp"

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

void spmv(const CsrMatrix& A,
          const Vector& x,
          Vector& y,
          double alpha = 1.0,
          double beta = 0.0);

}  // namespace core
}  // namespace fasteig
