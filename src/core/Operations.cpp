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

void spmv(const CsrMatrix& A, const Vector& x, Vector& y, double alpha, double beta) {
  require_same_executor(A.executor(), x.executor(), "core::spmv(A,x)");
  require_same_executor(A.executor(), y.executor(), "core::spmv(A,y)");
  A.spmv(x, y, alpha, beta);
}

}  // namespace core
}  // namespace fasteig
