#include "core/Vector.hpp"

#include <stdexcept>

namespace fasteig {
namespace core {

CpuVector::CpuVector(const Executor& exec, FASTEIG_INT size) : exec_(exec) {
  if (size < 0) {
    throw std::invalid_argument("CpuVector: size must be nonnegative.");
  }
  values_.setZero(static_cast<int>(size));
}

void CpuVector::copy_from_host(const Eigen::Ref<const VectorXd>& src) {
  if (src.size() != values_.size()) {
    throw std::invalid_argument("CpuVector::copy_from_host: size mismatch.");
  }
  values_ = src;
}

void CpuVector::copy_from(const Vector& src) {
  if (src.size() != size()) {
    throw std::invalid_argument("CpuVector::copy_from: size mismatch.");
  }
  const auto* cpu_src = dynamic_cast<const CpuVector*>(&src);
  if (!cpu_src) {
    throw std::invalid_argument("CpuVector::copy_from: source is not a CpuVector.");
  }
  values_ = cpu_src->values_;
}

void CpuVector::axpy(double alpha, const Vector& x) {
  if (x.size() != size()) {
    throw std::invalid_argument("CpuVector::axpy: size mismatch.");
  }
  const auto* cpu_x = dynamic_cast<const CpuVector*>(&x);
  if (!cpu_x) {
    throw std::invalid_argument("CpuVector::axpy: x is not a CpuVector.");
  }
  values_.noalias() += alpha * cpu_x->values_;
}

double CpuVector::dot(const Vector& x) const {
  if (x.size() != size()) {
    throw std::invalid_argument("CpuVector::dot: size mismatch.");
  }
  const auto* cpu_x = dynamic_cast<const CpuVector*>(&x);
  if (!cpu_x) {
    throw std::invalid_argument("CpuVector::dot: x is not a CpuVector.");
  }
  return values_.dot(cpu_x->values_);
}

}  // namespace core
}  // namespace fasteig
