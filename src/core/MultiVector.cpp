#include "core/MultiVector.hpp"

#include <stdexcept>

namespace fasteig {
namespace core {

CpuMultiVector::CpuMultiVector(const Executor& exec, FASTEIG_INT rows, FASTEIG_INT cols)
    : exec_(exec) {
  if (rows < 0 || cols < 0) {
    throw std::invalid_argument("CpuMultiVector: dimensions must be nonnegative.");
  }
  values_.setZero(static_cast<int>(rows), static_cast<int>(cols));
}

void CpuMultiVector::copy_from_host(const Eigen::Ref<const MatrixXd>& src) {
  if (src.rows() != values_.rows() || src.cols() != values_.cols()) {
    throw std::invalid_argument("CpuMultiVector::copy_from_host: dimension mismatch.");
  }
  values_ = src;
}

void CpuMultiVector::copy_from(const MultiVector& src) {
  if (src.rows() != rows() || src.cols() != cols()) {
    throw std::invalid_argument("CpuMultiVector::copy_from: dimension mismatch.");
  }
  const auto* cpu_src = dynamic_cast<const CpuMultiVector*>(&src);
  if (!cpu_src) {
    throw std::invalid_argument("CpuMultiVector::copy_from: source is not a CpuMultiVector.");
  }
  values_ = cpu_src->values_;
}

}  // namespace core
}  // namespace fasteig
