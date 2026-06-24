#pragma once

#ifdef USE_CUDA

#include "core/MultiVector.hpp"

namespace fasteig {
namespace core {

class CudaMultiVector final : public MultiVector {
public:
  CudaMultiVector(const Executor& exec, FASTEIG_INT rows, FASTEIG_INT cols);
  ~CudaMultiVector() override;

  CudaMultiVector(const CudaMultiVector&) = delete;
  CudaMultiVector& operator=(const CudaMultiVector&) = delete;

  const Executor& executor() const noexcept override { return exec_; }
  FASTEIG_INT rows() const noexcept override { return rows_; }
  FASTEIG_INT cols() const noexcept override { return cols_; }

  void copy_from_host(const Eigen::Ref<const MatrixXd>& src) override;
  MatrixXd copy_to_host() const override;

  void copy_from(const MultiVector& src) override;
  void set_zero() override;
  void scale(double alpha) override;

  double* data() noexcept { return d_values_; }
  const double* data() const noexcept { return d_values_; }

private:
  const Executor& exec_;
  FASTEIG_INT rows_ = 0;
  FASTEIG_INT cols_ = 0;
  double* d_values_ = nullptr;
};

}  // namespace core
}  // namespace fasteig

#endif  // USE_CUDA

