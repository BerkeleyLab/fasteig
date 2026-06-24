#pragma once

#ifdef USE_CUDA

#include "core/Vector.hpp"

namespace fasteig {
namespace core {

class CudaVector final : public Vector {
public:
  CudaVector(const Executor& exec, FASTEIG_INT size);
  ~CudaVector() override;

  CudaVector(const CudaVector&) = delete;
  CudaVector& operator=(const CudaVector&) = delete;

  const Executor& executor() const noexcept override { return exec_; }
  FASTEIG_INT size() const noexcept override { return size_; }

  void copy_from_host(const Eigen::Ref<const VectorXd>& src) override;
  VectorXd copy_to_host() const override;

  void copy_from(const Vector& src) override;
  void set_zero() override;
  void scale(double alpha) override;
  void axpy(double alpha, const Vector& x) override;
  double dot(const Vector& x) const override;
  double norm2() const override;

  double* data() noexcept { return d_values_; }
  const double* data() const noexcept { return d_values_; }

private:
  const Executor& exec_;
  FASTEIG_INT size_ = 0;
  double* d_values_ = nullptr;
};

}  // namespace core
}  // namespace fasteig

#endif  // USE_CUDA

