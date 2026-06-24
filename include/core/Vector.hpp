#pragma once

#include "core/Executor.hpp"
#include "fasteig.hpp"

namespace fasteig {
namespace core {

class Vector {
public:
  virtual ~Vector() = default;

  virtual const Executor& executor() const noexcept = 0;
  virtual FASTEIG_INT size() const noexcept = 0;

  virtual void copy_from_host(const Eigen::Ref<const VectorXd>& src) = 0;
  virtual VectorXd copy_to_host() const = 0;

  virtual void copy_from(const Vector& src) = 0;
  virtual void set_zero() = 0;
  virtual void scale(double alpha) = 0;
  virtual void axpy(double alpha, const Vector& x) = 0;
  virtual double dot(const Vector& x) const = 0;
  virtual double norm2() const = 0;
};

class CpuVector final : public Vector {
public:
  CpuVector(const Executor& exec, FASTEIG_INT size);

  const Executor& executor() const noexcept override { return exec_; }
  FASTEIG_INT size() const noexcept override {
    return static_cast<FASTEIG_INT>(values_.size());
  }

  void copy_from_host(const Eigen::Ref<const VectorXd>& src) override;
  VectorXd copy_to_host() const override { return values_; }

  void copy_from(const Vector& src) override;
  void set_zero() override { values_.setZero(); }
  void scale(double alpha) override { values_ *= alpha; }
  void axpy(double alpha, const Vector& x) override;
  double dot(const Vector& x) const override;
  double norm2() const override { return values_.norm(); }

  VectorXd& values() noexcept { return values_; }
  const VectorXd& values() const noexcept { return values_; }

private:
  const Executor& exec_;
  VectorXd values_;
};

}  // namespace core
}  // namespace fasteig
