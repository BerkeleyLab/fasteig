#pragma once

#include "core/Executor.hpp"
#include "fasteig.hpp"

namespace fasteig {
namespace core {

class MultiVector {
public:
  virtual ~MultiVector() = default;

  virtual const Executor& executor() const noexcept = 0;
  virtual FASTEIG_INT rows() const noexcept = 0;
  virtual FASTEIG_INT cols() const noexcept = 0;

  virtual void copy_from_host(const Eigen::Ref<const MatrixXd>& src) = 0;
  virtual MatrixXd copy_to_host() const = 0;

  virtual void copy_from(const MultiVector& src) = 0;
  virtual void set_zero() = 0;
  virtual void scale(double alpha) = 0;
};

class CpuMultiVector final : public MultiVector {
public:
  CpuMultiVector(const Executor& exec, FASTEIG_INT rows, FASTEIG_INT cols);

  const Executor& executor() const noexcept override { return exec_; }
  FASTEIG_INT rows() const noexcept override {
    return static_cast<FASTEIG_INT>(values_.rows());
  }
  FASTEIG_INT cols() const noexcept override {
    return static_cast<FASTEIG_INT>(values_.cols());
  }

  void copy_from_host(const Eigen::Ref<const MatrixXd>& src) override;
  MatrixXd copy_to_host() const override { return values_; }

  void copy_from(const MultiVector& src) override;
  void set_zero() override { values_.setZero(); }
  void scale(double alpha) override { values_ *= alpha; }

  MatrixXd& values() noexcept { return values_; }
  const MatrixXd& values() const noexcept { return values_; }

private:
  const Executor& exec_;
  MatrixXd values_;
};

}  // namespace core
}  // namespace fasteig
