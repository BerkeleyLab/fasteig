#pragma once

#include "core/Executor.hpp"
#include "core/Vector.hpp"
#include "fasteig.hpp"

#include <vector>

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
  virtual void copy_columns_from(FASTEIG_INT dst_first_col,
                                 const MultiVector& src,
                                 FASTEIG_INT src_first_col,
                                 FASTEIG_INT column_count) = 0;
  virtual void set_zero() = 0;
  virtual void scale(double alpha) = 0;
  virtual void copy_column_from(FASTEIG_INT col, const Vector& src) = 0;
  virtual void copy_column_to(FASTEIG_INT col, Vector& dst) const = 0;
  virtual VectorXd copy_column_to_host(FASTEIG_INT col) const = 0;
  virtual double column_dot(FASTEIG_INT col, const Vector& x) const = 0;
  virtual void column_axpy(double alpha, FASTEIG_INT col, Vector& y) const = 0;
  // Two-pass Gram-Schmidt: project with dual_basis^T, subtract with this
  // basis. Pass B*Q as dual_basis for a B-orthonormal Q; nullptr uses Q itself.
  virtual double orthogonalize_active_columns(FASTEIG_INT active_cols,
                                              const Vector& source,
                                              Vector& residual,
                                              const MultiVector* dual_basis = nullptr) const = 0;
  virtual void orthogonalize_active_columns(FASTEIG_INT active_cols,
                                            const Vector& source,
                                            Vector& residual,
                                            std::vector<double>& coeffs) const = 0;
  virtual MatrixXd multiply_active_columns(FASTEIG_INT active_cols,
                                           const Eigen::Ref<const MatrixXd>& coeffs) const = 0;
  virtual void right_multiply_active_columns(
      FASTEIG_INT active_cols,
      const Eigen::Ref<const MatrixXd>& coeffs,
      MultiVector& result) const = 0;
  virtual MatrixXd transpose_multiply(FASTEIG_INT active_cols,
                                      const MultiVector& right,
                                      FASTEIG_INT right_active_cols) const = 0;
  virtual void add_scaled_columns(const MultiVector& x,
                                  const Eigen::Ref<const VectorXd>& scales) = 0;
  virtual VectorXd column_norms() const = 0;
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
  void copy_columns_from(FASTEIG_INT dst_first_col,
                         const MultiVector& src,
                         FASTEIG_INT src_first_col,
                         FASTEIG_INT column_count) override;
  void set_zero() override { values_.setZero(); }
  void scale(double alpha) override { values_ *= alpha; }
  void copy_column_from(FASTEIG_INT col, const Vector& src) override;
  void copy_column_to(FASTEIG_INT col, Vector& dst) const override;
  VectorXd copy_column_to_host(FASTEIG_INT col) const override;
  double column_dot(FASTEIG_INT col, const Vector& x) const override;
  void column_axpy(double alpha, FASTEIG_INT col, Vector& y) const override;
  double orthogonalize_active_columns(FASTEIG_INT active_cols,
                                      const Vector& source,
                                      Vector& residual,
                                      const MultiVector* dual_basis = nullptr) const override;
  void orthogonalize_active_columns(FASTEIG_INT active_cols,
                                    const Vector& source,
                                    Vector& residual,
                                    std::vector<double>& coeffs) const override;
  MatrixXd multiply_active_columns(FASTEIG_INT active_cols,
                                   const Eigen::Ref<const MatrixXd>& coeffs) const override;
  void right_multiply_active_columns(
      FASTEIG_INT active_cols,
      const Eigen::Ref<const MatrixXd>& coeffs,
      MultiVector& result) const override;
  MatrixXd transpose_multiply(FASTEIG_INT active_cols,
                              const MultiVector& right,
                              FASTEIG_INT right_active_cols) const override;
  void add_scaled_columns(const MultiVector& x,
                          const Eigen::Ref<const VectorXd>& scales) override;
  VectorXd column_norms() const override;

  MatrixXd& values() noexcept { return values_; }
  const MatrixXd& values() const noexcept { return values_; }

private:
  const Executor& exec_;
  MatrixXd values_;
};

}  // namespace core
}  // namespace fasteig

