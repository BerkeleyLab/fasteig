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
  void copy_column_from(FASTEIG_INT col, const Vector& src) override;
  VectorXd copy_column_to_host(FASTEIG_INT col) const override;
  double column_dot(FASTEIG_INT col, const Vector& x) const override;
  void column_axpy(double alpha, FASTEIG_INT col, Vector& y) const override;
  double orthogonalize_active_columns(FASTEIG_INT active_cols,
                                      const Vector& source,
                                      Vector& residual) const override;
  void orthogonalize_active_columns(FASTEIG_INT active_cols,
                                    const Vector& source,
                                    Vector& residual,
                                    std::vector<double>& coeffs) const override;
  MatrixXd multiply_active_columns(FASTEIG_INT active_cols,
                                   const Eigen::Ref<const MatrixXd>& coeffs) const override;

  void orthogonalize_active_columns_device_coeffs(FASTEIG_INT active_cols,
                                                  const Vector& source,
                                                  Vector& residual,
                                                  double* d_coeffs,
                                                  double* d_correction) const;

  double* data() noexcept { return d_values_; }
  const double* data() const noexcept { return d_values_; }

private:
  void ensure_orthog_workspace(FASTEIG_INT active_cols) const;

  const Executor& exec_;
  FASTEIG_INT rows_ = 0;
  FASTEIG_INT cols_ = 0;
  double* d_values_ = nullptr;
  mutable double* d_h_ = nullptr;
  mutable double* d_s_ = nullptr;
  mutable FASTEIG_INT orthog_workspace_cols_ = 0;
};

}  // namespace core
}  // namespace fasteig

#endif  // USE_CUDA

