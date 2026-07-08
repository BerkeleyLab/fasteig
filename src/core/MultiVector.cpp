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

void CpuMultiVector::copy_column_from(FASTEIG_INT col, const Vector& src) {
  if (col < 0 || col >= cols()) {
    throw std::invalid_argument("CpuMultiVector::copy_column_from: column out of range.");
  }
  if (src.size() != rows()) {
    throw std::invalid_argument("CpuMultiVector::copy_column_from: size mismatch.");
  }
  const auto* cpu_src = dynamic_cast<const CpuVector*>(&src);
  if (!cpu_src) {
    throw std::invalid_argument("CpuMultiVector::copy_column_from: source is not a CpuVector.");
  }
  values_.col(static_cast<int>(col)) = cpu_src->values();
}

VectorXd CpuMultiVector::copy_column_to_host(FASTEIG_INT col) const {
  if (col < 0 || col >= cols()) {
    throw std::invalid_argument("CpuMultiVector::copy_column_to_host: column out of range.");
  }
  return values_.col(static_cast<int>(col));
}

double CpuMultiVector::column_dot(FASTEIG_INT col, const Vector& x) const {
  if (col < 0 || col >= cols()) {
    throw std::invalid_argument("CpuMultiVector::column_dot: column out of range.");
  }
  if (x.size() != rows()) {
    throw std::invalid_argument("CpuMultiVector::column_dot: size mismatch.");
  }
  const auto* cpu_x = dynamic_cast<const CpuVector*>(&x);
  if (!cpu_x) {
    throw std::invalid_argument("CpuMultiVector::column_dot: x is not a CpuVector.");
  }
  return values_.col(static_cast<int>(col)).dot(cpu_x->values());
}

void CpuMultiVector::column_axpy(double alpha, FASTEIG_INT col, Vector& y) const {
  if (col < 0 || col >= cols()) {
    throw std::invalid_argument("CpuMultiVector::column_axpy: column out of range.");
  }
  if (y.size() != rows()) {
    throw std::invalid_argument("CpuMultiVector::column_axpy: size mismatch.");
  }
  auto* cpu_y = dynamic_cast<CpuVector*>(&y);
  if (!cpu_y) {
    throw std::invalid_argument("CpuMultiVector::column_axpy: y is not a CpuVector.");
  }
  cpu_y->values().noalias() += alpha * values_.col(static_cast<int>(col));
}

void CpuMultiVector::orthogonalize_active_columns(FASTEIG_INT active_cols,
                                                  const Vector& source,
                                                  Vector& residual,
                                                  std::vector<double>& coeffs) const {
  if (active_cols < 0 || active_cols > cols()) {
    throw std::invalid_argument("CpuMultiVector::orthogonalize_active_columns: invalid active column count.");
  }
  if (source.size() != rows() || residual.size() != rows()) {
    throw std::invalid_argument("CpuMultiVector::orthogonalize_active_columns: size mismatch.");
  }
  const auto* cpu_source = dynamic_cast<const CpuVector*>(&source);
  auto* cpu_residual = dynamic_cast<CpuVector*>(&residual);
  if (!cpu_source || !cpu_residual) {
    throw std::invalid_argument(
        "CpuMultiVector::orthogonalize_active_columns: source and residual must be CpuVector objects.");
  }

  coeffs.assign(static_cast<std::size_t>(active_cols), 0.0);
  cpu_residual->values() = cpu_source->values();
  if (active_cols == 0) return;

  const auto V_active = values_.leftCols(static_cast<int>(active_cols));
  Eigen::VectorXd h = V_active.transpose() * cpu_source->values();
  cpu_residual->values().noalias() -= V_active * h;

  Eigen::VectorXd s = V_active.transpose() * cpu_residual->values();
  cpu_residual->values().noalias() -= V_active * s;
  h += s;

  for (FASTEIG_INT j = 0; j < active_cols; ++j) {
    coeffs[static_cast<std::size_t>(j)] = h(static_cast<int>(j));
  }
}

double CpuMultiVector::orthogonalize_active_columns(FASTEIG_INT active_cols,
                                                    const Vector& source,
                                                    Vector& residual) const {
  if (active_cols < 0 || active_cols > cols()) {
    throw std::invalid_argument("CpuMultiVector::orthogonalize_active_columns: invalid active column count.");
  }
  if (source.size() != rows() || residual.size() != rows()) {
    throw std::invalid_argument("CpuMultiVector::orthogonalize_active_columns: size mismatch.");
  }
  const auto* cpu_source = dynamic_cast<const CpuVector*>(&source);
  auto* cpu_residual = dynamic_cast<CpuVector*>(&residual);
  if (!cpu_source || !cpu_residual) {
    throw std::invalid_argument(
        "CpuMultiVector::orthogonalize_active_columns: source and residual must be CpuVector objects.");
  }

  cpu_residual->values() = cpu_source->values();
  if (active_cols == 0) return 0.0;

  const auto V_active = values_.leftCols(static_cast<int>(active_cols));
  Eigen::VectorXd h = V_active.transpose() * cpu_source->values();
  cpu_residual->values().noalias() -= V_active * h;

  Eigen::VectorXd s = V_active.transpose() * cpu_residual->values();
  cpu_residual->values().noalias() -= V_active * s;
  h += s;

  return h(static_cast<int>(active_cols - 1));
}

MatrixXd CpuMultiVector::multiply_active_columns(FASTEIG_INT active_cols,
                                                 const Eigen::Ref<const MatrixXd>& coeffs) const {
  if (active_cols < 0 || active_cols > cols()) {
    throw std::invalid_argument("CpuMultiVector::multiply_active_columns: invalid active column count.");
  }
  if (coeffs.rows() != static_cast<int>(active_cols)) {
    throw std::invalid_argument("CpuMultiVector::multiply_active_columns: coefficient row mismatch.");
  }
  MatrixXd out(rows(), coeffs.cols());
  out.noalias() = values_.leftCols(static_cast<int>(active_cols)) * coeffs;
  return out;
}

}  // namespace core
}  // namespace fasteig

