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

void CpuMultiVector::copy_columns_from(FASTEIG_INT dst_first_col,
                                       const MultiVector& src,
                                       FASTEIG_INT src_first_col,
                                       FASTEIG_INT column_count) {
  if (&src.executor() != &exec_) {
    throw std::invalid_argument("CpuMultiVector::copy_columns_from: executor mismatch.");
  }
  if (dst_first_col < 0 || src_first_col < 0 || column_count < 0 ||
      dst_first_col + column_count > cols() ||
      src_first_col + column_count > src.cols() || src.rows() != rows()) {
    throw std::invalid_argument("CpuMultiVector::copy_columns_from: dimension mismatch.");
  }
  const auto* cpu_src = dynamic_cast<const CpuMultiVector*>(&src);
  if (!cpu_src) {
    throw std::invalid_argument("CpuMultiVector::copy_columns_from: source is not a CpuMultiVector.");
  }
  if (column_count == 0) return;
  values_.middleCols(static_cast<int>(dst_first_col), static_cast<int>(column_count)) =
      cpu_src->values_.middleCols(static_cast<int>(src_first_col),
                                  static_cast<int>(column_count));
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

void CpuMultiVector::copy_column_to(FASTEIG_INT col, Vector& dst) const {
  if (col < 0 || col >= cols()) {
    throw std::invalid_argument("CpuMultiVector::copy_column_to: column out of range.");
  }
  if (dst.size() != rows() || &dst.executor() != &exec_) {
    throw std::invalid_argument("CpuMultiVector::copy_column_to: destination mismatch.");
  }
  auto* cpu_dst = dynamic_cast<CpuVector*>(&dst);
  if (!cpu_dst) {
    throw std::invalid_argument("CpuMultiVector::copy_column_to: destination is not a CpuVector.");
  }
  cpu_dst->values() = values_.col(static_cast<int>(col));
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
  auto result = exec_.create_multivector(rows(), static_cast<FASTEIG_INT>(coeffs.cols()));
  right_multiply_active_columns(active_cols, coeffs, *result);
  return result->copy_to_host();
}

void CpuMultiVector::right_multiply_active_columns(
    FASTEIG_INT active_cols,
    const Eigen::Ref<const MatrixXd>& coeffs,
    MultiVector& result) const {
  if (&result.executor() != &exec_) {
    throw std::invalid_argument("CpuMultiVector::right_multiply_active_columns: executor mismatch.");
  }
  if (active_cols < 0 || active_cols > cols() ||
      coeffs.rows() != static_cast<int>(active_cols) ||
      result.rows() != rows() || result.cols() != coeffs.cols()) {
    throw std::invalid_argument("CpuMultiVector::right_multiply_active_columns: dimension mismatch.");
  }
  auto* cpu_result = dynamic_cast<CpuMultiVector*>(&result);
  if (!cpu_result) {
    throw std::invalid_argument(
        "CpuMultiVector::right_multiply_active_columns: result is not a CpuMultiVector.");
  }
  cpu_result->values_.noalias() =
      values_.leftCols(static_cast<int>(active_cols)) * coeffs;
}

MatrixXd CpuMultiVector::transpose_multiply(FASTEIG_INT active_cols,
                                            const MultiVector& right,
                                            FASTEIG_INT right_active_cols) const {
  if (&right.executor() != &exec_ || right.rows() != rows() ||
      active_cols < 0 || active_cols > cols() ||
      right_active_cols < 0 || right_active_cols > right.cols()) {
    throw std::invalid_argument("CpuMultiVector::transpose_multiply: dimension or executor mismatch.");
  }
  const auto* cpu_right = dynamic_cast<const CpuMultiVector*>(&right);
  if (!cpu_right) {
    throw std::invalid_argument("CpuMultiVector::transpose_multiply: right is not a CpuMultiVector.");
  }
  return values_.leftCols(static_cast<int>(active_cols)).transpose() *
         cpu_right->values_.leftCols(static_cast<int>(right_active_cols));
}

void CpuMultiVector::add_scaled_columns(
    const MultiVector& x,
    const Eigen::Ref<const VectorXd>& scales) {
  if (&x.executor() != &exec_ || x.rows() != rows() || x.cols() != cols() ||
      scales.size() != cols()) {
    throw std::invalid_argument("CpuMultiVector::add_scaled_columns: dimension or executor mismatch.");
  }
  const auto* cpu_x = dynamic_cast<const CpuMultiVector*>(&x);
  if (!cpu_x) {
    throw std::invalid_argument("CpuMultiVector::add_scaled_columns: x is not a CpuMultiVector.");
  }
  for (int j = 0; j < values_.cols(); ++j) {
    values_.col(j).noalias() += scales(j) * cpu_x->values_.col(j);
  }
}

VectorXd CpuMultiVector::column_norms() const {
  return values_.colwise().norm().transpose();
}

}  // namespace core
}  // namespace fasteig

