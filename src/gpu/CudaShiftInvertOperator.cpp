#include "gpu/CudaShiftInvertOperator.hpp"

#if defined(USE_CUDA) && defined(USE_CUDSS)

#include "core/CsrMatrix.hpp"
#include "core/MultiVector.hpp"
#include "core/Vector.hpp"
#include "gpu/CudaMultiVector.hpp"
#include "gpu/CudaVector.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

namespace fasteig {
namespace core {
namespace {

CsrMatrixData shifted_csr(CsrMatrixData matrix, double shift) {
  validate_csr_matrix_data(matrix);
  if (matrix.rows != matrix.cols) {
    throw std::invalid_argument("CudaShiftInvertOperator: shifted matrix must be square.");
  }
  if (shift == 0.0) return matrix;

  CsrMatrixData shifted;
  shifted.rows = matrix.rows;
  shifted.cols = matrix.cols;
  shifted.rowptr.resize(static_cast<std::size_t>(matrix.rows + 1), 0);

  for (FASTEIG_INT row = 0; row < matrix.rows; ++row) {
    std::vector<std::pair<int, double>> entries;
    bool has_diag = false;
    const int begin = matrix.rowptr[static_cast<std::size_t>(row)];
    const int end = matrix.rowptr[static_cast<std::size_t>(row + 1)];
    entries.reserve(static_cast<std::size_t>(end - begin + 1));

    for (int k = begin; k < end; ++k) {
      const int col = matrix.colind[static_cast<std::size_t>(k)];
      double value = matrix.values[static_cast<std::size_t>(k)];
      if (col == row) {
        value -= shift;
        has_diag = true;
      }
      entries.emplace_back(col, value);
    }
    if (!has_diag) {
      entries.emplace_back(static_cast<int>(row), -shift);
    }

    std::sort(entries.begin(), entries.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    shifted.rowptr[static_cast<std::size_t>(row)] =
        static_cast<int>(shifted.colind.size());
    for (const auto& [col, value] : entries) {
      shifted.colind.push_back(col);
      shifted.values.push_back(value);
    }
  }
  shifted.rowptr[static_cast<std::size_t>(matrix.rows)] =
      static_cast<int>(shifted.colind.size());
  return shifted;
}

}  // namespace

CudaShiftInvertOperator::CudaShiftInvertOperator(const Executor& exec,
                                                 CsrMatrixData matrix,
                                                 double shift,
                                                 bool assume_symmetric)
    : exec_(exec),
      rows_(matrix.rows),
      cols_(matrix.cols),
      shift_(shift),
      symmetric_mode_(assume_symmetric) {
  CsrMatrixData shifted = shifted_csr(std::move(matrix), shift_);
  // Use general/full as the conservative CUDA SAI path.  cuDSS symmetric views
  // are more restrictive about accepted CSR descriptors and can reject full CSR
  // matrices before analysis.
  solver_ = std::make_unique<gpu::CudaDssSolver>(
      exec_, std::move(shifted),
      assume_symmetric ? CUDSS_MTYPE_SYMMETRIC : CUDSS_MTYPE_GENERAL,
      assume_symmetric ? CUDSS_MVIEW_LOWER : CUDSS_MVIEW_FULL);
}

void CudaShiftInvertOperator::apply(const Vector& x, Vector& y, double alpha, double beta) const {
  if (x.size() != cols_ || y.size() != rows_) {
    throw std::invalid_argument("CudaShiftInvertOperator::apply: dimension mismatch.");
  }
  const auto* cuda_x = dynamic_cast<const CudaVector*>(&x);
  auto* cuda_y = dynamic_cast<CudaVector*>(&y);
  if (!cuda_x || !cuda_y) {
    throw std::invalid_argument("CudaShiftInvertOperator::apply: x and y must be CUDA vectors.");
  }

  if (alpha == 1.0 && beta == 0.0 && cuda_x->data() != cuda_y->data()) {
    solver_->solve(cuda_x->data(), cuda_y->data());
    return;
  }

  auto tmp = exec_.create_vector(rows_);
  auto* cuda_tmp = dynamic_cast<CudaVector*>(tmp.get());
  if (!cuda_tmp) {
    throw std::logic_error("CudaShiftInvertOperator::apply: temporary vector is not CUDA-backed.");
  }
  solver_->solve(cuda_x->data(), cuda_tmp->data());
  y.scale(beta);
  y.axpy(alpha, *tmp);
}

void CudaShiftInvertOperator::apply_column(const MultiVector& X,
                                           FASTEIG_INT col,
                                           Vector& y,
                                           double alpha,
                                           double beta) const {
  if (col < 0 || col >= X.cols()) {
    throw std::invalid_argument("CudaShiftInvertOperator::apply_column: column out of range.");
  }
  if (X.rows() != cols_ || y.size() != rows_) {
    throw std::invalid_argument("CudaShiftInvertOperator::apply_column: dimension mismatch.");
  }
  const auto* cuda_X = dynamic_cast<const CudaMultiVector*>(&X);
  auto* cuda_y = dynamic_cast<CudaVector*>(&y);
  if (!cuda_X || !cuda_y) {
    throw std::invalid_argument(
        "CudaShiftInvertOperator::apply_column: X and y must be CUDA objects.");
  }

  const double* d_rhs = cuda_X->data() + static_cast<std::size_t>(col) *
                                            static_cast<std::size_t>(X.rows());
  if (alpha == 1.0 && beta == 0.0) {
    solver_->solve(d_rhs, cuda_y->data());
    return;
  }

  auto tmp = exec_.create_vector(rows_);
  auto* cuda_tmp = dynamic_cast<CudaVector*>(tmp.get());
  if (!cuda_tmp) {
    throw std::logic_error("CudaShiftInvertOperator::apply_column: temporary vector is not CUDA-backed.");
  }
  solver_->solve(d_rhs, cuda_tmp->data());
  y.scale(beta);
  y.axpy(alpha, *tmp);
}

void CudaShiftInvertOperator::apply_block(const MultiVector& X,
                                          FASTEIG_INT first_col,
                                          FASTEIG_INT block_size,
                                          MultiVector& Y,
                                          double alpha,
                                          double beta) const {
  if (first_col < 0 || block_size <= 0 || first_col + block_size > X.cols()) {
    throw std::invalid_argument("CudaShiftInvertOperator::apply_block: invalid column range.");
  }
  if (X.rows() != cols_ || Y.rows() != rows_ || Y.cols() < block_size) {
    throw std::invalid_argument("CudaShiftInvertOperator::apply_block: dimension mismatch.");
  }
  const auto* cuda_X = dynamic_cast<const CudaMultiVector*>(&X);
  auto* cuda_Y = dynamic_cast<CudaMultiVector*>(&Y);
  if (!cuda_X || !cuda_Y) {
    throw std::invalid_argument(
        "CudaShiftInvertOperator::apply_block: X and Y must be CUDA multivectors.");
  }

  const double* d_rhs = cuda_X->data() + static_cast<std::size_t>(first_col) *
                                            static_cast<std::size_t>(X.rows());
  if (alpha == 1.0 && beta == 0.0) {
    solver_->solve(d_rhs, cuda_Y->data(), block_size, X.rows(), Y.rows());
    return;
  }

  for (FASTEIG_INT j = 0; j < block_size; ++j) {
    auto y_col = exec_.create_vector(rows_);
    if (beta != 0.0) {
      y_col->copy_from_host(Y.copy_column_to_host(j));
    }
    apply_column(X, first_col + j, *y_col, alpha, beta);
    Y.copy_column_from(j, *y_col);
  }
}

}  // namespace core
}  // namespace fasteig

#endif  // defined(USE_CUDA) && defined(USE_CUDSS)

