#pragma once

#if defined(USE_CUDA) && defined(USE_CUDSS)

#include "core/LinearOperator.hpp"
#include "gpu/CudaDssSolver.hpp"

#include <memory>

namespace fasteig {
namespace core {

class CudaShiftInvertOperator final : public LinearOperator {
public:
  CudaShiftInvertOperator(const Executor& exec,
                          CsrMatrixData matrix,
                          double shift,
                          bool assume_symmetric = false);

  const Executor& executor() const noexcept override { return exec_; }
  FASTEIG_INT rows() const noexcept override { return rows_; }
  FASTEIG_INT cols() const noexcept override { return cols_; }

  void apply(const Vector& x, Vector& y, double alpha = 1.0, double beta = 0.0) const override;
  void apply_column(const MultiVector& X, FASTEIG_INT col, Vector& y,
                    double alpha = 1.0, double beta = 0.0) const override;
  void apply_block(const MultiVector& X, FASTEIG_INT first_col, FASTEIG_INT block_size,
                   MultiVector& Y, double alpha = 1.0, double beta = 0.0) const override;

  const gpu::CudaDssSolver& solver() const noexcept { return *solver_; }
  double shift() const noexcept { return shift_; }
  bool symmetric_mode() const noexcept { return symmetric_mode_; }

private:
  const Executor& exec_;
  FASTEIG_INT rows_ = 0;
  FASTEIG_INT cols_ = 0;
  double shift_ = 0.0;
  bool symmetric_mode_ = false;
  std::unique_ptr<gpu::CudaDssSolver> solver_;
};

}  // namespace core
}  // namespace fasteig

#endif  // defined(USE_CUDA) && defined(USE_CUDSS)

