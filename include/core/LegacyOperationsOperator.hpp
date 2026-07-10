#pragma once

#include "core/LinearOperator.hpp"
#include "fasteig.hpp"

namespace fasteig {
namespace core {

class LegacyOperationsOperator final : public LinearOperator {
public:
  LegacyOperationsOperator(const Executor& exec,
                           FASTEIG_INT n,
                           const operations& ops);

  const Executor& executor() const noexcept override { return exec_; }
  FASTEIG_INT rows() const noexcept override { return n_; }
  FASTEIG_INT cols() const noexcept override { return n_; }

  void apply(const Vector& x,
             Vector& y,
             double alpha = 1.0,
             double beta = 0.0) const override;

  void apply_column(const MultiVector& X,
                    FASTEIG_INT col,
                    Vector& y,
                    double alpha = 1.0,
                    double beta = 0.0) const override;

  void apply_block(const MultiVector& X,
                   FASTEIG_INT first_col,
                   FASTEIG_INT block_size,
                   MultiVector& Y,
                   double alpha = 1.0,
                   double beta = 0.0) const override;

private:
  const Executor& exec_;
  FASTEIG_INT n_ = 0;
  const operations& ops_;
};

}  // namespace core
}  // namespace fasteig

