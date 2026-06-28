#pragma once

#include "fasteig.hpp"

namespace fasteig {
namespace core {

class Executor;
class MultiVector;
class Vector;

class LinearOperator {
public:
  virtual ~LinearOperator() = default;

  virtual const Executor& executor() const noexcept = 0;
  virtual FASTEIG_INT rows() const noexcept = 0;
  virtual FASTEIG_INT cols() const noexcept = 0;

  virtual void apply(const Vector& x,
                     Vector& y,
                     double alpha = 1.0,
                     double beta = 0.0) const = 0;

  virtual void apply_column(const MultiVector& X,
                            FASTEIG_INT col,
                            Vector& y,
                            double alpha = 1.0,
                            double beta = 0.0) const = 0;

  virtual void apply_block(const MultiVector& X,
                           FASTEIG_INT first_col,
                           FASTEIG_INT block_size,
                           MultiVector& Y,
                           double alpha = 1.0,
                           double beta = 0.0) const = 0;
};

}  // namespace core
}  // namespace fasteig

