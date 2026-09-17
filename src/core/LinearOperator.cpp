#include "core/LinearOperator.hpp"
#include "core/Executor.hpp"
#include "core/MultiVector.hpp"
#include "core/Vector.hpp"
#include <stdexcept>
#include <utility>

namespace fasteig {
namespace core {

ProductLinearOperator::ProductLinearOperator(std::unique_ptr<LinearOperator> left,
                                             std::shared_ptr<const LinearOperator> right)
    : left_(std::move(left)), right_(std::move(right)) {
  if (!left_ || !right_ || &left_->executor() != &right_->executor() ||
      left_->cols() != right_->rows()) {
    throw std::invalid_argument("ProductLinearOperator: incompatible operators.");
  }
  intermediate_ = executor().create_vector(right_->rows());
}

ProductLinearOperator::~ProductLinearOperator() = default;

void ProductLinearOperator::apply(const Vector& x, Vector& y, double alpha, double beta) const {
  right_->apply(x, *intermediate_);
  left_->apply(*intermediate_, y, alpha, beta);
}

void ProductLinearOperator::apply_column(const MultiVector& X, FASTEIG_INT col,
                                        Vector& y, double alpha, double beta) const {
  right_->apply_column(X, col, *intermediate_);
  left_->apply(*intermediate_, y, alpha, beta);
}

void ProductLinearOperator::apply_block(const MultiVector& X, FASTEIG_INT first_col,
                                       FASTEIG_INT block_size, MultiVector& Y,
                                       double alpha, double beta) const {
  if (block_size <= 0) throw std::invalid_argument("ProductLinearOperator: invalid block size.");
  if (!block_intermediate_ || block_intermediate_->cols() != block_size) {
    block_intermediate_ = executor().create_multivector(right_->rows(), block_size);
  }
  right_->apply_block(X, first_col, block_size, *block_intermediate_);
  left_->apply_block(*block_intermediate_, 0, block_size, Y, alpha, beta);
}

}  // namespace core
}  // namespace fasteig

