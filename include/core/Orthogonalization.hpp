#pragma once

#include "fasteig.hpp"

#include <memory>

namespace fasteig {
namespace core {

class MultiVector;
class LinearOperator;

enum class QrMethod {
  Householder,
  Cholesky,
  Cholesky2,
  GramSchmidt,
  Svqb
};

FASTEIG_INT cholesky_qr(const Eigen::Ref<const MatrixXd>& X,
                        MatrixXd& Q,
                        MatrixXd& R);

FASTEIG_INT cholesky_qr2(const Eigen::Ref<const MatrixXd>& X,
                         MatrixXd& Q,
                         MatrixXd& R);

FASTEIG_INT gram_schmidt_qr(const Eigen::Ref<const MatrixXd>& X,
                            MatrixXd& Q,
                            MatrixXd& R,
                            int passes = 2);

FASTEIG_INT householder_qr(const Eigen::Ref<const MatrixXd>& X,
                           MatrixXd& Q,
                           MatrixXd& R);

FASTEIG_INT svqb(MatrixXd& U,
                 double tau = 1e-14,
                 char method = 'R');

FASTEIG_INT thin_qr_in_place(MultiVector& X,
                             MatrixXd& R,
                             QrMethod method = QrMethod::Householder,
                             const LinearOperator* metric = nullptr);

FASTEIG_INT block_gram_schmidt(const MultiVector& basis,
                               FASTEIG_INT active_cols,
                               const MultiVector& source,
                               MultiVector& residual,
                               MatrixXd& H,
                               const MultiVector* dual_basis = nullptr);

// QR of a residual block in the complement of basis. Use Cholesky QR for
// independent columns; after numerical rank loss, reorthogonalize and complete
// with deterministic random directions. Zero padding is used only once the
// ambient space is exhausted. R always describes the original residual, never
// the random completion. Returns true if random completion was needed.
// Only small Gram/coefficient matrices reach the host.
bool complete_block_qr(const MultiVector& basis, FASTEIG_INT active_cols,
                       const MultiVector& source, MultiVector& Q, MatrixXd& R,
                       const LinearOperator* metric,
                       const MultiVector* dual_basis,
                       unsigned int seed, double reference_norm);

// Orthonormal basis for the part of source outside basis(:,0:active_cols).
// Drop numerically dependent directions instead of completing QR with arbitrary
// columns. Returns nullptr if no direction survives. Large blocks stay on the
// executor; rank decisions use only the small Householder R factor on the host.
std::unique_ptr<MultiVector> orthonormalize_block(
    const MultiVector& basis,
    FASTEIG_INT active_cols,
    const MultiVector& source,
    double rank_tolerance = 1e-12);

FASTEIG_INT orthogonalize_block_against_basis(MultiVector& basis,
                                              FASTEIG_INT active_cols,
                                              const MultiVector& source,
                                              MultiVector& residual,
                                              MatrixXd& H,
                                              MatrixXd& R,
                                              QrMethod qr_method = QrMethod::Householder);

}  // namespace core
}  // namespace fasteig

