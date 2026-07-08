#pragma once

#include "fasteig.hpp"

namespace fasteig {
namespace core {

class MultiVector;

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
                             QrMethod method = QrMethod::Householder);

FASTEIG_INT block_gram_schmidt(const MultiVector& basis,
                               FASTEIG_INT active_cols,
                               const MultiVector& source,
                               MultiVector& residual,
                               MatrixXd& H);

FASTEIG_INT orthogonalize_block_against_basis(MultiVector& basis,
                                              FASTEIG_INT active_cols,
                                              const MultiVector& source,
                                              MultiVector& residual,
                                              MatrixXd& H,
                                              MatrixXd& R,
                                              QrMethod qr_method = QrMethod::Householder);

}  // namespace core
}  // namespace fasteig

