#include "fasteig.hpp"
#include <cmath>
#include <limits>

// KrylovSubspace: unified Krylov basis construction.
// - Vector version (Lanczos): uses ops.matvec
// - Block version (Block Lanczos): uses ops.block_matvec

void KrylovSubspace(VectorXd& f,
                    MatrixXd& Q,
                    MatrixXd& T,
                    const operations& ops,
                    size_t begin,
                    size_t end) {
  const size_t n = f.size();

  if (end == 0) throw std::runtime_error("Requested Krylov subspace of dim 0 in KrylovSubspace");
  if (end > n) throw std::runtime_error("Requested Krylov subspace dim > matrix size in KrylovSubspace");
  if (!ops.matvec) throw std::runtime_error("KrylovSubspace (vector): ops.matvec is not set");

  double beta;
  VectorXd v(n), Av(n);

  for (size_t i = begin; i < end; ++i) {
    beta = f.norm();
    if (!std::isfinite(beta)) {
      throw std::runtime_error("KrylovSubspace: non-finite residual norm encountered.");
    }
    // Handle breakdown: if f is (near) zero, restart with a random vector
    if (beta <= std::numeric_limits<double>::epsilon()) {
      f = VectorXd::Random(n);
      if (i > 0) {
        f -= Q.leftCols(i) * (Q.leftCols(i).transpose() * f);
      }
      beta = f.norm();
      if (beta <= std::numeric_limits<double>::epsilon()) {
        throw std::runtime_error("KrylovSubspace: breakdown, unable to generate new basis vector");
      }
    }
    v = f / beta;
    Q.col(i) = v;

    // Fill the symmetric tridiagonal coupling coefficient (beta).
    // NOTE: This routine is used by multiple algorithms with different restart conventions.
    // In particular, Krylov-Schur uses a restart step where the coupling at the start of a
    // multi-step expansion (i == begin, end-begin > 1) is handled outside this helper.
    if (i != begin || (begin != 0 && end - begin == 1)) {
      T(i, i - 1) = beta;
      T(i - 1, i) = beta;
    }

    ops.matvec(v, Av);
    if (!Av.allFinite()) {
      throw std::runtime_error("KrylovSubspace: non-finite value encountered in matvec output.");
    }

    VectorXd h = Q.leftCols(i + 1).transpose() * Av;
    f = Av - Q.leftCols(i + 1) * h;

    VectorXd s = Q.leftCols(i + 1).transpose() * f;
    h += s;
    f -= Q.leftCols(i + 1) * s;

    T(i, i) = h(i);
  }
}

void KrylovSubspace(const MatrixXd& V0,
                    MatrixXd& Q,
                    MatrixXd& T,
                    const operations& ops,
                    size_t begin,
                    size_t end,
                    MatrixXd& beta) {
  const size_t n = V0.rows();
  const size_t b = V0.cols();

  if (end == 0) throw std::runtime_error("Requested Krylov subspace of dim 0 in KrylovSubspace");
  if (end > n) throw std::runtime_error("Requested Krylov subspace dim > matrix size in KrylovSubspace");
  if (!ops.block_matvec) throw std::runtime_error("KrylovSubspace (block): ops.block_matvec is not set");

  MatrixXd f(n, b), V(n, b), AV(n, b);

  if (begin == 0) {
    Eigen::HouseholderQR<MatrixXd> qrInit(V0);
    MatrixXd Q_block0 = MatrixXd::Identity(n, b);
    Q_block0 = qrInit.householderQ() * Q_block0;
    Q.block(0, 0, n, b) = Q_block0;
  }

  for (size_t i = begin + 1; i <= end; ++i) {
    V = Q.block(0, (i - 1) * b, n, b);

    ops.block_matvec(V, AV);

    MatrixXd H = Q.leftCols(i * b).transpose() * AV;
    f = AV - Q.leftCols(i * b) * H;

    MatrixXd S = Q.leftCols(i * b).transpose() * f;
    H += S;
    f -= Q.leftCols(i * b) * S;

    Eigen::HouseholderQR<MatrixXd> qrR(f);
    MatrixXd Q_new = MatrixXd::Identity(n, b);
    MatrixXd R_top = qrR.matrixQR().topLeftCorner(b, b).template triangularView<Eigen::Upper>();
    Q_new = qrR.householderQ() * Q_new;
    Q.block(0, i * b, n, b) = Q_new;
    beta = R_top;

    T.block((i - 1) * b, (i - 1) * b, b, b) = H.bottomRows(b);
    T.block(i * b, (i - 1) * b, b, b) = beta;
    T.block((i - 1) * b, i * b, b, b) = beta.transpose();
  }
}

