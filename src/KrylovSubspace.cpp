#include "KrylovSubspace.hpp"

#include <cmath>
#include <limits>

// KrylovSubspace: unified Krylov basis construction.
// - Vector version (Lanczos): uses ops.matvec
// - Block version (Block Lanczos): uses ops.block_matvec

KrylovSubspace::KrylovSubspace(VectorXd& KrylovResidual,
                               MatrixXd& OrthoBasis,
                               MatrixXd& ProjectedMatrix)
    : krylov_residual_(&KrylovResidual),
      ortho_basis_(&OrthoBasis),
      projected_matrix_(&ProjectedMatrix) {}

KrylovSubspace::KrylovSubspace(const MatrixXd& V0,
                               MatrixXd& OrthoBasis,
                               MatrixXd& ProjectedMatrix,
                               MatrixXd& beta)
    : v0_(&V0),
      ortho_basis_(&OrthoBasis),
      projected_matrix_(&ProjectedMatrix),
      beta_(&beta) {}

void KrylovSubspace::build(const operations& ops,
                           size_t begin,
                           size_t end) {
  if (!krylov_residual_ || !ortho_basis_ || !projected_matrix_) {
    throw std::logic_error("KrylovSubspace: vector workspace is not initialized.");
  }

  VectorXd& KrylovResidual = *krylov_residual_;
  MatrixXd& OrthoBasis = *ortho_basis_;
  MatrixXd& ProjectedMatrix = *projected_matrix_;
  const size_t n = KrylovResidual.size();

  if (end == 0) throw std::runtime_error("Requested Krylov subspace of dim 0 in KrylovSubspace");
  if (end > n) throw std::runtime_error("Requested Krylov subspace dim > matrix size in KrylovSubspace");
  if (!ops.matvec) throw std::runtime_error("KrylovSubspace (vector): ops.matvec is not set");

  VectorXd v(n), Av(n);

  for (size_t i = begin; i < end; ++i) {
    beta_scalar_ = KrylovResidual.norm();
    if (!std::isfinite(beta_scalar_)) {
      throw std::runtime_error("KrylovSubspace: non-finite residual norm encountered.");
    }
    // Handle breakdown: if f is (near) zero, restart with a random vector
    if (beta_scalar_ <= std::numeric_limits<double>::epsilon()) {
      KrylovResidual = VectorXd::Random(n);
      if (i > 0) {
        KrylovResidual -= OrthoBasis.leftCols(i) *
                          (OrthoBasis.leftCols(i).transpose() * KrylovResidual);
      }
      beta_scalar_ = KrylovResidual.norm();
      if (beta_scalar_ <= std::numeric_limits<double>::epsilon()) {
        throw std::runtime_error("KrylovSubspace: breakdown, unable to generate new basis vector");
      }
    }
    v = KrylovResidual / beta_scalar_;
    OrthoBasis.col(i) = v;

    // Fill the symmetric tridiagonal coupling coefficient (beta).
    // NOTE: This routine is used by multiple algorithms with different restart conventions.
    // In particular, Krylov-Schur uses a restart step where the coupling at the start of a
    // multi-step expansion (i == begin, end-begin > 1) is handled outside this helper.
    if (i != begin || (begin != 0 && end - begin == 1)) {
      ProjectedMatrix(i, i - 1) = beta_scalar_;
      ProjectedMatrix(i - 1, i) = beta_scalar_;
    }

    ops.matvec(v, Av);
    if (!Av.allFinite()) {
      throw std::runtime_error("KrylovSubspace: non-finite value encountered in matvec output.");
    }

    VectorXd h = OrthoBasis.leftCols(i + 1).transpose() * Av;
    KrylovResidual = Av - OrthoBasis.leftCols(i + 1) * h;

    VectorXd s = OrthoBasis.leftCols(i + 1).transpose() * KrylovResidual;
    h += s;
    KrylovResidual -= OrthoBasis.leftCols(i + 1) * s;

    ProjectedMatrix(i, i) = h(i);
  }
}

void KrylovSubspace::build_block(const operations& ops,
                                 size_t begin,
                                 size_t end) {
  if (!v0_ || !ortho_basis_ || !projected_matrix_ || !beta_) {
    throw std::logic_error("KrylovSubspace: block workspace is not initialized.");
  }

  const MatrixXd& V0 = *v0_;
  MatrixXd& OrthoBasis = *ortho_basis_;
  MatrixXd& ProjectedMatrix = *projected_matrix_;
  MatrixXd& beta = *beta_;
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
    OrthoBasis.block(0, 0, n, b) = Q_block0;
  }

  for (size_t i = begin + 1; i <= end; ++i) {
    V = OrthoBasis.block(0, (i - 1) * b, n, b);

    ops.block_matvec(V, AV);

    MatrixXd H = OrthoBasis.leftCols(i * b).transpose() * AV;
    f = AV - OrthoBasis.leftCols(i * b) * H;

    MatrixXd S = OrthoBasis.leftCols(i * b).transpose() * f;
    H += S;
    f -= OrthoBasis.leftCols(i * b) * S;

    Eigen::HouseholderQR<MatrixXd> qrR(f);
    MatrixXd Q_new = MatrixXd::Identity(n, b);
    MatrixXd R_top = qrR.matrixQR().topLeftCorner(b, b).template triangularView<Eigen::Upper>();
    Q_new = qrR.householderQ() * Q_new;
    OrthoBasis.block(0, i * b, n, b) = Q_new;
    beta = R_top;

    ProjectedMatrix.block((i - 1) * b, (i - 1) * b, b, b) = H.bottomRows(b);
    ProjectedMatrix.block(i * b, (i - 1) * b, b, b) = beta;
    ProjectedMatrix.block((i - 1) * b, i * b, b, b) = beta.transpose();
  }
}

void KrylovSubspace::build(VectorXd& KrylovResidual,
                           MatrixXd& OrthoBasis,
                           MatrixXd& ProjectedMatrix,
                           const operations& ops,
                           size_t begin,
                           size_t end) {
  KrylovSubspace subspace(KrylovResidual, OrthoBasis, ProjectedMatrix);
  subspace.build(ops, begin, end);
}

void KrylovSubspace::build(const MatrixXd& V0,
                           MatrixXd& OrthoBasis,
                           MatrixXd& ProjectedMatrix,
                           const operations& ops,
                           size_t begin,
                           size_t end,
                           MatrixXd& beta) {
  KrylovSubspace subspace(V0, OrthoBasis, ProjectedMatrix, beta);
  subspace.build_block(ops, begin, end);
}

void KrylovSubspace::getevbounds(int n,
                                 const operations& ops,
                                 double& evlb,
                                 double& evub) {
  constexpr int k = 20; // number of Lanczos iterations, hard coded for now

  MatrixXd Q(n, k);
  MatrixXd H(k, k);
  VectorXd v0 = Eigen::VectorXd::Random(n);

  H.setZero();

  KrylovSubspace subspace(v0, Q, H);
  subspace.build(ops, 0, k);

  const double beta = v0.norm();
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(H.block(0, 0, k, k));

  const VectorXd eigenvalues = solver.eigenvalues();
  const MatrixXd eigenvectors = solver.eigenvectors();

  evlb = std::numeric_limits<double>::infinity();
  evub = -std::numeric_limits<double>::infinity();

  for (int j = 0; j < k; j++) {
    const double t = std::fabs(beta * eigenvectors(k - 1, j));
    double x = eigenvalues(j) - t;
    if (x < evlb) evlb = x;

    x = eigenvalues(j) + t;
    if (x > evub) evub = x;
  }
}

