#include "KrylovSubspace.hpp"

#include <cmath>
#include <limits>
#include <utility>

// KrylovSubspace: unified Krylov basis construction.
// - Vector version (Lanczos): uses ops.matvec
// - Block version (Block Lanczos): uses ops.block_matvec

KrylovSubspace::KrylovSubspace(VectorXd KrylovResidual,
                               MatrixXd OrthoBasis,
                               MatrixXd ProjectedMatrix)
    : krylov_residual_(std::move(KrylovResidual)),
      ortho_basis_(std::move(OrthoBasis)),
      projected_matrix_(std::move(ProjectedMatrix)) {}

KrylovSubspace::KrylovSubspace(MatrixXd V0,
                               MatrixXd OrthoBasis,
                               MatrixXd ProjectedMatrix,
                               MatrixXd beta)
    : V0_(std::move(V0)),
      ortho_basis_(std::move(OrthoBasis)),
      projected_matrix_(std::move(ProjectedMatrix)),
      beta_(std::move(beta)) {}

void KrylovSubspace::build(const operations& ops,
                           size_t begin,
                           size_t end) {
  const size_t n = krylov_residual_.size();

  if (end == 0) throw std::runtime_error("Requested Krylov subspace of dim 0 in KrylovSubspace");
  if (end > n) throw std::runtime_error("Requested Krylov subspace dim > matrix size in KrylovSubspace");
  if (!ops.matvec) throw std::runtime_error("KrylovSubspace (vector): ops.matvec is not set");

  VectorXd v(n), Av(n);

  for (size_t i = begin; i < end; ++i) {
    beta_scalar_ = krylov_residual_.norm();
    if (!std::isfinite(beta_scalar_)) {
      throw std::runtime_error("KrylovSubspace: non-finite residual norm encountered.");
    }
    // Handle breakdown: if f is (near) zero, restart with a random vector
    if (beta_scalar_ <= std::numeric_limits<double>::epsilon()) {
      krylov_residual_ = VectorXd::Random(n);
      if (i > 0) {
        krylov_residual_ -= ortho_basis_.leftCols(i) *
                            (ortho_basis_.leftCols(i).transpose() * krylov_residual_);
      }
      beta_scalar_ = krylov_residual_.norm();
      if (beta_scalar_ <= std::numeric_limits<double>::epsilon()) {
        throw std::runtime_error("KrylovSubspace: breakdown, unable to generate new basis vector");
      }
    }
    v = krylov_residual_ / beta_scalar_;
    ortho_basis_.col(i) = v;

    // Fill the symmetric tridiagonal coupling coefficient (beta).
    // NOTE: This routine is used by multiple algorithms with different restart conventions.
    // In particular, Krylov-Schur uses a restart step where the coupling at the start of a
    // multi-step expansion (i == begin, end-begin > 1) is handled outside this helper.
    if (i != begin || (begin != 0 && end - begin == 1)) {
      projected_matrix_(i, i - 1) = beta_scalar_;
      projected_matrix_(i - 1, i) = beta_scalar_;
    }

    ops.matvec(v, Av);
    if (!Av.allFinite()) {
      throw std::runtime_error("KrylovSubspace: non-finite value encountered in matvec output.");
    }

    VectorXd h = ortho_basis_.leftCols(i + 1).transpose() * Av;
    krylov_residual_ = Av - ortho_basis_.leftCols(i + 1) * h;

    VectorXd s = ortho_basis_.leftCols(i + 1).transpose() * krylov_residual_;
    h += s;
    krylov_residual_ -= ortho_basis_.leftCols(i + 1) * s;

    projected_matrix_(i, i) = h(i);
  }
}

void KrylovSubspace::build_block(const operations& ops,
                                 size_t begin,
                                 size_t end) {
  const size_t n = V0_.rows();
  const size_t b = V0_.cols();

  if (end == 0) throw std::runtime_error("Requested Krylov subspace of dim 0 in KrylovSubspace");
  if (end > n) throw std::runtime_error("Requested Krylov subspace dim > matrix size in KrylovSubspace");
  if (!ops.block_matvec) throw std::runtime_error("KrylovSubspace (block): ops.block_matvec is not set");

  MatrixXd f(n, b), V(n, b), AV(n, b);

  if (begin == 0) {
    Eigen::HouseholderQR<MatrixXd> qrInit(V0_);
    MatrixXd Q_block0 = MatrixXd::Identity(n, b);
    Q_block0 = qrInit.householderQ() * Q_block0;
    ortho_basis_.block(0, 0, n, b) = Q_block0;
  }

  for (size_t i = begin + 1; i <= end; ++i) {
    V = ortho_basis_.block(0, (i - 1) * b, n, b);

    ops.block_matvec(V, AV);

    MatrixXd H = ortho_basis_.leftCols(i * b).transpose() * AV;
    f = AV - ortho_basis_.leftCols(i * b) * H;

    MatrixXd S = ortho_basis_.leftCols(i * b).transpose() * f;
    H += S;
    f -= ortho_basis_.leftCols(i * b) * S;

    Eigen::HouseholderQR<MatrixXd> qrR(f);
    MatrixXd Q_new = MatrixXd::Identity(n, b);
    MatrixXd R_top = qrR.matrixQR().topLeftCorner(b, b).template triangularView<Eigen::Upper>();
    Q_new = qrR.householderQ() * Q_new;
    ortho_basis_.block(0, i * b, n, b) = Q_new;
    beta_ = R_top;

    projected_matrix_.block((i - 1) * b, (i - 1) * b, b, b) = H.bottomRows(b);
    projected_matrix_.block(i * b, (i - 1) * b, b, b) = beta_;
    projected_matrix_.block((i - 1) * b, i * b, b, b) = beta_.transpose();
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
  KrylovResidual = subspace.KrylovResidual();
  OrthoBasis = subspace.OrthoBasis();
  ProjectedMatrix = subspace.ProjectedMatrix();
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
  OrthoBasis = subspace.OrthoBasis();
  ProjectedMatrix = subspace.ProjectedMatrix();
  beta = subspace.beta();
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
  v0 = subspace.KrylovResidual();
  H = subspace.ProjectedMatrix();

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
