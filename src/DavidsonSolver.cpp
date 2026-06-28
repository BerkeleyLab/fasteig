#include "DavidsonSolver.hpp"

#include "fasteig_logger.hpp"
#include "fasteig_problem.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <iostream>

namespace fasteig {

DavidsonSolver::DavidsonSolver(::fasteig_problem& problem,
                               const fasteig_options& options)
    : fasteig_solver(problem, options) {}

FASTEIG_INT DavidsonSolver::solve() {
  auto lower_copy = [](std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
  };
  const std::string which_l = lower_copy(options_.which());
  const bool flip_sign = (which_l == "lr" || which_l == "lm");

  operations ops_eff = active_ops();
  if (flip_sign) {
    const auto orig_mv = ops_eff.matvec;
    const auto orig_bmv = ops_eff.block_matvec;
    if (orig_mv) {
      ops_eff.matvec = [orig_mv](const VectorXd& v, VectorXd& av) {
        orig_mv(v, av);
        av *= -1.0;
      };
    }
    if (orig_bmv) {
      ops_eff.block_matvec = [orig_bmv](const MatrixXd& V, MatrixXd& AV) {
        orig_bmv(V, AV);
        AV *= -1.0;
      };
    }
  }

  const FASTEIG_INT nev = options_.nev();
  const FASTEIG_INT maxiter = options_.maxiter();
  const FASTEIG_INT maxblks = options_.maxblks();

  if (options_.v0().size() > 0) {
    if (options_.v0().rows() != eigvecs_.rows()) {
      throw std::invalid_argument("DavidsonSolver: v0 row count does not match problem dimension.");
    }
    eigvecs_.setRandom();
    const int c = std::min<int>(options_.v0().cols(), static_cast<int>(nev));
    eigvecs_.leftCols(c) = options_.v0().leftCols(c);
  } else {
    if (options_.seed().has_value()) std::srand(*options_.seed());
    eigvecs_.setRandom();
  }

  const size_t nrows = eigvecs_.rows();
  const size_t ncols = eigvecs_.cols();
  int info = 0;
  int row0, col0;

  MatrixXd Q(nrows, maxblks * ncols), AQ(nrows, maxblks * ncols);
  MatrixXd AV(nrows, ncols), Ritzvecs(nrows, ncols), RES(nrows, ncols);
  MatrixXd T(ncols * maxblks, ncols * maxblks), R(ncols, ncols),
      G(ncols, ncols);
  VectorXd Ritzvals(ncols), resnrm(ncols);
  fasteig_logger stdout_logger(&std::cout);

  // Orthonormalize the initial basis vectors before Rayleigh-Ritz.
  RES = eigvecs_;
  info = CholeskyQR(RES, eigvecs_, R);
  if (info != 0) {
    stdout_logger.print_progress_note(
        "Cholesky factorization failed, info = " + std::to_string(info));
    std::exit(1);
  }

  // RES keeps the residual of the initial Rayleigh-Ritz approximation.
  RayleighRitz(ops_eff, eigvecs_, Ritzvals, Ritzvecs, G, RES);

  const FASTEIG_DOUBLE tol = options_.tol();
  size_t it_done = 0;
  for (size_t iter = 0; iter < static_cast<size_t>(maxiter); ++iter) {
    T.setZero();
    Q.leftCols(ncols) = Ritzvecs;
    for (size_t j = 0; j < ncols; j++) {
      resnrm[j] = RES.col(j).norm();
      T(j, j) = Ritzvals[j];
      // Back out A * Ritzvecs from the residual.
      AQ.col(j) = RES.col(j) + Ritzvals[j] * Ritzvecs.col(j);
    }

    for (size_t jblk = 1; jblk < maxblks; jblk++) {
      row0 = 0;
      col0 = jblk * ncols;

      problem_.logger().print_convergence_data(Ritzvals.head(nev), resnrm.head(nev),
                                                static_cast<int>(iter + 1));
      if (options_.verbosity() >= 2) {
        stdout_logger.print_convergence_data(Ritzvals.head(nev), resnrm.head(nev),
                                             static_cast<int>(iter + 1));
      }

      // Orthogonalize against previous basis vectors.
      MatrixXd Temp = Q.leftCols(jblk * ncols).transpose() * RES;
      RES = RES - Q.leftCols(jblk * ncols) * Temp;
      Temp = Q.leftCols(jblk * ncols).transpose() * RES;
      RES = RES - Q.leftCols(jblk * ncols) * Temp;

      // Orthonormalize within the block.
      info = CholeskyQR(RES, eigvecs_, R);
      if (info != 0) {
        stdout_logger.print_progress_note(
            "Cholesky factorization failed, info = " + std::to_string(info));
        std::exit(1);
      }

      Q.block(row0, col0, nrows, ncols) = eigvecs_;

      ops_eff.block_matvec(eigvecs_, AV);
      AQ.block(row0, col0, nrows, ncols) = AV;

      // Append the new projection blocks into T.
      T.block(row0, col0, jblk * ncols, ncols) =
          Q.leftCols(jblk * ncols).transpose() * AV;
      T.block(col0, row0, ncols, jblk * ncols) =
          T.block(row0, col0, jblk * ncols, ncols).transpose();
      row0 = jblk * ncols;
      col0 = row0;
      T.block(row0, col0, ncols, ncols) = eigvecs_.transpose() * AV;

      const int dim = ncols * (jblk + 1);
      MatrixXd T1 =
          (T.block(0, 0, dim, dim) + T.block(0, 0, dim, dim).transpose()) /
          2.0;

      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(
          T1.block(0, 0, dim, dim));
      MatrixXd eigenvectors = solver.eigenvectors();
      VectorXd eigenvalues = solver.eigenvalues();
      Ritzvecs = Q.leftCols((jblk + 1) * ncols) * eigenvectors.leftCols(ncols);
      Ritzvals = eigenvalues.head(ncols);
      RES = AQ.leftCols((jblk + 1) * ncols) * eigenvectors.leftCols(ncols);

      for (size_t j = 0; j < ncols; j++) {
        RES.col(j) = RES.col(j) - Ritzvecs.col(j) * Ritzvals[j];
        resnrm[j] = RES.col(j).norm();
      }
    }

    it_done = iter + 1;
    // Convergence check (relative residual norm, consistent with log reporting).
    bool converged = true;
    for (size_t j = 0; j < static_cast<size_t>(nev); ++j) {
      const double denom = (1.0 + std::abs(Ritzvals[static_cast<int>(j)])) * Ritzvecs.col(static_cast<int>(j)).norm();
      const double rel = (denom > 0.0) ? (resnrm[static_cast<int>(j)] / denom) : resnrm[static_cast<int>(j)];
      if (rel >= tol) {
        converged = false;
        break;
      }
    }
    if (converged) break;
  }

  eigvals_ = Ritzvals.head(nev);
  eigvecs_ = Ritzvecs.leftCols(nev);
  if (flip_sign) {
    eigvals_ *= -1.0;
  }
  compute_final_residual_norms();
  set_iterations(static_cast<long long>(it_done));
  return status_;
}

}  // namespace fasteig

