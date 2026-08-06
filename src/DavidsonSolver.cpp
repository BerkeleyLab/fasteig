#include "DavidsonSolver.hpp"

#include "core/Executor.hpp"
#include "core/LinearOperator.hpp"
#include "core/Orthogonalization.hpp"
#include "core/Operations.hpp"
#include "fasteig_logger.hpp"
#include "fasteig_problem.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

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

  std::shared_ptr<core::Executor> exec = problem_.executor(options_);
  std::unique_ptr<core::LinearOperator> op = problem_.make_linear_operator(options_, *exec);
  core::ScaledLinearOperator effective_op(*op, flip_sign ? -1.0 : 1.0);

  const FASTEIG_INT nev = options_.nev();
  const FASTEIG_INT maxiter = options_.maxiter();
  const FASTEIG_INT maxblks = options_.maxblks();

  if (maxblks < 1) {
    throw std::invalid_argument("DavidsonSolver: maxblks must be at least 1.");
  }

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

  const int nrows = eigvecs_.rows();
  const int ncols = eigvecs_.cols();
  const int max_subspace_cols = static_cast<int>(maxblks) * ncols;
  int info = 0;

  MatrixXd Q(nrows, max_subspace_cols), AQ(nrows, max_subspace_cols);
  MatrixXd AV(nrows, ncols), Ritzvecs(nrows, ncols), RES(nrows, ncols);
  MatrixXd T(max_subspace_cols, max_subspace_cols), R(ncols, ncols),
      G(ncols, ncols);
  VectorXd Ritzvals(ncols), resnrm(ncols);
  fasteig_logger stdout_logger(&std::cout);

  // Orthonormalize the initial basis vectors before Rayleigh-Ritz.
  RES = eigvecs_;
  info = core::cholesky_qr(RES, eigvecs_, R);
  if (info != 0) {
    stdout_logger.print_progress_note(
        "Cholesky factorization failed, info = " + std::to_string(info));
    std::exit(1);
  }

  // RES keeps the residual of the initial Rayleigh-Ritz approximation.
  RayleighRitz(effective_op, eigvecs_, Ritzvals, Ritzvecs, G, RES);

  const FASTEIG_DOUBLE tol = options_.tol();
  // Match the MATLAB implementation: a Ritz vector is excluded from subsequent
  // correction blocks only after satisfying the (possibly stricter) lock tolerance.
  const FASTEIG_DOUBLE locktol = std::min<FASTEIG_DOUBLE>(1.0e-5, tol);

  auto update_relative_residuals = [&]() {
    for (int j = 0; j < ncols; ++j) {
      const double denom =
          (1.0 + std::abs(Ritzvals[j])) * Ritzvecs.col(j).norm();
      const double abs_residual = RES.col(j).norm();
      resnrm[j] = (denom > 0.0) ? (abs_residual / denom) : abs_residual;
    }
  };

  update_relative_residuals();
  size_t it_done = 0;
  for (size_t iter = 0; iter < static_cast<size_t>(maxiter); ++iter) {
    T.setZero();
    Q.leftCols(ncols) = Ritzvecs;
    for (int j = 0; j < ncols; ++j) {
      T(j, j) = Ritzvals[j];
      // Back out A * Ritzvecs from the residual.
      AQ.col(j) = RES.col(j) + Ritzvals[j] * Ritzvecs.col(j);
    }
    int nv = ncols;

    for (int jblk = 1; jblk < static_cast<int>(maxblks); ++jblk) {
      problem_.logger().print_convergence_data(Ritzvals.head(nev), resnrm.head(nev),
                                                static_cast<int>(iter + 1));
      if (options_.verbosity() >= 2) {
        stdout_logger.print_convergence_data(Ritzvals.head(nev), resnrm.head(nev),
                                             static_cast<int>(iter + 1));
      }

      // ilock and iact are recomputed from the current Ritz residuals, as in
      // davidson.m. Locked Ritz vectors remain in the Ritz block but do not
      // contribute correction vectors to the expanding subspace.
      std::vector<int> ilock;
      std::vector<int> iact;
      ilock.reserve(ncols);
      iact.reserve(ncols);
      for (int j = 0; j < ncols; ++j) {
        if (std::abs(resnrm[j]) <= locktol) {
          ilock.push_back(j);
        } else {
          iact.push_back(j);
        }
      }

      const int nlock = static_cast<int>(ilock.size());
      const int nact = static_cast<int>(iact.size());
      if (nlock == ncols) break;

      MatrixXd W(nrows, nact);
      for (int j = 0; j < nact; ++j) {
        W.col(j) = RES.col(iact[j]);
      }

      // Orthogonalize W(:,iact) against all current basis vectors.
      MatrixXd Temp = Q.leftCols(nv).transpose() * W;
      W.noalias() -= Q.leftCols(nv) * Temp;
      Temp = Q.leftCols(nv).transpose() * W;
      W.noalias() -= Q.leftCols(nv) * Temp;

      // [V(:,b2), R] = qr(W(:,iact), 0), followed by H * V(:,b2).
      MatrixXd new_basis;
      info = core::householder_qr(W, new_basis, R);
      if (info != 0) {
        stdout_logger.print_progress_note(
            "QR factorization failed, info = " + std::to_string(info));
        std::exit(1);
      }

      Q.middleCols(nv, nact) = new_basis;

      AV = core::apply_to_host(effective_op, new_basis);
      AQ.middleCols(nv, nact) = AV;

      // Append the variable-width projection blocks into T.
      T.block(0, nv, nv, nact) = Q.leftCols(nv).transpose() * AV;
      T.block(nv, 0, nact, nv) = T.block(0, nv, nv, nact).transpose();
      T.block(nv, nv, nact, nact) = new_basis.transpose() * AV;
      nv += nact;

      MatrixXd T1 =
          (T.topLeftCorner(nv, nv) + T.topLeftCorner(nv, nv).transpose()) / 2.0;

      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(T1);
      if (solver.info() != Eigen::Success) {
        throw std::runtime_error("DavidsonSolver: projected eigensolve failed.");
      }
      const MatrixXd eigenvectors = solver.eigenvectors();
      const VectorXd eigenvalues = solver.eigenvalues();
      Ritzvecs = Q.leftCols(nv) * eigenvectors.leftCols(ncols);
      Ritzvals = eigenvalues.head(ncols);
      RES = AQ.leftCols(nv) * eigenvectors.leftCols(ncols);

      for (int j = 0; j < ncols; ++j) {
        RES.col(j).noalias() -= Ritzvecs.col(j) * Ritzvals[j];
      }
      update_relative_residuals();
    }

    it_done = iter + 1;
    // Convergence and locking deliberately use different tolerances, matching
    // the nconv/nlock distinction in davidson.m.
    bool converged = true;
    for (int j = 0; j < static_cast<int>(nev); ++j) {
      if (std::abs(resnrm[j]) >= tol) {
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

