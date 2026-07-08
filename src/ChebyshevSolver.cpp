#include "ChebyshevSolver.hpp"

#include "KrylovSubspace.hpp"
#include "core/Orthogonalization.hpp"
#include "fasteig_logger.hpp"
#include "fasteig_problem.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace fasteig {

ChebyshevSolver::ChebyshevSolver(::fasteig_problem& problem,
                                 const fasteig_options& options)
    : fasteig_solver(problem, options) {}

FASTEIG_INT ChebyshevSolver::solve(::fasteig_problem& problem,
                                   const fasteig_options& options) {
  options.validate();
  fasteig_options local = options;
  const FASTEIG_INT nev = local.nev();
  local.ensure_initial_vectors(problem.n(), nev);

  problem.eigenvalues().resize(static_cast<int>(nev));
  problem.residual_norms().resize(static_cast<int>(nev));
  problem.eigenvectors() = local.v0();

  ChebyshevSolver solver(problem, local);
  const FASTEIG_INT status = solver.solve();
  problem.residual_norms() =
      fasteig_solver::compute_residual_norms(problem.opsA(),
                                             problem.is_generalized() ? &problem.opsB() : nullptr,
                                             problem.eigenvalues(),
                                             problem.eigenvectors());
  return status;
}

FASTEIG_INT ChebyshevSolver::solve() {
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

  const FASTEIG_INT n = eigvecs_.rows();
  const FASTEIG_INT nev = options_.nev();
  const FASTEIG_INT maxiter = options_.maxiter();
  const FASTEIG_INT degree = options_.degree();
  FASTEIG_DOUBLE lambda_max;
  FASTEIG_DOUBLE lambda_min;
  FASTEIG_DOUBLE subspace_resnrm;

  MatrixXd V0;
  if (options_.v0().size() > 0) {
    V0 = options_.v0().leftCols(static_cast<int>(nev));
  } else {
    V0 = MatrixXd::Random(n, nev);
  }

  MatrixXd X = V0;
  MatrixXd basis_vecs(static_cast<int>(nev), static_cast<int>(nev));
  MatrixXd residuals(n, nev);
  FASTEIG_INT info = 0;
  FASTEIG_INT it_done = 0;
  const bool verbose = options_.verbosity() != 0;
  fasteig_logger stdout_logger;
  if (verbose) {
    stdout_logger.attach(&std::cout);
  }
  fasteig_logger stderr_logger(&std::cerr);

  const std::string filtering_message =
      "Chebyshev filtering for the low end of the spectrum...";
  problem_.logger().print_progress_note(filtering_message);
  if (verbose) {
    stdout_logger.print_progress_note(filtering_message);
  }
  KrylovSubspace::getevbounds(n, ops_eff, lambda_min, lambda_max);
  problem_.logger().print_progress_note("Spectral bounds from Lanczos:");
  if (verbose) {
    stdout_logger.print_progress_note("Spectral bounds from Lanczos:");
  }
  std::ostringstream bounds_message;
  bounds_message << std::scientific << std::setprecision(6)
                 << "lb = " << lambda_min << " ub = " << lambda_max;
  problem_.logger().print_progress_note(bounds_message.str());
  if (verbose) {
    stdout_logger.print_progress_note(bounds_message.str());
  }

  for (FASTEIG_INT iter = 0; iter < maxiter; ++iter) {
    const std::string iteration_message =
        "Cheby filter iter = " + std::to_string(iter);
    problem_.logger().print_progress_note(iteration_message);
    if (verbose) {
      stdout_logger.print_progress_note(iteration_message);
    }

    chebyshev_filter(ops_eff, X, degree, lambda_min, lambda_max);

    MatrixXd R(static_cast<int>(nev), static_cast<int>(nev));
    info = core::cholesky_qr(X, eigvecs_, R);
    if (info != 0) {
      const std::string failure_message =
          "Decomposition failed. Matrix may not be positive definite.";
      problem_.logger().print_progress_note(failure_message);
      stderr_logger.print_progress_note(failure_message);
      break;
    }

    X = eigvecs_;
    ops_eff.block_matvec(X, eigvecs_);
    basis_vecs = X.transpose() * eigvecs_;
    eigvecs_ = eigvecs_ - X * basis_vecs;
    subspace_resnrm = eigvecs_.norm();
    std::ostringstream residual_message;
    residual_message << std::scientific << std::setprecision(2)
                     << "subspace residual norm = " << subspace_resnrm;
    problem_.logger().print_progress_note(residual_message.str());
    if (verbose) {
      stdout_logger.print_progress_note(residual_message.str());
    }
    it_done = iter + 1;
  }

  if (info == 0) {
    RayleighRitz(ops_eff, X, eigvals_, eigvecs_, basis_vecs, residuals);
  }

  if (flip_sign) {
    eigvals_ *= -1.0;
  }
  status_ = info;
  if (status_ == 0) {
    compute_final_residual_norms();
  }
  set_iterations(static_cast<long long>(it_done));
  return status_;
}

void ChebyshevSolver::chebyshev_filter(const operations& ops,
                                       MatrixXd& V,
                                       FASTEIG_INT degree,
                                       FASTEIG_DOUBLE lambda_min,
                                       FASTEIG_DOUBLE lambda_max) {
  MatrixXd V_old = V;
  MatrixXd V_new = V;
  MatrixXd AV = V;

  ops.block_matvec(V, V_new);

  const FASTEIG_DOUBLE rat = 1.0e3; // ratio of Cheby(xlb) and Cheby(lb), hard-coded for now.
  const FASTEIG_DOUBLE xlb = std::cosh(std::acosh(rat) / degree);

  const FASTEIG_DOUBLE ub = lambda_max;
  const FASTEIG_DOUBLE lb = (2.0 * lambda_min + lambda_max * xlb - lambda_max) / (1.0 + xlb);

  const FASTEIG_DOUBLE c = 2.0 / (ub - lb);
  const FASTEIG_DOUBLE d = -(ub + lb) / (ub - lb);

  V_new = c * V_new + d * V;
  for (FASTEIG_INT k = 2; k <= degree; ++k) {
    ops.block_matvec(V_new, AV);
    MatrixXd V_tmp = 2.0 * (c * AV + d * V_new) - V_old;
    V_old = V_new;
    V_new = V_tmp;
  }
  V = V_new;
}

}  // namespace fasteig

