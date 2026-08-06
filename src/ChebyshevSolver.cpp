#include "ChebyshevSolver.hpp"

#include "core/Executor.hpp"
#include "core/KrylovSubspace.hpp"
#include "core/LinearOperator.hpp"
#include "core/Orthogonalization.hpp"
#include "core/Operations.hpp"
#include "core/MultiVector.hpp"
#include "core/Vector.hpp"
#include "fasteig_logger.hpp"
#include "fasteig_problem.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace fasteig {
namespace {

MatrixXd apply_effective_to_host(const core::LinearOperator& op,
                                 const Eigen::Ref<const MatrixXd>& X) {
  return core::apply_to_host(op, X);
}

void estimate_bounds_with_executor(const core::LinearOperator& op,
                                   FASTEIG_INT n,
                                   unsigned int seed,
                                   double& evlb,
                                   double& evub) {
  constexpr FASTEIG_INT k_requested = 20;
  const FASTEIG_INT k = std::min(k_requested, n);
  if (k <= 0) throw std::invalid_argument("ChebyshevSolver: invalid dimension for bound estimate.");

  auto residual = op.executor().create_random_vector(n, seed);
  auto operator_result = op.executor().create_vector(n);
  auto basis = op.executor().create_multivector(n, k);
  MatrixXd H(static_cast<int>(k), static_cast<int>(k));
  H.setZero();

  core::KrylovSubspace subspace(op, *residual, *operator_result, *basis, H, seed);
  subspace.build(0, k);

  const double beta = core::norm2(*residual);
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(H.block(0, 0, static_cast<int>(k), static_cast<int>(k)));
  const VectorXd eigenvalues = solver.eigenvalues();
  const MatrixXd eigenvectors = solver.eigenvectors();

  evlb = std::numeric_limits<double>::infinity();
  evub = -std::numeric_limits<double>::infinity();
  for (FASTEIG_INT j = 0; j < k; ++j) {
    const double t = std::fabs(beta * eigenvectors(static_cast<int>(k - 1), static_cast<int>(j)));
    evlb = std::min(evlb, eigenvalues(static_cast<int>(j)) - t);
    evub = std::max(evub, eigenvalues(static_cast<int>(j)) + t);
  }
}

}  // namespace

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

  std::shared_ptr<core::Executor> exec = problem_.executor(options_);
  std::unique_ptr<core::LinearOperator> op = problem_.make_linear_operator(options_, *exec);
  core::ScaledLinearOperator effective_op(*op, flip_sign ? -1.0 : 1.0);

  const FASTEIG_INT n = eigvecs_.rows();
  const FASTEIG_INT nev = options_.nev();
  const FASTEIG_INT maxiter = options_.maxiter();
  const FASTEIG_INT degree = options_.degree();
  const FASTEIG_DOUBLE tol = options_.tol();
  FASTEIG_DOUBLE lambda_max;
  FASTEIG_DOUBLE lambda_min;

  MatrixXd V0;
  if (options_.v0().size() > 0) {
    V0 = options_.v0().leftCols(static_cast<int>(nev));
  } else {
    V0 = MatrixXd::Random(n, nev);
  }

  // Keep one spare vector per requested eigenpair. Each
  // locked Ritz vector is removed from the filtered block and one spare is
  // activated, so the number of actively filtered vectors stays constant.
  // For very small problems the ambient dimension limits the number of spares.
  const FASTEIG_INT workspace_cols = std::min<FASTEIG_INT>(n, 2 * nev);
  if (workspace_cols < nev) {
    throw std::invalid_argument(
        "ChebyshevSolver: nev cannot exceed the problem dimension.");
  }

  MatrixXd X(n, workspace_cols);
  X.leftCols(static_cast<int>(nev)) = V0;
  if (workspace_cols > nev) {
    std::mt19937 generator(options_.seed().value_or(1u));
    std::normal_distribution<double> distribution(0.0, 1.0);
    for (int j = static_cast<int>(nev); j < X.cols(); ++j) {
      for (int i = 0; i < X.rows(); ++i) {
        X(i, j) = distribution(generator);
      }
    }
  }

  MatrixXd qr_basis;
  MatrixXd qr_factor;
  FASTEIG_INT info = core::cholesky_qr(X, qr_basis, qr_factor);
  if (info != 0) {
    status_ = info;
    set_iterations(0);
    return status_;
  }
  X = std::move(qr_basis);

  std::vector<int> active;
  active.reserve(static_cast<size_t>(nev));
  for (int j = 0; j < static_cast<int>(nev); ++j) active.push_back(j);

  FASTEIG_INT nlocked = 0;
  FASTEIG_INT subspace_cols = nev;
  VectorXd ritz_values;
  MatrixXd ritz_vectors;
  MatrixXd projected_vectors;
  MatrixXd residuals;
  VectorXd relative_residuals(static_cast<int>(nev));
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
  estimate_bounds_with_executor(effective_op, n, options_.seed().value_or(1u), lambda_min, lambda_max);
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

    MatrixXd filtered(n, static_cast<int>(active.size()));
    for (int j = 0; j < filtered.cols(); ++j) {
      filtered.col(j) = X.col(active[static_cast<size_t>(j)]);
    }
    chebyshev_filter(effective_op, filtered, degree, lambda_min, lambda_max,
                     flip_sign);

    // Deflate the active block against the ordered, locked Ritz vectors.
    if (nlocked > 0) {
      const auto locked = X.leftCols(static_cast<int>(nlocked));
      filtered.noalias() -= locked * (locked.transpose() * filtered);
    }
    for (int j = 0; j < filtered.cols(); ++j) {
      X.col(active[static_cast<size_t>(j)]) = filtered.col(j);
    }

    MatrixXd basis;
    MatrixXd R;
    info = core::cholesky_qr(
        X.leftCols(static_cast<int>(subspace_cols)), basis, R);
    if (info != 0) {
      const std::string failure_message =
          "QR decomposition failed during Chebyshev iteration.";
      problem_.logger().print_progress_note(failure_message);
      stderr_logger.print_progress_note(failure_message);
      break;
    }

    RayleighRitz(effective_op, basis, ritz_values, ritz_vectors,
                 projected_vectors, residuals);
    X.leftCols(static_cast<int>(subspace_cols)) = ritz_vectors;

    for (int j = 0; j < static_cast<int>(nev); ++j) {
      const double denominator = std::abs(ritz_values[j]);
      const double absolute_residual = residuals.col(j).norm();
      relative_residuals[j] =
          (denominator > 0.0) ? absolute_residual / denominator
                              : absolute_residual;
    }

    VectorXd displayed_values = ritz_values.head(static_cast<int>(nev));
    if (flip_sign) displayed_values *= -1.0;
    problem_.logger().print_convergence_data(
        displayed_values, relative_residuals, static_cast<int>(iter + 1));
    if (options_.verbosity() >= 2) {
      stdout_logger.print_convergence_data(
          displayed_values, relative_residuals, static_cast<int>(iter + 1));
    }

    // Locking in order: All preceding Ritz vectors must have already been 
    // locked to prevent a higher Ritz pair from being deflated before 
    // a lower one has converged. This conservative strategy tends to 
    // reduce the likelihood of missing some eigenvalues
    while (nlocked < nev && relative_residuals[nlocked] <= tol) {
      const int locked_index = static_cast<int>(nlocked);
      active.erase(std::remove(active.begin(), active.end(), locked_index),
                   active.end());
      ++nlocked;
      problem_.logger().print_locking_note(static_cast<int>(nlocked),
                                           static_cast<int>(iter + 1));
      if (options_.verbosity() >= 2) {
        stdout_logger.print_locking_note(static_cast<int>(nlocked),
                                         static_cast<int>(iter + 1));
      }

      if (subspace_cols < workspace_cols) {
        active.push_back(static_cast<int>(subspace_cols));
        ++subspace_cols;
      }
    }

    const FASTEIG_DOUBLE subspace_resnrm = residuals.norm();
    std::ostringstream residual_message;
    residual_message << std::scientific << std::setprecision(2)
                     << "subspace residual norm = " << subspace_resnrm;
    problem_.logger().print_progress_note(residual_message.str());
    if (verbose) {
      stdout_logger.print_progress_note(residual_message.str());
    }
    it_done = iter + 1;
    if (nlocked == nev) break;
  }

  if (info == 0 && ritz_values.size() == 0) {
    MatrixXd basis;
    MatrixXd R;
    info = core::cholesky_qr(
        X.leftCols(static_cast<int>(subspace_cols)), basis, R);
    if (info == 0) {
      RayleighRitz(effective_op, basis, ritz_values, ritz_vectors,
                   projected_vectors, residuals);
    }
  }

  if (info == 0) {
    eigvals_ = ritz_values.head(static_cast<int>(nev));
    eigvecs_ = ritz_vectors.leftCols(static_cast<int>(nev));
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

void ChebyshevSolver::chebyshev_filter(const core::LinearOperator& op,
                                       MatrixXd& V,
                                       FASTEIG_INT degree,
                                       FASTEIG_DOUBLE lambda_min,
                                       FASTEIG_DOUBLE lambda_max,
                                       bool flip_sign) {
  MatrixXd V_old = V;
  MatrixXd V_new = V;
  MatrixXd AV = V;

  V_new = apply_effective_to_host(op, V);

  const FASTEIG_DOUBLE rat = 1.0e3; // ratio of Cheby(xlb) and Cheby(lb), hard-coded for now.
  const FASTEIG_DOUBLE xlb = std::cosh(std::acosh(rat) / degree);

  const FASTEIG_DOUBLE ub = lambda_max;
  const FASTEIG_DOUBLE lb = (2.0 * lambda_min + lambda_max * xlb - lambda_max) / (1.0 + xlb);

  const FASTEIG_DOUBLE c = 2.0 / (ub - lb);
  const FASTEIG_DOUBLE d = -(ub + lb) / (ub - lb);

  V_new = c * V_new + d * V;
  for (FASTEIG_INT k = 2; k <= degree; ++k) {
    AV = apply_effective_to_host(op, V_new);
    MatrixXd V_tmp = 2.0 * (c * AV + d * V_new) - V_old;
    V_old = V_new;
    V_new = V_tmp;
  }
  V = V_new;
  (void)flip_sign;
}

}  // namespace fasteig

