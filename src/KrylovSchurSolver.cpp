#include "KrylovSchurSolver.hpp"

#include "core/Executor.hpp"
#include "core/KrylovSubspace.hpp"
#include "core/LanczosWorkspace.hpp"
#include "core/LinearOperator.hpp"
#include "core/MultiVector.hpp"
#include "core/RitzExtraction.hpp"
#include "core/Vector.hpp"
#include "fasteig_logger.hpp"
#include "fasteig_problem.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fasteig {
KrylovSchurSolver::KrylovSchurSolver(::fasteig_problem& problem,
                                     const fasteig_options& options)
    : fasteig_solver(problem, options) {}

FASTEIG_INT KrylovSchurSolver::solve() {
  const FASTEIG_INT n = eigvecs_.rows();
  const FASTEIG_INT nev = options_.nev();
  const FASTEIG_INT maxiter = options_.maxiter();
  const FASTEIG_DOUBLE shift = options_.shift();
  const FASTEIG_DOUBLE tol = options_.tol();
  const int verbosity = options_.verbosity();
  const FASTEIG_INT print_every = options_.print_every();
  const std::string which = core::lanczos_lower_copy(options_.which());

  if (n <= 0) throw std::invalid_argument("KrylovSchurSolver: n must be > 0.");
  if (nev <= 0 || nev > n) {
    throw std::invalid_argument("KrylovSchurSolver: nev must be in [1,n].");
  }
  if (maxiter <= 0) {
    throw std::invalid_argument("KrylovSchurSolver: maxiter must be > 0.");
  }
  core::validate_lanczos_which(which, /*allow_sai=*/true, "KrylovSchurSolver");

  constexpr FASTEIG_INT kMinimumSubspace = 20;
  const FASTEIG_INT subspace_dim =
      std::min(std::max({2 * nev, kMinimumSubspace, options_.maxdim()}), n);

  fasteig_logger stdout_logger;
  if (verbosity >= 1) stdout_logger.attach(&std::cout);
  fasteig_logger* log_logger =
      (print_every > 0 && problem_.log_stream()) ? &problem_.logger() : nullptr;
  fasteig_logger* console_logger = (verbosity >= 2) ? &stdout_logger : nullptr;

  // All large vectors and the projected matrix are owned by the selected
  // executor. CPU and CUDA use the same Krylov-Schur algorithm below.
  std::shared_ptr<core::Executor> exec = problem_.executor(options_);
  std::unique_ptr<core::LinearOperator> op = problem_.make_linear_operator(options_, *exec);
  auto original_A = problem_.make_matrix_operator(*exec);
  auto metric = problem_.make_metric_operator(*exec);
  // Candidate Ritz vectors and true A*x-lambda*B*x residuals stay on the executor.
  auto ritz_vectors = exec->create_multivector(n, nev);
  auto ritz_residuals = exec->create_multivector(n, nev);
  auto metric_vectors = exec->create_multivector(n, nev);
  const unsigned int base_seed = options_.seed().value_or(1u);
  auto residual = (options_.v0().size() > 0)
                      ? exec->create_vector(options_.v0().col(0))
                      : exec->create_random_vector(n, base_seed);
  auto operator_result = exec->create_vector(n);
  auto basis = exec->create_multivector(n, subspace_dim);
  auto projection =
      core::make_lanczos_workspace(*exec, n, subspace_dim, subspace_dim);
  core::KrylovSubspace subspace(
      *op, *residual, *operator_result, *basis, base_seed, metric.get());

  auto expand_subspace = [&](FASTEIG_INT begin) {
    for (FASTEIG_INT col = begin; col < subspace_dim; ++col) {
      const double beta = subspace.normalize_residual_into_basis(col);
      subspace.apply_operator_to_basis_column(col);
      const double alpha = subspace.orthogonalize_operator_result(col + 1);
      const double beta_next = subspace.residual_norm();
      projection->record_step(col, beta, alpha, beta_next);
    }
  };

  if (metric) {
    if (log_logger) log_logger->print_progress_note("Projected residual estimates use the B inner product; final residuals use A*x-lambda*B*x.");
    if (console_logger) console_logger->print_progress_note("Projected residual estimates use the B inner product; final residuals use A*x-lambda*B*x.");
  }
  expand_subspace(0);

  for (FASTEIG_INT iter = 1; iter <= maxiter; ++iter) {
    // Diagonalize the projected matrix and order all Ritz pairs according to
    // `which`. CUDA uses the existing cuSOLVER workspace implementation.
    core::LanczosSnapshot full = projection->compute_snapshot(
        subspace_dim, subspace_dim, which, /*need_vectors=*/true,
        "KrylovSchurSolver");

    FASTEIG_INT nconv = 0;
    for (FASTEIG_INT j = 0; j < nev; ++j) {
      const int idx = static_cast<int>(j);
      const double magnitude = std::abs(full.lam(idx));
      // Preserve the established standard-problem stopping rule. For the
      // generalized path, use the zero-safe relative estimate as a trigger
      // for the original-problem verification below.
      const bool estimated_converged = metric
          ? full.rel_res(idx) <= tol
          : full.rel_res(idx) * (1.0 + magnitude) < tol * magnitude;
      if (estimated_converged) ++nconv;
    }

    const bool do_progress =
        (print_every > 0) && ((iter % print_every) == 0);
    const core::LanczosSnapshot report = core::krylov_schur_progress_snapshot(
        full, nev, which, shift, "KrylovSchurSolver");
    if (do_progress) {
      core::lanczos_emit_snapshot(report, iter, log_logger, console_logger);
    }

    // Projected estimates are inexpensive, but transformed/weighted residuals
    // must not be mistaken for the requested original-problem tolerance.
    const bool at_limit = (subspace_dim == n) || (iter == maxiter);
    if (nconv >= nev || at_limit) {
      const auto final = core::lanczos_snapshot_head(full, nev, "KrylovSchurSolver");
      resnrms_ = core::evaluate_ritz_pairs(
          *original_A, metric.get(), *basis, subspace_dim, final, which, shift,
          *ritz_vectors, *ritz_residuals, *metric_vectors, eigvals_);
      const bool verified = (resnrms_.array() <= tol).all();
      if (!metric || verified || at_limit) {
        if (!do_progress && print_every > 0) {
          core::lanczos_emit_snapshot(report, iter, log_logger, console_logger);
        }
        eigvecs_ = ritz_vectors->copy_to_host();
        if (metric && !verified) {
          status_ = 1;
          problem_.logger().print_progress_note("Stopped without meeting the original-problem residual tolerance.");
          stdout_logger.print_progress_note("Stopped without meeting the original-problem residual tolerance.");
        }
        set_iterations(static_cast<long long>(iter));
        return status_;
      }
    }

    // Retain the wanted Ritz vectors, seed the dense restart coupling row,
    // then expand the compacted basis back to the target subspace dimension.
    const FASTEIG_INT spare = (subspace_dim - nev) / 2;
    FASTEIG_INT retained = nev + std::min(nconv, spare);
    if (retained == 1 && subspace_dim > 3) retained = subspace_dim / 2;

    subspace.rotate_basis(
        subspace_dim, full.Csel.leftCols(static_cast<int>(retained)));
    projection->prepare_krylov_schur_restart(
        retained, full, "KrylovSchurSolver");
    expand_subspace(retained);
  }

  throw std::logic_error("KrylovSchurSolver: unreachable solve termination.");
}

}  // namespace fasteig

