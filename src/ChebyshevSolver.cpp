#include "ChebyshevSolver.hpp"

#include "core/Executor.hpp"
#include "core/KrylovSubspace.hpp"
#include "core/LanczosWorkspace.hpp"
#include "core/LinearOperator.hpp"
#include "core/MultiVector.hpp"
#include "core/Operations.hpp"
#include "core/Orthogonalization.hpp"
#include "core/RitzExtraction.hpp"
#include "core/Vector.hpp"
#include "fasteig_logger.hpp"
#include "fasteig_problem.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace fasteig {
namespace {

void estimate_bounds_with_executor(const core::LinearOperator& op,
                                   unsigned int seed,
                                   double& lower,
                                   double& upper) {
  const FASTEIG_INT n = op.rows();
  const FASTEIG_INT steps = std::min<FASTEIG_INT>(20, n);
  auto residual = op.executor().create_random_vector(n, seed);
  auto operator_result = op.executor().create_vector(n);
  auto basis = op.executor().create_multivector(n, steps);
  auto projection = core::make_lanczos_workspace(op.executor(), n, steps, steps);

  // Reuse the Lanczos recurrence and projected eigensolve on either executor.
  core::KrylovSubspace subspace(op, *residual, *operator_result, *basis,
                                *projection, seed);
  subspace.build(0, steps);
  const auto snapshot = projection->compute_snapshot(
      steps, steps, "sr", /*need_vectors=*/false, "ChebyshevSolver");
  const VectorXd residual_bounds = core::lanczos_absolute_residuals(snapshot);
  lower = (snapshot.lam - residual_bounds).minCoeff();
  upper = (snapshot.lam + residual_bounds).maxCoeff();
  if (!std::isfinite(lower) || !std::isfinite(upper) || lower > upper) {
    throw std::runtime_error("ChebyshevSolver: invalid spectral bound estimates.");
  }
}

}  // namespace

ChebyshevSolver::ChebyshevSolver(::fasteig_problem& problem,
                                 const fasteig_options& options)
    : fasteig_solver(problem, options) {}

FASTEIG_INT ChebyshevSolver::solve(::fasteig_problem& problem,
                                   const fasteig_options& options) {
  options.validate();
  problem.eigenvalues().resize(options.nev());
  problem.residual_norms().resize(options.nev());
  problem.eigenvectors().resize(problem.n(), options.nev());
  ChebyshevSolver solver(problem, options);
  return solver.solve();
}

FASTEIG_INT ChebyshevSolver::solve() {
  options_.validate();
  const FASTEIG_INT n = eigvecs_.rows();
  const FASTEIG_INT nev = options_.nev();
  const FASTEIG_INT maxiter = options_.maxiter();
  const FASTEIG_INT degree = options_.degree();
  const FASTEIG_INT print_every = options_.print_every();
  const FASTEIG_DOUBLE tol = options_.tol();
  const std::string which = core::lanczos_lower_copy(options_.which());
  if (n <= 0 || nev <= 0 || nev > n) {
    throw std::invalid_argument("ChebyshevSolver: nev must be in [1,n].");
  }
  if (which != "sr" && which != "lr") {
    throw std::invalid_argument("ChebyshevSolver: only SR and LR are supported.");
  }
  if (problem_.is_generalized()) {
    throw std::invalid_argument("ChebyshevSolver: generalized problems are not supported.");
  }

  // Executor-owned blocks: the same algorithm runs on CPU or CUDA. Only the
  // small projected matrices, convergence data, and final public results are
  // host-resident. LR uses the low-end filter on -A.
  auto exec = problem_.executor(options_);
  auto op = problem_.make_linear_operator(options_, *exec);
  const bool flip_sign = which == "lr";
  core::ScaledLinearOperator effective_op(*op, flip_sign ? -1.0 : 1.0);
  const unsigned int seed = options_.seed().value_or(1u);
  const FASTEIG_INT capacity = nev + std::min(nev, n - nev);
  auto vectors = exec->create_random_multivector(n, capacity, seed);
  if (options_.v0().size() > 0) {
    if (options_.v0().rows() != n) {
      throw std::invalid_argument("ChebyshevSolver: v0 row count does not match the problem.");
    }
    const FASTEIG_INT provided_cols = std::min<FASTEIG_INT>(nev, options_.v0().cols());
    auto initial = exec->create_multivector(options_.v0().leftCols(provided_cols));
    vectors->copy_columns_from(0, *initial, 0, provided_cols);
  }
  MatrixXd qr_factor;
  if (core::thin_qr_in_place(*vectors, qr_factor, core::QrMethod::Cholesky) != 0) {
    throw std::runtime_error("ChebyshevSolver: initial Cholesky QR failed.");
  }

  // Keep spare columns outside the Ritz space until a requested pair locks.
  // Ordered locking makes the active columns a contiguous block after the
  // locked prefix. Reallocate only when locking changes a block's dimensions.
  FASTEIG_INT locked = 0;
  FASTEIG_INT subspace_cols = nev;
  auto filtered = exec->create_multivector(n, nev);
  auto previous = exec->create_multivector(n, nev);
  auto scratch = exec->create_multivector(n, nev);
  auto basis = exec->create_multivector(n, nev);
  auto operator_basis = exec->create_multivector(n, nev);
  auto ritz_vectors = exec->create_multivector(n, nev);
  auto operator_ritz_vectors = exec->create_multivector(n, nev);
  auto residuals = exec->create_multivector(n, nev);

  double lower = 0.0;
  double upper = 0.0;
  estimate_bounds_with_executor(effective_op, seed, lower, upper);

  fasteig_logger stdout_logger;
  if (options_.verbosity() >= 2) stdout_logger.attach(&std::cout);
  fasteig_logger* log_logger =
      (print_every > 0 && problem_.log_stream()) ? &problem_.logger() : nullptr;
  fasteig_logger* console_logger =
      (print_every > 0 && options_.verbosity() >= 2) ? &stdout_logger : nullptr;
  core::ProjectedRitzPairs ritz;
  long long it_done = 0;

  for (FASTEIG_INT iter = 0; iter < maxiter; ++iter) {
    const FASTEIG_INT active_cols = subspace_cols - locked;
    if (filtered->cols() != active_cols) {
      filtered = exec->create_multivector(n, active_cols);
      previous = exec->create_multivector(n, active_cols);
      scratch = exec->create_multivector(n, active_cols);
    }
    if (basis->cols() != subspace_cols) {
      basis = exec->create_multivector(n, subspace_cols);
      operator_basis = exec->create_multivector(n, subspace_cols);
      ritz_vectors = exec->create_multivector(n, subspace_cols);
      operator_ritz_vectors = exec->create_multivector(n, subspace_cols);
      residuals = exec->create_multivector(n, subspace_cols);
    }

    // 1. Polynomial filtering: Y = T_degree(c*op + d*I) X_active.
    // The recurrence uses three reusable blocks and no host vector transfers.
    filtered->copy_columns_from(0, *vectors, locked, active_cols);
    chebyshev_filter(effective_op, *filtered, *previous, *scratch,
                     degree, lower, upper);

    // 2. Deflate against locked Ritz vectors with the shared two-pass block
    // Gram--Schmidt operation, then QR the combined current subspace.
    if (locked > 0) {
      MatrixXd projection_coefficients;
      if (core::block_gram_schmidt(*vectors, locked, *filtered, *scratch,
                                   projection_coefficients) != 0) {
        throw std::runtime_error("ChebyshevSolver: block deflation failed.");
      }
      filtered.swap(scratch);
    }
    vectors->copy_columns_from(locked, *filtered, 0, active_cols);
    basis->copy_columns_from(0, *vectors, 0, subspace_cols);
    if (core::thin_qr_in_place(*basis, qr_factor, core::QrMethod::Cholesky) != 0) {
      throw std::runtime_error("ChebyshevSolver: Cholesky QR failed during filtering.");
    }

    // 3. Rayleigh--Ritz: project H, solve the small eigenproblem, rotate Q and
    // HQ, and evaluate true Ritz residuals through the shared extraction path.
    core::apply_block(effective_op, *basis, 0, subspace_cols, *operator_basis);
    const MatrixXd projected = core::transpose_multiply(
        *basis, subspace_cols, *operator_basis, subspace_cols);
    ritz = core::extract_projected_ritz_pairs(
        projected, *basis, *operator_basis, subspace_cols, subspace_cols, "sr",
        *ritz_vectors, *operator_ritz_vectors, *residuals, "ChebyshevSolver");
    if (!ritz.values.allFinite() || !ritz.relative_residuals.allFinite()) {
      throw std::runtime_error("ChebyshevSolver: non-finite Ritz values or residuals.");
    }
    vectors->copy_columns_from(0, *ritz_vectors, 0, subspace_cols);

    // 4. Recheck the converged prefix after every QR/Rayleigh--Ritz rotation.
    // Reactivate from the first unconverged pair and retain any expanded space.
    locked = 0;
    while (locked < nev && ritz.relative_residuals(locked) <= tol) {
      ++locked;
    }
    subspace_cols = std::max(subspace_cols, nev + std::min(locked, capacity - nev));
    it_done = static_cast<long long>(iter + 1);

    // 5. Report original eigenvalues and the same relative residual definition
    // used by the final results. Convergence is checked every filter cycle.
    const bool converged = ritz.relative_residuals.head(nev).maxCoeff() <= tol;
    const bool progress_due = print_every > 0 &&
        ((iter + 1) % print_every == 0 || converged || iter + 1 == maxiter);
    if (progress_due) {
      core::LanczosSnapshot snapshot;
      snapshot.lam = ritz.values.head(nev);
      if (flip_sign) snapshot.lam *= -1.0;
      snapshot.rel_res = ritz.relative_residuals.head(nev);
      core::lanczos_emit_snapshot(snapshot, iter + 1, log_logger, console_logger);
    }
    if (converged) break;
  }

  // Return only the requested vectors. All n-by-block state stayed with the
  // executor until this final copy; residuals already refer to the original
  // standard problem (negating H and its Ritz values leaves their norms intact).
  eigvals_ = ritz.values.head(nev);
  if (flip_sign) eigvals_ *= -1.0;
  auto result = exec->create_multivector(n, nev);
  result->copy_columns_from(0, *ritz_vectors, 0, nev);
  eigvecs_ = result->copy_to_host();
  resnrms_ = ritz.relative_residuals.head(nev);
  set_iterations(it_done);
  return status_;
}

void ChebyshevSolver::chebyshev_filter(const core::LinearOperator& op,
                                       core::MultiVector& vectors,
                                       core::MultiVector& previous,
                                       core::MultiVector& scratch,
                                       FASTEIG_INT degree,
                                       FASTEIG_DOUBLE lower,
                                       FASTEIG_DOUBLE upper) {
  // A collapsed spectrum needs no amplification; Rayleigh--Ritz suffices.
  if (upper == lower) return;
  const double amplification = 1.0e3;
  const double xlb = std::cosh(std::acosh(amplification) / degree);
  const double cutoff = (2.0 * lower + upper * (xlb - 1.0)) / (1.0 + xlb);
  const double c = 2.0 / (upper - cutoff);
  const double d = -(upper + cutoff) / (upper - cutoff);
  if (!std::isfinite(c) || !std::isfinite(d)) {
    throw std::runtime_error("ChebyshevSolver: invalid filter scaling.");
  }
  const FASTEIG_INT block_cols = vectors.cols();
  const VectorXd twice_d = VectorXd::Constant(block_cols, 2.0 * d);

  // Y_0 = X, Y_1 = (c*H + d*I) X.
  previous.copy_from(vectors);
  core::apply_block(op, previous, 0, block_cols, vectors, c, d);
  core::MultiVector* prev = &previous;
  core::MultiVector* current = &vectors;
  core::MultiVector* next = &scratch;
  for (FASTEIG_INT k = 2; k <= degree; ++k) {
    // Y_k = 2*(c*H + d*I)*Y_{k-1} - Y_{k-2}.
    next->copy_from(*prev);
    core::apply_block(op, *current, 0, block_cols, *next, 2.0 * c, -1.0);
    core::add_scaled_columns(*current, twice_d, *next);
    std::swap(prev, current);
    std::swap(current, next);
  }
  if (current != &vectors) vectors.copy_from(*current);
}

}  // namespace fasteig

