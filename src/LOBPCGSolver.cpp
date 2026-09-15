#include "LOBPCGSolver.hpp"

#include "core/Executor.hpp"
#include "core/LinearOperator.hpp"
#include "core/MultiVector.hpp"
#include "core/Operations.hpp"
#include "core/Orthogonalization.hpp"
#include "core/RitzExtraction.hpp"
#include "fasteig_logger.hpp"
#include "fasteig_problem.hpp"

#include <algorithm>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace fasteig {

LOBPCGSolver::LOBPCGSolver(::fasteig_problem& problem,
                           const fasteig_options& options)
    : fasteig_solver(problem, options) {}

FASTEIG_INT LOBPCGSolver::solve() {
  const FASTEIG_INT n = eigvecs_.rows();
  const FASTEIG_INT nev = options_.nev();
  const FASTEIG_INT maxiter = options_.maxiter();
  const FASTEIG_INT print_every = options_.print_every();
  const double tol = options_.tol();
  const std::string which = core::lanczos_lower_copy(options_.which());
  if (n <= 0 || nev <= 0 || nev > n || maxiter <= 0) {
    throw std::invalid_argument("LOBPCGSolver: require 0 < nev <= n and maxiter > 0.");
  }
  if (which != "sr" && which != "lr") {
    throw std::invalid_argument("LOBPCGSolver: only SR and LR are supported.");
  }
  if (problem_.is_generalized()) {
    throw std::invalid_argument("LOBPCGSolver: generalized problems are not supported.");
  }

  // 1. Initialize X in executor memory. LR minimizes the Rayleigh quotient of
  // -A; eigenvalues are converted back for both progress and final output.
  auto exec = problem_.executor(options_);
  auto op = problem_.make_linear_operator(options_, *exec);
  const bool flip_sign = which == "lr";
  core::ScaledLinearOperator effective_op(*op, flip_sign ? -1.0 : 1.0);
  auto X = exec->create_random_multivector(n, nev, options_.seed().value_or(1u));
  if (options_.v0().size() > 0) {
    if (options_.v0().rows() != n) {
      throw std::invalid_argument("LOBPCGSolver: v0 row count does not match the problem.");
    }
    const FASTEIG_INT provided_cols = std::min<FASTEIG_INT>(nev, options_.v0().cols());
    auto initial = exec->create_multivector(options_.v0().leftCols(provided_cols));
    X->copy_columns_from(0, *initial, 0, provided_cols);
  }
  MatrixXd qr_factor;
  if (core::thin_qr_in_place(*X, qr_factor, core::QrMethod::Householder) != 0) {
    throw std::runtime_error("LOBPCGSolver: initial QR failed.");
  }

  // At most three blocks: current Ritz vectors X, residual directions W, and
  // previous search directions P. P is empty on the first iteration.
  const FASTEIG_INT capacity = static_cast<FASTEIG_INT>(
      std::min<long long>(n, 3LL * nev));
  auto basis = exec->create_multivector(n, capacity);
  auto operator_basis = exec->create_multivector(n, capacity);
  auto AX = exec->create_multivector(n, nev);
  auto residuals = exec->create_multivector(n, nev);
  std::unique_ptr<core::MultiVector> P;

  basis->copy_columns_from(0, *X, 0, nev);
  core::apply_block(effective_op, *X, 0, nev, *AX);
  operator_basis->copy_columns_from(0, *AX, 0, nev);
  MatrixXd projected = core::transpose_multiply(*X, nev, *AX, nev);
  auto ritz = core::extract_projected_ritz_pairs(
      projected, *basis, *operator_basis, nev, nev, "sr", *X, *AX, *residuals,
      "LOBPCGSolver");

  fasteig_logger stdout_logger;
  if (options_.verbosity() >= 2) stdout_logger.attach(&std::cout);
  auto* log_logger = (print_every > 0 && problem_.log_stream()) ? &problem_.logger() : nullptr;
  auto* console_logger = (print_every > 0 && options_.verbosity() >= 2) ? &stdout_logger : nullptr;
  long long it_done = 0;

  for (FASTEIG_INT iter = 0; iter < maxiter; ++iter) {
    if (!ritz.values.allFinite() || !ritz.relative_residuals.allFinite()) {
      throw std::runtime_error("LOBPCGSolver: non-finite Ritz values or residuals.");
    }
    if (ritz.relative_residuals.maxCoeff() <= tol) break;

    // 2. Soft locking: retain every X in Rayleigh--Ritz, but expand only with
    // residuals of unconverged pairs. No preconditioner is applied yet (W=R).
    // Unlike hard locking, a pair can become active again after a Ritz update.
    std::vector<FASTEIG_INT> active;
    for (FASTEIG_INT j = 0; j < nev; ++j) {
      if (ritz.relative_residuals(j) > tol) active.push_back(j);
    }
    const FASTEIG_INT active_count = static_cast<FASTEIG_INT>(active.size());
    auto W_source = exec->create_multivector(n, active_count);
    for (FASTEIG_INT j = 0; j < active_count; ++j) {
      W_source->copy_columns_from(j, *residuals, active[j], 1);
    }

    // 3. Form an orthonormal basis of span{X,W,P}. Restore X orthogonality to
    // avoid accumulated Ritz rotations corrupting the standard projection.
    // The shared block routine drops dependent directions and reorthogonalizes
    // after normalization; it never completes a vanished residual with noise.
    if (core::thin_qr_in_place(*X, qr_factor, core::QrMethod::Householder) != 0) {
      throw std::runtime_error("LOBPCGSolver: Ritz-vector QR failed.");
    }
    basis->copy_columns_from(0, *X, 0, nev);
    FASTEIG_INT subspace_cols = nev;
    auto W = core::orthonormalize_block(*basis, subspace_cols, *W_source);
    if (W) {
      basis->copy_columns_from(subspace_cols, *W, 0, W->cols());
      subspace_cols += W->cols();
    }
    if (P && subspace_cols < capacity) {
      auto P_source = exec->create_multivector(n, active_count);
      for (FASTEIG_INT j = 0; j < active_count; ++j) {
        P_source->copy_columns_from(j, *P, active[j], 1);
      }
      auto P_orthogonal = core::orthonormalize_block(*basis, subspace_cols, *P_source);
      if (P_orthogonal) {
        basis->copy_columns_from(subspace_cols, *P_orthogonal, 0, P_orthogonal->cols());
        subspace_cols += P_orthogonal->cols();
      }
    }

    // 4. Rayleigh--Ritz in S=[X,W,P]. Refresh A*S after orthogonalization so
    // residuals are based on the actual current basis, not drifting cached A*X.
    // Only the <=3*nev projection and Ritz coefficients live on the host.
    core::apply_block(effective_op, *basis, 0, subspace_cols, *operator_basis);
    projected = core::transpose_multiply(*basis, subspace_cols, *operator_basis, subspace_cols);
    ritz = core::extract_projected_ritz_pairs(
        projected, *basis, *operator_basis, subspace_cols, nev, "sr",
        *X, *AX, *residuals, "LOBPCGSolver");
    if (!ritz.values.allFinite() || !ritz.relative_residuals.allFinite()) {
      throw std::runtime_error("LOBPCGSolver: non-finite Ritz values or residuals.");
    }

    // 5. Keep the locally optimal search directions:
    // X_new = X_old*C_X + W*C_W + P_old*C_P,
    // P_new = W*C_W + P_old*C_P. They are orthogonalized on the next iteration.
    // Zeroing C_X implements this with the same executor GEMM as the Ritz build.
    if (!P) P = exec->create_multivector(n, nev);
    MatrixXd direction_coefficients = ritz.coefficients;
    direction_coefficients.topRows(nev).setZero();
    basis->right_multiply_active_columns(subspace_cols, direction_coefficients, *P);

    // 6. Check true relative residuals and report original eigenvalues.
    it_done = static_cast<long long>(iter + 1);
    const bool converged = ritz.relative_residuals.maxCoeff() <= tol;
    const bool exhausted = subspace_cols == nev;
    if (print_every > 0 && ((iter + 1) % print_every == 0 || converged || exhausted || iter + 1 == maxiter)) {
      core::LanczosSnapshot snapshot;
      snapshot.lam = ritz.values;
      if (flip_sign) snapshot.lam *= -1.0;
      snapshot.rel_res = ritz.relative_residuals;
      core::lanczos_emit_snapshot(snapshot, iter + 1, log_logger, console_logger);
    }
    if (converged || exhausted) break;
  }

  // Transfer the requested vectors only once, at the public result boundary.
  eigvals_ = ritz.values;
  if (flip_sign) eigvals_ *= -1.0;
  eigvecs_ = X->copy_to_host();
  resnrms_ = ritz.relative_residuals;
  set_iterations(it_done);
  return status_;
}

}  // namespace fasteig

