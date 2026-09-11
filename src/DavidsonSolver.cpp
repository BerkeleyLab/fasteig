#include "DavidsonSolver.hpp"

#include "core/Executor.hpp"
#include "core/LinearOperator.hpp"
#include "core/MultiVector.hpp"
#include "core/Orthogonalization.hpp"
#include "core/Operations.hpp"
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
#include <vector>

namespace fasteig {

DavidsonSolver::DavidsonSolver(::fasteig_problem& problem,
                               const fasteig_options& options)
    : fasteig_solver(problem, options) {}

FASTEIG_INT DavidsonSolver::solve() {
  const FASTEIG_INT n = eigvecs_.rows();
  const FASTEIG_INT nev = options_.nev();
  const FASTEIG_INT maxiter = options_.maxiter();
  const FASTEIG_INT maxblks = options_.maxblks();
  const FASTEIG_INT print_every = options_.print_every();
  const FASTEIG_DOUBLE tol = options_.tol();
  const std::string which = core::lanczos_lower_copy(options_.which());

  if (n <= 0) throw std::invalid_argument("DavidsonSolver: n must be > 0.");
  if (nev <= 0 || nev > n) {
    throw std::invalid_argument("DavidsonSolver: nev must be in [1,n].");
  }
  if (maxiter <= 0) {
    throw std::invalid_argument("DavidsonSolver: maxiter must be > 0.");
  }
  if (maxblks < 1) {
    throw std::invalid_argument("DavidsonSolver: maxblks must be at least 1.");
  }
  if (which == "sai") {
    throw std::invalid_argument(
        "DavidsonSolver: SAI is not supported yet; use SR/LR/SM/LM/BE.");
  }
  core::validate_lanczos_which(which, /*allow_sai=*/false, "DavidsonSolver");

  // All large vectors and matrices are owned by the selected executor.  The
  // projected eigenproblem is small and remains host-resident.
  std::shared_ptr<core::Executor> exec = problem_.executor(options_);
  std::unique_ptr<core::LinearOperator> op =
      problem_.make_linear_operator(options_, *exec);
  const unsigned int base_seed = options_.seed().value_or(1u);

  auto initial_block = exec->create_random_multivector(n, nev, base_seed);
  if (options_.v0().size() > 0) {
    if (options_.v0().rows() != n) {
      throw std::invalid_argument(
          "DavidsonSolver: v0 row count does not match problem dimension.");
    }
    const FASTEIG_INT provided_cols = std::min<FASTEIG_INT>(
        nev, static_cast<FASTEIG_INT>(options_.v0().cols()));
    if (provided_cols > 0) {
      auto provided = exec->create_multivector(
          options_.v0().leftCols(static_cast<int>(provided_cols)));
      initial_block->copy_columns_from(0, *provided, 0, provided_cols);
    }
  }

  MatrixXd initial_R;
  FASTEIG_INT info =
      core::thin_qr_in_place(*initial_block, initial_R, core::QrMethod::Cholesky);
  if (info != 0) {
    throw std::runtime_error("DavidsonSolver: initial Cholesky QR failed.");
  }

  const FASTEIG_INT max_subspace_cols =
      std::min<FASTEIG_INT>(n, maxblks * nev);
  auto basis = exec->create_multivector(n, max_subspace_cols);
  auto operator_basis = exec->create_multivector(n, max_subspace_cols);
  auto initial_operator_block = exec->create_multivector(n, nev);
  auto ritz_vectors = exec->create_multivector(n, nev);
  auto operator_ritz_vectors = exec->create_multivector(n, nev);
  auto residuals = exec->create_multivector(n, nev);
  auto column_scratch = exec->create_vector(n);

  core::apply_block(*op, *initial_block, 0, nev, *initial_operator_block);
  basis->copy_columns_from(0, *initial_block, 0, nev);
  operator_basis->copy_columns_from(0, *initial_operator_block, 0, nev);

  MatrixXd projected = MatrixXd::Zero(
      static_cast<int>(max_subspace_cols), static_cast<int>(max_subspace_cols));
  projected.topLeftCorner(static_cast<int>(nev), static_cast<int>(nev)) =
      core::transpose_multiply(*basis, nev, *operator_basis, nev);

  core::ProjectedRitzPairs ritz = core::extract_projected_ritz_pairs(
      projected, *basis, *operator_basis, nev, nev, which,
      *ritz_vectors, *operator_ritz_vectors, *residuals, "DavidsonSolver");

  fasteig_logger stdout_logger;
  if (print_every > 0 && options_.verbosity() >= 2) {
    stdout_logger.attach(&std::cout);
  }
  fasteig_logger* log_logger =
      (print_every > 0 && problem_.log_stream()) ? &problem_.logger() : nullptr;
  fasteig_logger* console_logger =
      (print_every > 0 && options_.verbosity() >= 2) ? &stdout_logger : nullptr;

  auto emit_progress = [&](FASTEIG_INT iteration) {
    if (log_logger) {
      log_logger->print_convergence_data(
          ritz.values, ritz.relative_residuals, static_cast<int>(iteration));
    }
    if (console_logger) {
      console_logger->print_convergence_data(
          ritz.values, ritz.relative_residuals, static_cast<int>(iteration));
    }
  };

  // A Ritz vector is excluded from subsequent correction blocks only after
  // satisfying the possibly stricter lock tolerance.
  const FASTEIG_DOUBLE locktol = std::min<FASTEIG_DOUBLE>(1.0e-5, tol);
  long long it_done = 0;
  FASTEIG_INT last_emitted_iter = 0;

  for (FASTEIG_INT iter = 0; iter < maxiter; ++iter) {
    // Thick restart: restore V^T V = I before using a standard projected
    // eigensolve. Repeated Ritz rotations otherwise accumulate loss of
    // orthogonality, especially when corrections approach roundoff.
    MatrixXd restart_R;
    info = core::thin_qr_in_place(
        *ritz_vectors, restart_R, core::QrMethod::Householder);
    if (info != 0) {
      throw std::runtime_error("DavidsonSolver: restart QR failed.");
    }

    // QR changes the retained vectors. Refresh AV and V^T AV together rather
    // than carrying the previous Ritz values over as a diagonal projection.
    core::apply_block(*op, *ritz_vectors, 0, nev, *operator_ritz_vectors);
    basis->set_zero();
    operator_basis->set_zero();
    basis->copy_columns_from(0, *ritz_vectors, 0, nev);
    operator_basis->copy_columns_from(0, *operator_ritz_vectors, 0, nev);

    projected.setZero();
    projected.topLeftCorner(static_cast<int>(nev), static_cast<int>(nev)) =
        core::transpose_multiply(*basis, nev, *operator_basis, nev);
    FASTEIG_INT active_subspace_cols = nev;
    ritz = core::extract_projected_ritz_pairs(
        projected, *basis, *operator_basis, nev, nev, which,
        *ritz_vectors, *operator_ritz_vectors, *residuals, "DavidsonSolver");

    for (FASTEIG_INT block = 1; block < maxblks; ++block) {
      std::vector<FASTEIG_INT> active_indices;
      active_indices.reserve(static_cast<std::size_t>(nev));
      for (FASTEIG_INT j = 0; j < nev; ++j) {
        if (ritz.relative_residuals(static_cast<int>(j)) > locktol) {
          active_indices.push_back(j);
        }
      }
      const FASTEIG_INT correction_cols =
          static_cast<FASTEIG_INT>(active_indices.size());
      if (correction_cols == 0 ||
          active_subspace_cols + correction_cols > max_subspace_cols) {
        break;
      }

      // Davidson correction block.  This implementation currently uses the
      // unpreconditioned residuals; a diagonal/preconditioner operator can be
      // inserted here later without changing the executor-facing algorithm.
      auto correction_source = exec->create_multivector(n, correction_cols);
      for (FASTEIG_INT j = 0; j < correction_cols; ++j) {
        residuals->copy_column_to(active_indices[static_cast<std::size_t>(j)],
                                  *column_scratch);
        correction_source->copy_column_from(j, *column_scratch);
      }

      // Two-pass block Gram--Schmidt and thin QR are backend operations.
      auto correction = exec->create_multivector(n, correction_cols);
      MatrixXd orthogonalization_coefficients;
      info = core::block_gram_schmidt(
          *basis, active_subspace_cols, *correction_source, *correction,
          orthogonalization_coefficients);
      if (info != 0) {
        throw std::runtime_error(
            "DavidsonSolver: correction-block orthogonalization failed.");
      }

      MatrixXd correction_R;
      info = core::thin_qr_in_place(
          *correction, correction_R, core::QrMethod::Householder);
      if (info != 0) {
        throw std::runtime_error("DavidsonSolver: correction-block QR failed.");
      }

      auto operator_correction = exec->create_multivector(n, correction_cols);
      core::apply_block(*op, *correction, 0, correction_cols,
                        *operator_correction);

      // Update the small projected matrix before appending the new block.
      const MatrixXd cross = core::transpose_multiply(
          *basis, active_subspace_cols, *operator_correction, correction_cols);
      MatrixXd diagonal = core::transpose_multiply(
          *correction, correction_cols, *operator_correction, correction_cols);
      diagonal = (0.5 * (diagonal + diagonal.transpose())).eval();

      projected.block(0, static_cast<int>(active_subspace_cols),
                      static_cast<int>(active_subspace_cols),
                      static_cast<int>(correction_cols)) = cross;
      projected.block(static_cast<int>(active_subspace_cols), 0,
                      static_cast<int>(correction_cols),
                      static_cast<int>(active_subspace_cols)) = cross.transpose();
      projected.block(static_cast<int>(active_subspace_cols),
                      static_cast<int>(active_subspace_cols),
                      static_cast<int>(correction_cols),
                      static_cast<int>(correction_cols)) = diagonal;

      basis->copy_columns_from(active_subspace_cols, *correction, 0,
                               correction_cols);
      operator_basis->copy_columns_from(active_subspace_cols,
                                        *operator_correction, 0,
                                        correction_cols);
      active_subspace_cols += correction_cols;

      ritz = core::extract_projected_ritz_pairs(
          projected, *basis, *operator_basis, active_subspace_cols, nev,
          which, *ritz_vectors, *operator_ritz_vectors, *residuals,
          "DavidsonSolver");
    }

    it_done = static_cast<long long>(iter + 1);
    const bool converged =
        ritz.relative_residuals.size() > 0 &&
        ritz.relative_residuals.maxCoeff() < tol;
    const bool progress_due =
        print_every > 0 && ((iter + 1) % print_every == 0);

    if (progress_due || (converged && print_every > 0)) {
      emit_progress(iter + 1);
      last_emitted_iter = iter + 1;
    }

    if (converged) {
      break;
    }
  }

  // Always report the final available approximation when progress reporting
  // is enabled, even when maxiter is not a multiple of print_every.
  if (print_every > 0 && it_done > 0 &&
      last_emitted_iter != static_cast<FASTEIG_INT>(it_done)) {
    emit_progress(static_cast<FASTEIG_INT>(it_done));
  }

  eigvals_ = ritz.values;
  eigvecs_ = ritz_vectors->copy_to_host();
  resnrms_ = ritz.relative_residuals;
  set_iterations(it_done);
  return status_;
}

}  // namespace fasteig

