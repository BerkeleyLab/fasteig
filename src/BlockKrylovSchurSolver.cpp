#include "BlockKrylovSchurSolver.hpp"

#include "core/Executor.hpp"
#include "core/KrylovSubspace.hpp"
#include "core/LanczosWorkspace.hpp"
#include "core/LinearOperator.hpp"
#include "core/MultiVector.hpp"
#include "core/RitzExtraction.hpp"
#include "fasteig_logger.hpp"
#include "fasteig_problem.hpp"

#include <algorithm>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace fasteig {

BlockKrylovSchurSolver::BlockKrylovSchurSolver(
    ::fasteig_problem& problem,
    const fasteig_options& options)
    : fasteig_solver(problem, options) {}

FASTEIG_INT BlockKrylovSchurSolver::solve() {
  const FASTEIG_INT n = eigvecs_.rows();
  const FASTEIG_INT nev = options_.nev();
  const FASTEIG_INT maxiter = options_.maxiter();  // number of restart cycles
  const FASTEIG_DOUBLE tol = options_.tol();
  const FASTEIG_INT print_every = options_.print_every();
  const int verbosity = options_.verbosity();
  const std::string which = core::lanczos_lower_copy(options_.which());

  if (n <= 0) throw std::invalid_argument("BlockKrylovSchurSolver: n must be > 0.");
  if (nev <= 0 || nev > n) {
    throw std::invalid_argument("BlockKrylovSchurSolver: nev must be in [1,n].");
  }
  if (maxiter <= 0) {
    throw std::invalid_argument("BlockKrylovSchurSolver: maxiter must be > 0.");
  }
  if (problem_.is_generalized() && which == "sai") {
    throw std::invalid_argument("Generalized SAI is currently supported only by KrylovSchur.");
  }
  core::validate_lanczos_which(
      which, /*allow_sai=*/true, "BlockKrylovSchurSolver");

  // Block width is independent of the number of requested eigenpairs. Round
  // the cycle down to whole blocks, except for a partial block at full space.
  const FASTEIG_INT block_size = options_.blksize();
  if (block_size <= 0 || block_size > n) {
    throw std::invalid_argument("BlockKrylovSchurSolver: blksize must be in [1,n].");
  }
  const FASTEIG_INT requested_dim =
      options_.maxdim() > 0 ? options_.maxdim() : std::max<FASTEIG_INT>(2 * nev, 20);
  const FASTEIG_INT block_count = requested_dim >= n
      ? 1 + (n - 1) / block_size : requested_dim / block_size;
  const FASTEIG_INT cycle_dim = std::min(n, block_count * block_size);
  const FASTEIG_INT wanted_blocks = 1 + (nev - 1) / block_size;
  if (cycle_dim < n && block_count <= wanted_blocks) {
    throw std::invalid_argument(
        "BlockKrylovSchurSolver: maxdim must accommodate ceil(nev/blksize) "
        "retained blocks plus at least one expansion block (or the full matrix dimension).");
  }
  const FASTEIG_INT projection_dim = (block_count + 1) * block_size;

  // Thick restart: keep the requested pairs and, when there is room, guard
  // Ritz vectors up to roughly half the cycle. Leave a full block to expand.
  // Retention and expansion widths need not be the same.
  const FASTEIG_INT retained_blocks = cycle_dim == n ? 0
      : std::max(wanted_blocks, block_count / 2);
  const FASTEIG_INT retained_dim = retained_blocks * block_size;
  const FASTEIG_INT snapshot_count = std::max(nev, retained_dim);

  fasteig_logger stdout_logger;
  if (verbosity >= 2) stdout_logger.attach(&std::cout);
  fasteig_logger* log_logger =
      (print_every > 0 && problem_.log_stream()) ? &problem_.logger() : nullptr;
  fasteig_logger* console_logger = (verbosity >= 2) ? &stdout_logger : nullptr;

  // Executor-owned state. CPU and CUDA share the algorithm below; vectors,
  // block operations, QR, and projected storage are backend implementations.
  // For SAI the operator factors A - shift*I once and reuses those factors
  // for every block solve, including across restarts.
  std::shared_ptr<core::Executor> exec = problem_.executor(options_);
  std::unique_ptr<core::LinearOperator> op =
      problem_.make_linear_operator(options_, *exec);
  auto original_A = problem_.make_matrix_operator(*exec);
  auto metric = problem_.make_metric_operator(*exec);
  auto ritz_vectors = exec->create_multivector(n, nev);
  auto ritz_residuals = exec->create_multivector(n, nev);
  auto metric_vectors = exec->create_multivector(n, nev);
  const unsigned int base_seed = options_.seed().value_or(1u);
  if (options_.v0().size() > 0 &&
      (options_.v0().rows() != n || options_.v0().cols() < block_size)) {
    throw std::invalid_argument(
        "BlockKrylovSchurSolver: v0 must have n rows and at least blksize columns.");
  }
  auto initial_block =
      (options_.v0().size() > 0)
          ? exec->create_multivector(
                options_.v0().leftCols(static_cast<int>(block_size)))
          : exec->create_random_multivector(n, block_size, base_seed);
  auto basis = exec->create_multivector(n, projection_dim);
  auto projection =
      core::make_lanczos_workspace(*exec, n, projection_dim, nev);
  core::BlockKrylovSubspace subspace(*op, *initial_block, *basis, metric.get(), base_seed);

  // Expand Q from first_block through the final active block. Q_block_count is
  // the extra residual block used to estimate Ritz residuals.
  auto expand_subspace = [&](FASTEIG_INT first_block) {
    for (FASTEIG_INT block = first_block; block < block_count; ++block) {
      // 1. Z = op * Q_block.
      subspace.apply_operator_to_basis_block(block);

      // 2. Orthogonalize Z against all active basis blocks.
      subspace.orthogonalize_operator_block(std::min(n, (block + 1) * block_size));

      // 3. QR-factor the residual; its Q factor becomes Q_{block+1}.
      subspace.qr_residual_block(block + 1);

      // 4. Store the diagonal and coupling blocks of the projection.
      projection->record_block_step(block, block_size,
                                    subspace.diagonal_block(),
                                    subspace.coupling_block());
    }
  };

  if (metric) {
    const char* note = "Projected residual estimates use the B inner product; final residuals use A*x-lambda*B*x.";
    if (log_logger) log_logger->print_progress_note(note);
    if (console_logger) console_logger->print_progress_note(note);
  }

  subspace.initialize_basis_block();
  expand_subspace(0);

  core::LanczosSnapshot final;
  FASTEIG_INT final_iteration = 0;
  bool last_was_printed = false;
  bool converged = false;

  for (FASTEIG_INT iteration = 1; iteration <= maxiter; ++iteration) {
    // Rayleigh--Ritz extraction from the block projected matrix. CUDA keeps
    // the projected matrix and dense eigensolve in its workspace.
    core::LanczosSnapshot retained = projection->compute_block_snapshot(
        block_count, block_size, snapshot_count, which, /*need_vectors=*/true,
        "BlockKrylovSchurSolver", cycle_dim);
    auto snapshot = core::lanczos_snapshot_head(retained, nev, "BlockKrylovSchurSolver");
    final_iteration = iteration;

    const bool do_progress =
        (print_every > 0) && ((iteration % print_every) == 0);
    if (do_progress) {
      core::lanczos_emit_snapshot(
          core::lanczos_progress_snapshot(
              snapshot, which, options_.shift(), "BlockKrylovSchurSolver"),
          iteration, log_logger, console_logger);
    }
    last_was_printed = do_progress;

    converged =
        snapshot.rel_res.size() > 0 && snapshot.rel_res.maxCoeff() < tol;
    const bool at_limit = iteration == maxiter || cycle_dim == n;
    if (converged || at_limit) {
      resnrms_ = core::evaluate_ritz_pairs(
          *original_A, metric.get(), *basis, cycle_dim, snapshot, which, options_.shift(),
          *ritz_vectors, *ritz_residuals, *metric_vectors, eigvals_);
      converged = (resnrms_.array() <= tol).all();
    }
    if (converged || at_limit) {
      final = std::move(snapshot);
      break;
    }

    // Block Krylov--Schur restart:
    //   U <- Q*C_keep, append Q_F after U,
    //   T_keep <- diag(theta_keep), F <- beta*C_keep_last.
    // F is blksize-by-retained_dim: keep its FULL dense coupling to U.
    // Large basis transformations remain on the selected executor; only the
    // small projected coefficients are host-visible.
    // In SAI, theta is still an inverse Ritz value mu (selected by largest
    // magnitude). Backtransformation is only for reporting and finalization.
    const MatrixXd restart_coupling =
        subspace.coupling_block() *
        retained.Csel.bottomRows(static_cast<int>(block_size));
    subspace.restart_with_ritz_block(
        cycle_dim, retained.Csel, block_count);
    projection->prepare_block_krylov_schur_restart(
        block_size, retained.lam, restart_coupling,
        "BlockKrylovSchurSolver");

    // Resume at the preserved residual block, after all retained Ritz blocks.
    expand_subspace(retained_blocks);
  }

  if (final.lam.size() == 0 || final.Csel.size() == 0) {
    throw std::logic_error(
        "BlockKrylovSchurSolver: solve terminated without a final Ritz snapshot.");
  }
  if (print_every > 0 && !last_was_printed) {
    core::lanczos_emit_snapshot(
        core::lanczos_progress_snapshot(
            final, which, options_.shift(), "BlockKrylovSchurSolver"),
        final_iteration, log_logger, console_logger);
  }

  // The retained basis, B images and Ritz-vector products stay on the executor.
  resnrms_ = core::evaluate_ritz_pairs(
      *original_A, metric.get(), *basis, cycle_dim, final, which, options_.shift(),
      *ritz_vectors, *ritz_residuals, *metric_vectors, eigvals_);
  eigvecs_ = ritz_vectors->copy_to_host();
  status_ = (resnrms_.array() <= tol).all() ? 0 : 1;

  set_iterations(static_cast<long long>(final_iteration));
  return status_;
}

}  // namespace fasteig

