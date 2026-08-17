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
  if (problem_.is_generalized()) {
    throw std::invalid_argument(
        "BlockKrylovSchurSolver: generalized eigenproblems are not supported yet.");
  }
  core::validate_lanczos_which(
      which, /*allow_sai=*/false, "BlockKrylovSchurSolver");

  // This follows the original bks.m convention: the requested eigenpair count
  // is also the block width. maxdim controls the restarted Krylov dimension.
  const FASTEIG_INT block_size = nev;
  const FASTEIG_INT requested_dim =
      (options_.maxdim() > 0)
          ? options_.maxdim()
          : std::max<FASTEIG_INT>(2 * nev, 20);
  if (requested_dim < 2 * block_size) {
    throw std::invalid_argument(
        "BlockKrylovSchurSolver: maxdim must be at least 2*nev.");
  }

  const FASTEIG_INT block_count =
      std::min<FASTEIG_INT>(requested_dim / block_size, n / block_size);
  if (block_count < 2) {
    throw std::invalid_argument(
        "BlockKrylovSchurSolver: matrix dimension must accommodate at least "
        "two nev-sized blocks.");
  }
  const FASTEIG_INT cycle_dim = block_count * block_size;
  const FASTEIG_INT projection_dim = cycle_dim + block_size;

  fasteig_logger stdout_logger;
  if (verbosity >= 2) stdout_logger.attach(&std::cout);
  fasteig_logger* log_logger =
      (print_every > 0 && problem_.log_stream()) ? &problem_.logger() : nullptr;
  fasteig_logger* console_logger = (verbosity >= 2) ? &stdout_logger : nullptr;

  // Executor-owned state. CPU and CUDA share the algorithm below; vectors,
  // block operations, QR, and projected storage are backend implementations.
  std::shared_ptr<core::Executor> exec = problem_.executor(options_);
  std::unique_ptr<core::LinearOperator> op =
      problem_.make_linear_operator(options_, *exec);
  const unsigned int base_seed = options_.seed().value_or(1u);
  if (options_.v0().size() > 0 &&
      (options_.v0().rows() != n || options_.v0().cols() < block_size)) {
    throw std::invalid_argument(
        "BlockKrylovSchurSolver: v0 must have n rows and at least nev columns.");
  }
  auto initial_block =
      (options_.v0().size() > 0)
          ? exec->create_multivector(
                options_.v0().leftCols(static_cast<int>(block_size)))
          : exec->create_random_multivector(n, block_size, base_seed);
  auto basis = exec->create_multivector(n, projection_dim);
  auto projection =
      core::make_lanczos_workspace(*exec, n, projection_dim, nev);
  core::BlockKrylovSubspace subspace(*op, *initial_block, *basis);

  // Expand Q from first_block through the final active block. Q_block_count is
  // the extra residual block used to estimate Ritz residuals.
  auto expand_subspace = [&](FASTEIG_INT first_block) {
    for (FASTEIG_INT block = first_block; block < block_count; ++block) {
      // 1. Z = op * Q_block.
      subspace.apply_operator_to_basis_block(block);

      // 2. Orthogonalize Z against all active basis blocks.
      subspace.orthogonalize_operator_block((block + 1) * block_size);

      // 3. QR-factor the residual; its Q factor becomes Q_{block+1}.
      subspace.qr_residual_block(block + 1);

      // 4. Store the diagonal and coupling blocks of the projection.
      projection->record_block_step(block, block_size,
                                    subspace.diagonal_block(),
                                    subspace.coupling_block());
    }
  };

  subspace.initialize_basis_block();
  expand_subspace(0);

  core::LanczosSnapshot final;
  FASTEIG_INT final_iteration = 0;
  bool last_was_printed = false;

  for (FASTEIG_INT iteration = 1; iteration <= maxiter; ++iteration) {
    // Rayleigh--Ritz extraction from the block projected matrix. CUDA keeps
    // the projected matrix and dense eigensolve in its workspace.
    core::LanczosSnapshot snapshot = projection->compute_block_snapshot(
        block_count, block_size, nev, which, /*need_vectors=*/true,
        "BlockKrylovSchurSolver");
    final_iteration = iteration;

    const bool do_progress =
        (print_every > 0) && ((iteration % print_every) == 0);
    if (do_progress) {
      core::lanczos_emit_snapshot(
          snapshot, iteration, log_logger, console_logger);
    }
    last_was_printed = do_progress;

    const bool converged =
        snapshot.rel_res.size() > 0 && snapshot.rel_res.maxCoeff() < tol;
    if (converged || iteration == maxiter) {
      final = std::move(snapshot);
      break;
    }

    // Block Krylov--Schur restart:
    //   Q_0 <- Q*C,  Q_1 <- Q_F,
    //   T(0,0) <- diag(theta),  T(1,0) <- beta*C_last.
    // Large basis transformations remain on the selected executor; only the
    // small projected coefficients are host-visible.
    const MatrixXd restart_coupling =
        subspace.coupling_block() *
        snapshot.Csel.bottomRows(static_cast<int>(block_size));
    subspace.restart_with_ritz_block(
        cycle_dim, snapshot.Csel, block_count);
    projection->prepare_block_krylov_schur_restart(
        block_size, snapshot.lam, restart_coupling,
        "BlockKrylovSchurSolver");

    // Q_0 and Q_1 are supplied by the restart, so expansion resumes at Q_1.
    expand_subspace(1);
  }

  if (final.lam.size() == 0 || final.Csel.size() == 0) {
    throw std::logic_error(
        "BlockKrylovSchurSolver: solve terminated without a final Ritz snapshot.");
  }
  if (print_every > 0 && !last_was_printed) {
    core::lanczos_emit_snapshot(
        final, final_iteration, log_logger, console_logger);
  }

  // CUDA performs the basis-times-coefficient GEMM and residual checks on
  // device, then returns only the requested user-facing eigenpairs.
  projection->finalize_eigenpairs(
      *op, *basis, cycle_dim, final, which, options_.shift(),
      eigvals_, eigvecs_, resnrms_);

  set_iterations(static_cast<long long>(final_iteration));
  return status_;
}

}  // namespace fasteig

