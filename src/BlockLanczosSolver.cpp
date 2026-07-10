#include "BlockLanczosSolver.hpp"

#include "core/RitzExtraction.hpp"
#include "core/Executor.hpp"
#include "core/KrylovSubspace.hpp"
#include "core/LanczosWorkspace.hpp"
#include "core/LinearOperator.hpp"
#include "core/MultiVector.hpp"
#include "fasteig_problem.hpp"
#include "fasteig_logger.hpp"

#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace fasteig {

BlockLanczosSolver::BlockLanczosSolver(::fasteig_problem& problem,
                                       const fasteig_options& options)
    : fasteig_solver(problem, options) {}

FASTEIG_INT BlockLanczosSolver::solve() {
  const FASTEIG_INT n = eigvecs_.rows();
  const FASTEIG_INT nev = options_.nev();
  const FASTEIG_INT maxiter = options_.maxiter();     // number of block iterations
  const std::string which = options_.which();
  const FASTEIG_INT blksize = options_.blksize();
  const int verbosity = options_.verbosity();
  const FASTEIG_DOUBLE tol = options_.tol();

  if (n <= 0) throw std::invalid_argument("BlockLanczosSolver: n must be > 0.");
  if (nev <= 0) throw std::invalid_argument("BlockLanczosSolver: nev must be > 0.");
  if (maxiter <= 0) throw std::invalid_argument("BlockLanczosSolver: maxiter must be > 0.");
  if (blksize <= 0) throw std::invalid_argument("BlockLanczosSolver: blksize must be > 0.");

  const int b = static_cast<int>(blksize);
  const int max_blocks = static_cast<int>(maxiter);
  const int subspace_dim = max_blocks * b;
  if (subspace_dim < static_cast<int>(nev)) {
    throw std::invalid_argument("BlockLanczosSolver: need maxiter*blksize >= nev.");
  }

  const std::string sigma = core::lanczos_lower_copy(which);
  core::validate_lanczos_which(sigma, /*allow_sai=*/false, "BlockLanczosSolver");

  // Progress output mirrors LanczosSolver's logger setup.  For Block Lanczos,
  // one "iteration" means one block expansion, so enabled progress is emitted
  // after each block rather than after print_every scalar matvecs.
  fasteig_logger stdout_logger;
  if (verbosity >= 2) stdout_logger.attach(&std::cout);
  fasteig_logger* log_logger = problem_.log_stream() ? &problem_.logger() : nullptr;
  fasteig_logger* console_logger = (verbosity >= 2) ? &stdout_logger : nullptr;

  // Executor-owned objects.  CPU and CUDA differ only behind the executor,
  // operator, multivectors, and workspace implementations.
  std::shared_ptr<core::Executor> exec = problem_.executor(options_);
  std::unique_ptr<core::LinearOperator> op = problem_.make_linear_operator(options_, *exec);
  const FASTEIG_INT projection_dim = static_cast<FASTEIG_INT>((max_blocks + 1) * b);
  const unsigned int base_seed = options_.seed().value_or(1u);
  if (options_.v0().size() > 0 &&
      (options_.v0().rows() != n || options_.v0().cols() < b)) {
    throw std::invalid_argument("BlockLanczosSolver: v0 must have shape n x at least blksize.");
  }
  auto initial_block = (options_.v0().size() > 0)
                           ? exec->create_multivector(options_.v0().leftCols(b))
                           : exec->create_random_multivector(n, blksize, base_seed);
  auto basis = exec->create_multivector(n, projection_dim);
  auto lanczos_workspace = core::make_lanczos_workspace(*exec, n, projection_dim, nev);

  // Block Krylov recurrence.  After iter block expansions, the current Ritz
  // space is span{Q_0, ..., Q_{iter-1}} with dimension iter*blksize, and the
  // workspace stores the coupling to Q_iter for the residual estimate.
  bool converged = false;
  int it_stop = max_blocks;
  int last_emitted_iter = 0;
  core::LanczosSnapshot final;
  core::BlockKrylovSubspace subspace(*op, *initial_block, *basis);
  subspace.initialize_basis_block();

  for (int iter = 1; iter <= max_blocks; ++iter) {
    // 1. Apply the selected linear operator to the previous block:
    //    Z = op * Q_{iter-1}.
    subspace.apply_operator_to_basis_block(static_cast<FASTEIG_INT>(iter - 1));

    // 2. Orthogonalize Z against all active basis blocks Q_0 ... Q_{iter-1}.
    subspace.orthogonalize_operator_block(static_cast<FASTEIG_INT>(iter * b));

    // 3. QR-factor the residual block.  The Q factor becomes Q_iter and the
    //    R factor is the block coupling used in the projected matrix.
    subspace.qr_residual_block(static_cast<FASTEIG_INT>(iter));

    // 4. Record the block projected matrix coefficients.
    lanczos_workspace->record_block_step(static_cast<FASTEIG_INT>(iter - 1), blksize,
                                         subspace.diagonal_block(),
                                         subspace.coupling_block());

    // Always compute/check convergence once the subspace is large enough to represent nev.
    const bool do_check = (iter * b >= static_cast<int>(nev));
    const bool do_progress = (log_logger != nullptr) || (console_logger != nullptr);
    if (!do_check && !do_progress) continue;

    core::LanczosSnapshot snap = lanczos_workspace->compute_block_snapshot(
        static_cast<FASTEIG_INT>(iter), blksize, nev, sigma,
        /*need_vectors=*/false, "BlockLanczosSolver");
    if (do_progress) {
      core::lanczos_emit_snapshot(snap, iter, log_logger, console_logger);
      last_emitted_iter = iter;
    }

    if (do_check && snap.rel_res.size() > 0 && snap.rel_res.maxCoeff() < tol) {
      converged = true;
      it_stop = iter;
      final = lanczos_workspace->compute_block_snapshot(
          static_cast<FASTEIG_INT>(iter), blksize, nev, sigma,
          /*need_vectors=*/true, "BlockLanczosSolver");
      break;
    }
  }

  if (!converged) {
    final = lanczos_workspace->compute_block_snapshot(
        static_cast<FASTEIG_INT>(max_blocks), blksize, nev, sigma,
        /*need_vectors=*/true, "BlockLanczosSolver");
    // Ensure we have a final progress record even if verbosity==1.
    if ((log_logger != nullptr || console_logger != nullptr) && last_emitted_iter != max_blocks) {
      core::lanczos_emit_snapshot(final, max_blocks, log_logger, console_logger);
    }
  }

  // Final Ritz vectors are formed by the workspace so the CPU/CUDA choice stays
  // hidden from the solver.  CUDA keeps the basis and GEMM on device until the
  // final eigenvectors are copied back for user-facing results.
  const int m_dim = it_stop * b;
  lanczos_workspace->finalize_eigenpairs(*op, *basis, static_cast<FASTEIG_INT>(m_dim),
                                         final, sigma, options_.shift(),
                                         eigvals_, eigvecs_, resnrms_);

  set_iterations(static_cast<long long>(it_stop));
  return status_;
}

}  // namespace fasteig

