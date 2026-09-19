#include "LanczosSolver.hpp"

#include "core/RitzExtraction.hpp"
#include "core/Executor.hpp"
#include "core/KrylovSubspace.hpp"
#include "core/LanczosWorkspace.hpp"
#include "core/LinearOperator.hpp"
#include "core/MultiVector.hpp"
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

namespace fasteig {

LanczosSolver::LanczosSolver(::fasteig_problem& problem,
                             const fasteig_options& options)
    : fasteig_solver(problem, options) {}

FASTEIG_INT LanczosSolver::solve() {
  const FASTEIG_INT n = eigvecs_.rows();
  const FASTEIG_INT nev = options_.nev();
  const FASTEIG_INT maxiter = options_.maxiter();
  std::string which = core::lanczos_lower_copy(options_.which());
  const int verbosity = options_.verbosity();
  const FASTEIG_INT print_every = options_.print_every();
  const FASTEIG_DOUBLE tol = options_.tol();

  if (maxiter <= 0) throw std::invalid_argument("LanczosSolver: maxiter must be > 0.");
  if (nev <= 0) throw std::invalid_argument("LanczosSolver: nev must be > 0.");
  if (maxiter < nev) throw std::invalid_argument("LanczosSolver: maxiter must be >= nev.");
  if (maxiter > n) throw std::invalid_argument("LanczosSolver: maxiter must be <= n.");
  core::validate_lanczos_which(which, /*allow_sai=*/true, "LanczosSolver");

  // Progress output setup.  A scalar Lanczos "iteration" is one Krylov vector
  // expansion, and print_every controls how often snapshots are emitted.
  fasteig_logger stdout_logger;
  if (verbosity >= 2) stdout_logger.attach(&std::cout);
  fasteig_logger* log_logger =
      (print_every > 0 && problem_.log_stream()) ? &problem_.logger() : nullptr;
  fasteig_logger* console_logger = (verbosity >= 2) ? &stdout_logger : nullptr;

  // Executor-owned objects.  CPU and CUDA differ only behind the executor,
  // operator, vectors, multivectors, and workspace implementations.
  std::shared_ptr<core::Executor> exec = problem_.executor(options_);
  std::unique_ptr<core::LinearOperator> op = problem_.make_linear_operator(options_, *exec);
  auto original_A = problem_.make_matrix_operator(*exec);
  auto metric = problem_.make_metric_operator(*exec);
  auto ritz_vectors = exec->create_multivector(n, nev);
  auto ritz_residuals = exec->create_multivector(n, nev);
  auto metric_vectors = exec->create_multivector(n, nev);
  const unsigned int base_seed = options_.seed().value_or(1u);
  auto x = (options_.v0().size() > 0)
               ? exec->create_vector(options_.v0().col(0))
               : exec->create_random_vector(n, base_seed);
  auto av = exec->create_vector(n);
  auto basis = exec->create_multivector(n, maxiter);
  auto lanczos_workspace = core::make_lanczos_workspace(*exec, n, maxiter, nev);
  core::KrylovSubspace subspace(*op, *x, *av, *basis, base_seed, metric.get());

  if (metric) {
    const char* note = "Projected residual estimates use the B inner product; final residuals use A*x-lambda*B*x.";
    if (log_logger) log_logger->print_progress_note(note);
    if (console_logger) console_logger->print_progress_note(note);
  }

  // Scalar Krylov recurrence.  After m iterations, the current Ritz space is
  // span{q_0, ..., q_{m-1}}, and the workspace stores beta_m for residual estimates.
  bool converged = false;
  bool encountered_breakdown = false;
  FASTEIG_INT it_stop = maxiter;
  FASTEIG_INT last_emitted_iter = 0;
  core::LanczosSnapshot final;

  for (FASTEIG_INT it = 0; it < maxiter; ++it) {
    // 1. Normalize the current residual into the next basis vector q_it.
    const double beta = subspace.normalize_residual_into_basis(it);

    // 2. Apply the selected linear operator: z = op * q_it.
    subspace.apply_operator_to_basis_column(it);

    // 3. Orthogonalize z against the active basis and keep the new residual.
    const double alpha = subspace.orthogonalize_operator_result(it + 1);
    const double beta_next = subspace.residual_norm();
    encountered_breakdown = encountered_breakdown ||
        beta_next <= 64.0 * std::numeric_limits<double>::epsilon() *
                         std::max({1.0, std::abs(alpha), std::abs(beta)});

    // 4. Record the projected tridiagonal coefficients.
    lanczos_workspace->record_step(it, beta, alpha, beta_next);

    const FASTEIG_INT m = it + 1;
    const bool do_progress = (print_every > 0) && ((m % print_every) == 0);
    const FASTEIG_INT check_every = (print_every > 0) ? print_every : 5;
    const bool do_check = (m >= nev) && ((m % check_every) == 0 || m == maxiter);

    // Snapshot construction diagonalizes the projected tridiagonal matrix held
    // by the workspace.  For CUDA, that projected eigensolve can stay on device.
    if (!do_progress && !do_check) continue;

    core::LanczosSnapshot snap = lanczos_workspace->compute_snapshot(
        m, nev, which, /*need_vectors=*/false, "LanczosSolver");
    if (do_progress) {
      core::lanczos_emit_snapshot(
          core::lanczos_progress_snapshot(snap, which, options_.shift(), "LanczosSolver"),
          m, log_logger, console_logger);
      last_emitted_iter = m;
    }

    // A terminated scalar chain may contain only one vector from a repeated
    // eigenspace. Allow independent replacement chains to recover multiplicity
    // instead of accepting exact residuals for the wrong spectral subset.
    const bool resolve_multiplicity = metric && encountered_breakdown && m < maxiter;

    if (!resolve_multiplicity && do_check && snap.rel_res.size() > 0 && snap.rel_res.maxCoeff() < tol) {
      final = lanczos_workspace->compute_snapshot(m, nev, which,
                                                  /*need_vectors=*/true, "LanczosSolver");
      if (metric) {
        resnrms_ = core::evaluate_ritz_pairs(
            *original_A, metric.get(), *basis, m, final, which, options_.shift(),
            *ritz_vectors, *ritz_residuals, *metric_vectors, eigvals_);
        if (!(resnrms_.array() <= tol).all()) continue;
      }
      converged = true;
      it_stop = m;
      if (!do_progress && print_every > 0) {
        core::lanczos_emit_snapshot(
            core::lanczos_progress_snapshot(final, which, options_.shift(), "LanczosSolver"),
            m, log_logger, console_logger);
      }
      break;
    }
  }

  if (!converged) {
    final = lanczos_workspace->compute_snapshot(maxiter, nev, which,
                                                /*need_vectors=*/true, "LanczosSolver");
    if (print_every > 0 && last_emitted_iter != maxiter) {
      core::lanczos_emit_snapshot(
          core::lanczos_progress_snapshot(final, which, options_.shift(), "LanczosSolver"),
          maxiter, log_logger, console_logger);
    }
  }

  // Form only the requested Ritz vectors on the executor and check the
  // original A*x-lambda*B*x residual, rather than the transformed operator.
  resnrms_ = core::evaluate_ritz_pairs(
      *original_A, metric.get(), *basis, it_stop, final, which, options_.shift(),
      *ritz_vectors, *ritz_residuals, *metric_vectors, eigvals_);
  eigvecs_ = ritz_vectors->copy_to_host();
  if (metric && !(resnrms_.array() <= tol).all()) status_ = 1;

  set_iterations(static_cast<long long>(it_stop));

  return status_;
}

}  // namespace fasteig

