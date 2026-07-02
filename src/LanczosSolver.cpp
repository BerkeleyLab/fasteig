#include "LanczosSolver.hpp"

#include "LanczosCommon.hpp"
#include "core/Executor.hpp"
#include "core/KrylovSubspace.hpp"
#include "core/LanczosWorkspace.hpp"
#include "core/LinearOperator.hpp"
#include "core/MultiVector.hpp"
#include "core/Vector.hpp"
#include "fasteig_logger.hpp"
#include "fasteig_problem.hpp"

#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fasteig {

LanczosSolver::LanczosSolver(::fasteig_problem& problem,
                             const fasteig_options& options)
    : fasteig_solver(problem, options) {}

FASTEIG_INT LanczosSolver::solve() {
  const FASTEIG_INT n = eigvecs_.rows();
  const FASTEIG_INT nev = options_.nev();
  const FASTEIG_INT maxiter = options_.maxiter();
  std::string which = lanczos_lower_copy(options_.which());
  const int verbosity = options_.verbosity();
  const FASTEIG_INT print_every = options_.print_every();
  const FASTEIG_DOUBLE tol = options_.tol();

  if (maxiter <= 0) throw std::invalid_argument("LanczosSolver: maxiter must be > 0.");
  if (nev <= 0) throw std::invalid_argument("LanczosSolver: nev must be > 0.");
  if (maxiter < nev) throw std::invalid_argument("LanczosSolver: maxiter must be >= nev.");
  if (maxiter > n) throw std::invalid_argument("LanczosSolver: maxiter must be <= n.");
  validate_lanczos_which(which, /*allow_sai=*/true, "LanczosSolver");

  fasteig_logger stdout_logger;
  if (verbosity >= 2) stdout_logger.attach(&std::cout);
  fasteig_logger* log_logger =
      (print_every > 0 && problem_.log_stream()) ? &problem_.logger() : nullptr;
  fasteig_logger* console_logger = (verbosity >= 2) ? &stdout_logger : nullptr;

  MatrixXd V;

  std::shared_ptr<core::Executor> exec = problem_.executor(options_);
  std::unique_ptr<core::LinearOperator> op = problem_.make_linear_operator(options_, *exec);
  const unsigned int base_seed = options_.seed().value_or(1u);
  auto x = (options_.v0().size() > 0)
               ? exec->create_vector(options_.v0().col(0))
               : exec->create_random_vector(n, base_seed);
  auto av = exec->create_vector(n);
  auto basis = exec->create_multivector(n, maxiter);
  auto lanczos_workspace = core::make_lanczos_workspace(*exec, n, maxiter, nev);
  core::KrylovSubspace krylov(*op, *x, *av, *basis, *lanczos_workspace, base_seed);

  bool converged = false;
  FASTEIG_INT it_stop = maxiter;
  FASTEIG_INT last_emitted_iter = 0;
  LanczosSnapshot final;

  for (FASTEIG_INT it = 0; it < maxiter; ++it) {
    krylov.build(it, it + 1);

    const FASTEIG_INT m = it + 1;
    const bool do_progress = (print_every > 0) && ((m % print_every) == 0);
    const FASTEIG_INT check_every = (print_every > 0) ? print_every : 5;
    const bool do_check = (m >= nev) && ((m % check_every) == 0 || m == maxiter);

    if (!do_progress && !do_check) continue;

    LanczosSnapshot snap = lanczos_workspace->compute_snapshot(
        m, nev, which, /*need_vectors=*/false, "LanczosSolver");
    if (do_progress) {
      lanczos_emit_snapshot(snap, m, log_logger, console_logger);
      last_emitted_iter = m;
    }

    if (do_check && snap.rel_res.size() > 0 && snap.rel_res.maxCoeff() < tol) {
      converged = true;
      it_stop = m;
      final = lanczos_workspace->compute_snapshot(m, nev, which,
                                                  /*need_vectors=*/true, "LanczosSolver");
      if (!do_progress && print_every > 0) {
        lanczos_emit_snapshot(final, m, log_logger, console_logger);
      }
      break;
    }
  }

  if (!converged) {
    final = lanczos_workspace->compute_snapshot(maxiter, nev, which,
                                                /*need_vectors=*/true, "LanczosSolver");
    if (print_every > 0 && last_emitted_iter != maxiter) {
      lanczos_emit_snapshot(final, maxiter, log_logger, console_logger);
    }
  }

  if (lanczos_workspace->needs_host_basis_for_ritz_vectors() || which == "sai") {
    V = basis->copy_to_host();
  }

  if (which == "sai") {
    const FASTEIG_DOUBLE shift = options_.shift();
    std::vector<size_t> Ind(nev, 0);
    if (V.cols() < it_stop || final.Csel.rows() < it_stop || final.Csel.cols() < nev) {
      throw std::runtime_error("LanczosSolver: SAI Ritz-vector assembly has inconsistent basis dimensions.");
    }

    for (size_t j = 0; j < static_cast<size_t>(nev); ++j) {
      final.lam(static_cast<int>(j)) = 1.0 / final.lam(static_cast<int>(j)) + shift;
    }

    Ind = sort_indices(final.lam, "sr");
    for (size_t j = 0; j < static_cast<size_t>(nev); ++j) {
      eigvals_(static_cast<int>(j)) = final.lam(static_cast<int>(Ind[j]));
      eigvecs_.col(static_cast<int>(j)) =
          V.leftCols(static_cast<int>(it_stop)) * final.Csel.col(static_cast<int>(Ind[j]));
    }
  } else {
    lanczos_workspace->finalize_eigenpairs(*op, *basis, V, it_stop, final,
                                           eigvals_, eigvecs_, resnrms_);
  }

  if (which == "sai") compute_final_residual_norms();

  set_iterations(static_cast<long long>(it_stop));

  return status_;
}

}  // namespace fasteig

