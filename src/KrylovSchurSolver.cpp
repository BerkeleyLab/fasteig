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
namespace {

double b_norm(const operations& opsB, const VectorXd& v, VectorXd& Bv) {
  if (!opsB.matvec) throw std::invalid_argument("b_norm: opsB.matvec is not set");
  opsB.matvec(v, Bv);
  const double vv = v.dot(Bv);
  if (!(vv >= 0.0) || !std::isfinite(vv)) {
    throw std::runtime_error("b_norm: non-finite or negative B-norm encountered.");
  }
  return std::sqrt(vv);
}

// Legacy host-only expansion for generalized A*x=lambda*B*x problems. The
// standard problem uses the executor-aware core::KrylovSubspace path below.
void KrylovSubspaceGeneralized(VectorXd& f,
                               MatrixXd& Q,
                               MatrixXd& H,
                               MatrixXd& BQ,
                               const operations& opsA,
                               const operations& opsB,
                               size_t begin,
                               size_t end) {
  const size_t n = static_cast<size_t>(f.size());

  if (end == 0) throw std::runtime_error("KrylovSubspaceGeneralized: requested dim 0");
  if (end > n) throw std::runtime_error("KrylovSubspaceGeneralized: requested dim > n");
  if (!opsA.matvec) throw std::runtime_error("KrylovSubspaceGeneralized: opsA.matvec is not set");
  if (!opsB.matvec) throw std::runtime_error("KrylovSubspaceGeneralized: opsB.matvec is not set");
  if (!opsB.solve) throw std::runtime_error("KrylovSubspaceGeneralized: opsB.solve is not set");

  VectorXd v(n), Av(n), y(n), Bf(n);
  for (size_t i = begin; i < end; ++i) {
    double beta = b_norm(opsB, f, Bf);
    if (beta <= std::numeric_limits<double>::epsilon()) {
      f = VectorXd::Random(static_cast<int>(n));
      if (i > 0) {
        opsB.matvec(f, Bf);
        VectorXd s = BQ.leftCols(static_cast<int>(i)).transpose() * f;
        f.noalias() -= Q.leftCols(static_cast<int>(i)) * s;
      }
      beta = b_norm(opsB, f, Bf);
      if (beta <= std::numeric_limits<double>::epsilon()) {
        throw std::runtime_error(
            "KrylovSubspaceGeneralized: breakdown, unable to generate new basis vector");
      }
    }

    v.noalias() = f / beta;
    Q.col(static_cast<int>(i)) = v;
    BQ.col(static_cast<int>(i)).noalias() = Bf / beta;

    if (i != begin || (begin != 0 && end - begin == 1)) {
      H(static_cast<int>(i), static_cast<int>(i - 1)) = beta;
      H(static_cast<int>(i - 1), static_cast<int>(i)) = beta;
    }

    opsA.matvec(v, Av);
    opsB.solve(Av, y);

    VectorXd h = Q.leftCols(static_cast<int>(i + 1)).transpose() * Av;
    f.noalias() = y - Q.leftCols(static_cast<int>(i + 1)) * h;

    VectorXd s = BQ.leftCols(static_cast<int>(i + 1)).transpose() * f;
    h.noalias() += s;
    f.noalias() -= Q.leftCols(static_cast<int>(i + 1)) * s;

    s = BQ.leftCols(static_cast<int>(i + 1)).transpose() * f;
    h.noalias() += s;
    f.noalias() -= Q.leftCols(static_cast<int>(i + 1)) * s;

    H(static_cast<int>(i), static_cast<int>(i)) = h(static_cast<int>(i));
  }
}

}  // namespace

KrylovSchurSolver::KrylovSchurSolver(::fasteig_problem& problem,
                                     const fasteig_options& options)
    : fasteig_solver(problem, options) {}

FASTEIG_INT KrylovSchurSolver::solve() {
  if (problem_.is_generalized()) {
    return solve_generalized_host();
  }

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
      std::min(std::max<FASTEIG_INT>(2 * nev, kMinimumSubspace), n);

  fasteig_logger stdout_logger;
  if (verbosity >= 2) stdout_logger.attach(&std::cout);
  fasteig_logger* log_logger =
      (print_every > 0 && problem_.log_stream()) ? &problem_.logger() : nullptr;
  fasteig_logger* console_logger = (verbosity >= 2) ? &stdout_logger : nullptr;

  // All large vectors and the projected matrix are owned by the selected
  // executor. CPU and CUDA use the same Krylov-Schur algorithm below.
  std::shared_ptr<core::Executor> exec = problem_.executor(options_);
  std::unique_ptr<core::LinearOperator> op = problem_.make_linear_operator(options_, *exec);
  const unsigned int base_seed = options_.seed().value_or(1u);
  auto residual = (options_.v0().size() > 0)
                      ? exec->create_vector(options_.v0().col(0))
                      : exec->create_random_vector(n, base_seed);
  auto operator_result = exec->create_vector(n);
  auto basis = exec->create_multivector(n, subspace_dim);
  auto projection =
      core::make_lanczos_workspace(*exec, n, subspace_dim, subspace_dim);
  core::KrylovSubspace subspace(
      *op, *residual, *operator_result, *basis, base_seed);

  auto expand_subspace = [&](FASTEIG_INT begin) {
    for (FASTEIG_INT col = begin; col < subspace_dim; ++col) {
      const double beta = subspace.normalize_residual_into_basis(col);
      subspace.apply_operator_to_basis_column(col);
      const double alpha = subspace.orthogonalize_operator_result(col + 1);
      const double beta_next = subspace.residual_norm();
      projection->record_step(col, beta, alpha, beta_next);
    }
  };

  expand_subspace(0);

  for (FASTEIG_INT iter = 1; iter <= maxiter; ++iter) {
    // Diagonalize the projected matrix and order all Ritz pairs according to
    // `which`. CUDA uses the existing cuSOLVER workspace implementation.
    core::LanczosSnapshot full = projection->compute_snapshot(
        subspace_dim, subspace_dim, which, /*need_vectors=*/true,
        "KrylovSchurSolver");
    const VectorXd abs_residuals = core::lanczos_absolute_residuals(full);

    FASTEIG_INT nconv = 0;
    for (FASTEIG_INT j = 0; j < nev; ++j) {
      if (abs_residuals(static_cast<int>(j)) <
          tol * std::abs(full.lam(static_cast<int>(j)))) {
        ++nconv;
      }
    }

    const bool do_progress =
        (print_every > 0) && ((iter % print_every) == 0);
    const core::LanczosSnapshot report = core::krylov_schur_progress_snapshot(
        full, nev, which, shift, "KrylovSchurSolver");
    if (do_progress) {
      core::lanczos_emit_snapshot(report, iter, log_logger, console_logger);
    }

    const bool finished =
        (subspace_dim == nev) || (nconv >= nev) || (iter == maxiter);
    if (finished) {
      core::LanczosSnapshot final = core::lanczos_snapshot_head(
          full, nev, "KrylovSchurSolver");
      if (!do_progress && print_every > 0) {
        core::lanczos_emit_snapshot(report, iter, log_logger, console_logger);
      }
      projection->finalize_eigenpairs(*op, *basis, subspace_dim, final,
                                      which, shift, eigvals_, eigvecs_, resnrms_);
      if (which == "sai") compute_final_residual_norms();
      set_iterations(static_cast<long long>(iter));
      return status_;
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

FASTEIG_INT KrylovSchurSolver::solve_generalized_host() {
  const FASTEIG_INT n = eigvecs_.rows();
  const FASTEIG_INT nev = options_.nev();
  const FASTEIG_INT maxiter = options_.maxiter();
  const FASTEIG_DOUBLE tol = options_.tol();
  const int verbosity = options_.verbosity();
  const FASTEIG_INT print_every = options_.print_every();
  const std::string which = core::lanczos_lower_copy(options_.which());

  if (n <= 0 || nev <= 0 || nev > n || maxiter <= 0) {
    throw std::invalid_argument("KrylovSchurSolver: invalid generalized problem dimensions/options.");
  }
  core::validate_lanczos_which(which, /*allow_sai=*/false, "KrylovSchurSolver");

  constexpr FASTEIG_INT kMinimumSubspace = 20;
  const FASTEIG_INT k =
      std::min(std::max<FASTEIG_INT>(2 * nev, kMinimumSubspace), n);

  fasteig_logger stdout_logger;
  if (verbosity >= 2) stdout_logger.attach(&std::cout);
  fasteig_logger* log_logger =
      (print_every > 0 && problem_.log_stream()) ? &problem_.logger() : nullptr;
  fasteig_logger* console_logger = (verbosity >= 2) ? &stdout_logger : nullptr;

  MatrixXd Q = MatrixXd::Zero(n, k);
  MatrixXd H = MatrixXd::Zero(k, k);
  MatrixXd BQ = MatrixXd::Zero(n, k);
  VectorXd f;
  if (options_.v0().size() > 0) {
    f = options_.v0().col(0);
  } else {
    f = VectorXd::Random(n);
  }

  KrylovSubspaceGeneralized(f, Q, H, BQ, opsA_, opsB_, 0,
                            static_cast<size_t>(k));

  for (FASTEIG_INT iter = 1; iter <= maxiter; ++iter) {
    Eigen::SelfAdjointEigenSolver<MatrixXd> solver(H.topLeftCorner(k, k));
    const VectorXd D = solver.eigenvalues();
    const MatrixXd C = solver.eigenvectors();

    VectorXd Bf(n);
    const double beta = b_norm(opsB_, f, Bf);
    VectorXd brow = beta * C.row(k - 1).transpose();
    const std::vector<size_t> indices = core::lanczos_select_ritz_indices(D, which);

    VectorXd values(nev);
    VectorXd residuals(nev);
    FASTEIG_INT nconv = 0;
    for (FASTEIG_INT j = 0; j < nev; ++j) {
      const int idx = static_cast<int>(indices[static_cast<size_t>(j)]);
      values(static_cast<int>(j)) = D(idx);
      residuals(static_cast<int>(j)) = std::abs(brow(idx));
      if (residuals(static_cast<int>(j)) < tol * std::abs(D(idx))) ++nconv;
    }

    const bool do_progress =
        (print_every > 0) && ((iter % print_every) == 0);
    if (do_progress) {
      core::LanczosSnapshot report;
      report.lam = values;
      report.rel_res = residuals;
      core::lanczos_emit_snapshot(report, iter, log_logger, console_logger);
    }

    if (k == nev || nconv >= nev || iter == maxiter) {
      eigvals_ = values;
      MatrixXd selected(k, nev);
      for (FASTEIG_INT j = 0; j < nev; ++j) {
        selected.col(static_cast<int>(j)) =
            C.col(static_cast<int>(indices[static_cast<size_t>(j)]));
      }
      eigvecs_.noalias() = Q.leftCols(k) * selected;
      if (!do_progress && print_every > 0) {
        core::LanczosSnapshot report;
        report.lam = values;
        report.rel_res = residuals;
        core::lanczos_emit_snapshot(report, iter, log_logger, console_logger);
      }
      compute_final_residual_norms();
      set_iterations(static_cast<long long>(iter));
      return status_;
    }

    const FASTEIG_INT spare = (k - nev) / 2;
    FASTEIG_INT retained = nev + std::min(nconv, spare);
    if (retained == 1 && k > 3) retained = k / 2;

    MatrixXd selected(k, retained);
    for (FASTEIG_INT j = 0; j < retained; ++j) {
      selected.col(static_cast<int>(j)) =
          C.col(static_cast<int>(indices[static_cast<size_t>(j)]));
    }
    MatrixXd rotated_basis = Q.leftCols(k) * selected;
    Q.leftCols(retained) = std::move(rotated_basis);

    H.setZero();
    for (FASTEIG_INT j = 0; j < retained; ++j) {
      const int idx = static_cast<int>(indices[static_cast<size_t>(j)]);
      H(static_cast<int>(j), static_cast<int>(j)) = D(idx);
      H(static_cast<int>(retained), static_cast<int>(j)) = brow(idx);
      H(static_cast<int>(j), static_cast<int>(retained)) = brow(idx);
    }

    MatrixXd BQtmp(n, retained);
    if (!opsB_.block_matvec) {
      throw std::runtime_error("KrylovSchurSolver: generalized requires opsB.block_matvec.");
    }
    opsB_.block_matvec(Q.leftCols(retained), BQtmp);
    BQ.leftCols(retained) = BQtmp;
    KrylovSubspaceGeneralized(f, Q, H, BQ, opsA_, opsB_,
                              static_cast<size_t>(retained),
                              static_cast<size_t>(k));
  }

  throw std::logic_error("KrylovSchurSolver: unreachable generalized solve termination.");
}

}  // namespace fasteig

