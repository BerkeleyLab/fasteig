#include "BlockKrylovSchurSolver.hpp"

#include "KrylovSubspace.hpp"
#include "fasteig_logger.hpp"
#include "fasteig_problem.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace fasteig {
namespace {

std::string lower_copy(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

std::vector<size_t> selected_indices(const VectorXd& values,
                                     const std::string& which) {
  const size_t n = static_cast<size_t>(values.size());
  if (which != "be") return sort_indices(values, which);

  std::vector<size_t> indices(n);
  size_t pos = 0;
  for (size_t j = 0; j < n / 2; ++j) {
    indices[pos++] = n - 1 - j;
    indices[pos++] = j;
  }
  if (n % 2 != 0) indices[pos] = n / 2;
  return indices;
}

}  // namespace

BlockKrylovSchurSolver::BlockKrylovSchurSolver(
    ::fasteig_problem& problem,
    const fasteig_options& options)
    : fasteig_solver(problem, options) {}

FASTEIG_INT BlockKrylovSchurSolver::solve() {
  const operations& ops = active_ops();
  const int n = static_cast<int>(eigvecs_.rows());
  const int nev = static_cast<int>(options_.nev());
  const int maxiter = static_cast<int>(options_.maxiter());
  const double tol = options_.tol();
  const int verbosity = options_.verbosity();
  const int print_every = static_cast<int>(options_.print_every());
  const std::string which = lower_copy(options_.which());

  if (n <= 0) throw std::invalid_argument("BlockKrylovSchurSolver: n must be > 0.");
  if (nev <= 0 || nev > n) {
    throw std::invalid_argument("BlockKrylovSchurSolver: nev must be in [1,n].");
  }
  if (maxiter <= 0) {
    throw std::invalid_argument("BlockKrylovSchurSolver: maxiter must be > 0.");
  }
  if (!ops.block_matvec) {
    throw std::invalid_argument(
        "BlockKrylovSchurSolver: ops.block_matvec is not set.");
  }
  if (problem_.bmatrix() != nullptr) {
    throw std::invalid_argument(
        "BlockKrylovSchurSolver: generalized eigenproblems are not supported.");
  }
  if (which != "sr" && which != "lr" && which != "sm" &&
      which != "lm" && which != "be") {
    throw std::invalid_argument(
        "BlockKrylovSchurSolver: invalid which option (use SR/LR/SM/LM/BE).");
  }

  // bks.m uses the number of requested eigenpairs as its block width.  Its
  // nblks argument therefore corresponds to maxdim / nev in this interface.
  const int requested_dim =
      (options_.maxdim() > 0)
          ? static_cast<int>(options_.maxdim())
          : std::max(2 * nev, 20);
  if (requested_dim < 2 * nev) {
    throw std::invalid_argument(
        "BlockKrylovSchurSolver: maxdim must be at least 2*nev.");
  }

  const int block_count = std::min(requested_dim / nev, n / nev);
  if (block_count < 2) {
    throw std::invalid_argument(
        "BlockKrylovSchurSolver: matrix dimension must accommodate at least "
        "two nev-sized blocks.");
  }
  const int cycle_dim = block_count * nev;

  MatrixXd V0(n, nev);
  if (options_.v0().size() > 0) {
    if (options_.v0().rows() != n || options_.v0().cols() < nev) {
      throw std::invalid_argument(
          "BlockKrylovSchurSolver: v0 must have at least nev columns and n rows.");
    }
    V0 = options_.v0().leftCols(nev);
  } else {
    V0.setRandom();
  }

  // One extra basis block holds the factor Q_F in F = Q_F*beta.  This is the
  // F returned by blocklan/expand_bklan in the MATLAB implementation.
  MatrixXd basis(n, cycle_dim + nev);
  MatrixXd projected(cycle_dim + nev, cycle_dim + nev);
  MatrixXd beta(nev, nev);
  basis.setZero();
  projected.setZero();
  beta.setZero();

  KrylovSubspace initial(V0, basis, projected, beta);
  initial.build_block(ops, 0, static_cast<size_t>(block_count));

  std::ostream* log = (print_every > 0) ? problem_.log_stream() : nullptr;
  fasteig_logger stdout_logger;
  if (verbosity >= 2) stdout_logger.attach(&std::cout);

  VectorXd final_values(nev);
  MatrixXd final_vectors(n, nev);
  VectorXd final_residuals(nev);
  int final_iteration = 0;
  bool last_was_printed = false;

  for (int iteration = 1; iteration <= maxiter; ++iteration) {
    projected.topLeftCorner(cycle_dim, cycle_dim) =
        0.5 * (projected.topLeftCorner(cycle_dim, cycle_dim) +
               projected.topLeftCorner(cycle_dim, cycle_dim).transpose().eval());

    Eigen::SelfAdjointEigenSolver<MatrixXd> eigensolver(
        projected.topLeftCorner(cycle_dim, cycle_dim));
    if (eigensolver.info() != Eigen::Success) {
      throw std::runtime_error(
          "BlockKrylovSchurSolver: projected eigensolve failed.");
    }

    const VectorXd& all_values = eigensolver.eigenvalues();
    const MatrixXd& all_vectors = eigensolver.eigenvectors();
    const std::vector<size_t> indices = selected_indices(all_values, which);

    MatrixXd selected(cycle_dim, nev);
    for (int j = 0; j < nev; ++j) {
      const int index = static_cast<int>(indices[static_cast<size_t>(j)]);
      final_values(j) = all_values(index);
      selected.col(j) = all_vectors.col(index);
    }

    final_vectors.noalias() = basis.leftCols(cycle_dim) * selected;
    const MatrixXd coupling = beta * selected.bottomRows(nev);
    for (int j = 0; j < nev; ++j) {
      final_residuals(j) =
          coupling.col(j).norm() / (1.0 + std::abs(final_values(j)));
    }
    final_iteration = iteration;

    const bool do_progress =
        print_every > 0 && (iteration % print_every == 0);
    if (do_progress) {
      if (log) {
        problem_.logger().print_convergence_data(
            final_values, final_residuals, iteration);
      }
      if (verbosity >= 2) {
        stdout_logger.print_convergence_data(
            final_values, final_residuals, iteration);
      }
    }
    last_was_printed = do_progress;

    const bool converged = final_residuals.maxCoeff() < tol;
    if (converged || iteration == maxiter) break;

    // Krylov--Schur restart from bks.m:
    //   V_1 = V*C, V_2 = qr(F), T(1:2p,1:p) = [D; beta*C_last].
    // KrylovSubspace then performs the expand_bklan portion of the cycle.
    const MatrixXd residual_basis = basis.middleCols(cycle_dim, nev);
    basis.setZero();
    basis.leftCols(nev) = final_vectors;
    basis.middleCols(nev, nev) = residual_basis;

    projected.setZero();
    projected.topLeftCorner(nev, nev) = final_values.asDiagonal();
    projected.block(nev, 0, nev, nev) = coupling;
    projected.block(0, nev, nev, nev) = coupling.transpose();

    KrylovSubspace expansion(V0, basis, projected, beta);
    expansion.build_block(ops, 1, static_cast<size_t>(block_count));
  }

  eigvals_ = final_values;
  eigvecs_ = final_vectors;

  if (print_every > 0 && !last_was_printed) {
    if (log) {
      problem_.logger().print_convergence_data(
          final_values, final_residuals, final_iteration);
    }
    if (verbosity >= 2) {
      stdout_logger.print_convergence_data(
          final_values, final_residuals, final_iteration);
    }
  }

  compute_final_residual_norms();
  set_iterations(final_iteration);
  return status_;
}

}  // namespace fasteig
