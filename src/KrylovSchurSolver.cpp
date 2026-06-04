#include "KrylovSchurSolver.hpp"

#include "fasteig_problem.hpp"
#include "print_util.hpp"

#include <cmath>
#include <limits>

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

// Generalized (A,B) Lanczos-style expansion for B-inner product.
// Builds a B-orthonormal basis Q (Q^T B Q = I) and the projected matrix H = Q^T A Q.
//
// Uses operator application y = B^{-1} (A v) via opsB.solve.
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

  double beta = 0.0;
  VectorXd v(n), Av(n), y(n), Bf(n);

  for (size_t i = begin; i < end; ++i) {
    beta = b_norm(opsB, f, Bf);

    // Handle breakdown: if f is (near) zero in B-norm, restart with a random vector.
    if (beta <= std::numeric_limits<double>::epsilon()) {
      f = VectorXd::Random(static_cast<int>(n));
      if (i > 0) {
        // f -= Q*(Q^T B f) using cached BQ
        opsB.matvec(f, Bf);
        VectorXd s = BQ.leftCols(static_cast<int>(i)).transpose() * f;
        f.noalias() -= Q.leftCols(static_cast<int>(i)) * s;
      }
      beta = b_norm(opsB, f, Bf);
      if (beta <= std::numeric_limits<double>::epsilon()) {
        throw std::runtime_error("KrylovSubspaceGeneralized: breakdown, unable to generate new basis vector");
      }
    }

    v.noalias() = f / beta;
    Q.col(static_cast<int>(i)) = v;
    BQ.col(static_cast<int>(i)).noalias() = Bf / beta;

    // Similar to KrylovSubspace(): fill the off-diagonal coupling coefficient.
    if (i != begin || (begin != 0 && end - begin == 1)) {
      H(static_cast<int>(i), static_cast<int>(i - 1)) = beta;
      H(static_cast<int>(i - 1), static_cast<int>(i)) = beta;
    }

    // Apply A then solve with B: y = B^{-1} (A v).
    opsA.matvec(v, Av);
    opsB.solve(Av, y);

    // Projected coefficients h_j = q_j^T A v = q_j^T B y.
    VectorXd h = Q.leftCols(static_cast<int>(i + 1)).transpose() * Av;
    f.noalias() = y - Q.leftCols(static_cast<int>(i + 1)) * h;

    // Re-orthogonalize in the B-inner product: s = Q^T B f = (BQ^T) f.
    VectorXd s = BQ.leftCols(static_cast<int>(i + 1)).transpose() * f;
    h.noalias() += s;
    f.noalias() -= Q.leftCols(static_cast<int>(i + 1)) * s;

    // Second pass for stability.
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
  const FASTEIG_INT n = eigvecs_.rows();
  const FASTEIG_INT nev = options_.nev();
  const FASTEIG_INT maxiter = options_.maxiter();
  const std::string which = options_.which();
  const FASTEIG_DOUBLE shift = options_.shift();
  const FASTEIG_DOUBLE tol = options_.tol();
  const int verbosity = options_.verbosity();
  const FASTEIG_INT print_every = options_.print_every();

  VectorXd v0;
  if (options_.v0().size() > 0) {
    v0 = options_.v0().col(0);
  } else {
    v0 = VectorXd::Random(n);
  }

  const size_t n_sz = static_cast<size_t>(v0.size());

  if (nev == 0) {
    throw std::runtime_error("Desired number of eigenpairs should be larger than 0");
  }
  if (static_cast<size_t>(nev) > n_sz) {
    throw std::runtime_error("Desired number of eigenpairs should be smaller than the size of the matrix");
  }

  const size_t upp = 20;
  const size_t k = std::min(std::max(2 * static_cast<size_t>(nev), upp), n_sz);
  const size_t nev0 = static_cast<size_t>(nev);

  std::vector<size_t> Ind(k, 0);

  if (which != "SR" && which != "sr" &&
      which != "LR" && which != "lr" &&
      which != "BE" && which != "be" &&
      which != "LM" && which != "lm" &&
      which != "SM" && which != "sm" &&
      which != "SAI" && which != "sai") {
    throw std::invalid_argument(
        "Invalid which option. Use 'sr' for smallest, 'lr' for largest, 'be' for both ends, "
        "'lm' for largest (magnitude), or 'sm' for smallest (magnitude) eigenpairs.");
  }

  std::string which_lower = which;
  std::transform(which_lower.begin(), which_lower.end(), which_lower.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

  std::ostream* log = (print_every > 0) ? problem_.log_stream() : nullptr;

  MatrixXd Q(n_sz, static_cast<int>(k));
  MatrixXd H(static_cast<int>(k), static_cast<int>(k));
  MatrixXd BQ;  // only used for generalized problems

  Q.setZero();
  H.setZero();

  FASTEIG_DOUBLE beta;
  VectorXd f = v0;
  FASTEIG_SIZET sai = 0;

  const bool generalized = (problem_.bmatrix() != nullptr);
  if (generalized && (which_lower == "sai")) {
    throw std::invalid_argument("KrylovSchur: generalized eigenproblems do not support which=SAI.");
  }

  if (which_lower == "sai") {
    KrylovSubspace(f, Q, H, opsSAI_, 0, k);
    sai = 1;
  } else if (!generalized) {
    KrylovSubspace(f, Q, H, opsA_, 0, k);
  } else {
    BQ.resize(static_cast<int>(n_sz), static_cast<int>(k));
    BQ.setZero();
    KrylovSubspaceGeneralized(f, Q, H, BQ, opsA_, opsB_, 0, k);
  }

  // Parameters currently unused by this solver class interface (kept for algorithm parity).
  

  long long iter_done = 0;
  for (size_t i = 0; i < static_cast<size_t>(maxiter); ++i) {
    iter_done = static_cast<long long>(i + 1);
    Eigen::SelfAdjointEigenSolver<MatrixXd> solver(
        H.block(0, 0, static_cast<int>(k), static_cast<int>(k)));
    const auto D = solver.eigenvalues();
    const auto C = solver.eigenvectors();

    beta = f.norm();
    if (generalized) {
      VectorXd Bf(static_cast<int>(n_sz));
      beta = b_norm(opsB_, f, Bf);
    }

    std::vector<FASTEIG_DOUBLE> brow(k);
    for (size_t j = 0; j < k; ++j) {
      brow[j] = beta * C(static_cast<int>(k - 1), static_cast<int>(j));
    }

    if (which_lower == "be") {
      for (size_t j = 0; j < k / 2; ++j) {
        Ind[2 * j] = k - 1 - j;
        Ind[2 * j + 1] = j;
      }
    } else {
      Ind = sort_indices(D, which_lower);
    }

    size_t nconv = 0;

    VectorXd lam_iter(static_cast<int>(nev0));
    VectorXd res_iter(static_cast<int>(nev0));
    for (size_t j = 0; j < nev0; ++j) {
      const auto resid = abs(brow[Ind[j]]);
      const auto eval = (sai == 1) ? (shift + 1.0 / D(static_cast<int>(Ind[j])))
                                   : D(static_cast<int>(Ind[j]));
      lam_iter(static_cast<int>(j)) = eval;
      res_iter(static_cast<int>(j)) = resid;
      if (resid < tol * abs(D(static_cast<int>(Ind[j])))) {
        ++nconv;
      }
    }

    const bool do_progress = (print_every > 0) && (((i + 1) % static_cast<size_t>(print_every)) == 0);
    if (do_progress) {
      if (log) {
        print_convergence_data(2, *log, lam_iter, res_iter, static_cast<int>(i + 1));
      }
      if (verbosity >= 2) {
        print_convergence_data(1, std::cout, lam_iter, res_iter, static_cast<int>(i + 1));
      }
    }

    if (k == nev0) {
      eigvals_ = lam_iter;
      for (size_t j = 0; j < nev0; ++j) {
        eigvecs_.col(static_cast<int>(j)).noalias() =
            Q.block(0, 0, n_sz, static_cast<int>(k)) *
            C.block(0, static_cast<int>(Ind[j]), static_cast<int>(k), 1);
      }

      if (print_every > 0 && !do_progress) {
        if (log) {
          print_convergence_data(2, *log, lam_iter, res_iter, static_cast<int>(i + 1));
        }
        if (verbosity >= 2) {
          print_convergence_data(1, std::cout, lam_iter, res_iter, static_cast<int>(i + 1));
        }
      }
      compute_final_residual_norms();
      set_iterations(iter_done);
      return status_;
    }

    const size_t temp = static_cast<size_t>(std::floor((k - nev0) / 2));
    size_t current_nev = nev0 + std::min(nconv, temp);
    if (current_nev == 1 && k > 3) {
      current_nev = static_cast<size_t>(std::floor(k / 2.0));
    }

    MatrixXd C_TEMP(static_cast<int>(k), static_cast<int>(current_nev));
    for (size_t j = 0; j < current_nev; ++j) {
      C_TEMP.block(0, static_cast<int>(j), static_cast<int>(k), 1) =
          C.block(0, static_cast<int>(Ind[j]), static_cast<int>(k), 1);
    }

    Q.block(0, 0, n_sz, static_cast<int>(current_nev)) =
        Q.block(0, 0, n_sz, static_cast<int>(k)) * C_TEMP;

    H.setZero();

    for (size_t j = 0; j < current_nev; ++j) {
      H(static_cast<int>(j), static_cast<int>(j)) = D(static_cast<int>(Ind[j]));
      H(static_cast<int>(current_nev), static_cast<int>(j)) = brow[Ind[j]];
      H(static_cast<int>(j), static_cast<int>(current_nev)) = brow[Ind[j]];
    }

    if (nconv >= nev0 || i == static_cast<size_t>(maxiter) - 1) {

      for (size_t j = 0; j < nev0; ++j) {
        eigvals_(static_cast<int>(j)) = (sai == 1)
                                            ? (shift + 1.0 / H(static_cast<int>(j), static_cast<int>(j)))
                                            : H(static_cast<int>(j), static_cast<int>(j));
      }
      eigvecs_ = Q.block(0, 0, n_sz, static_cast<int>(nev0));

      std::iota(Ind.begin(), Ind.end(), 0); // natural ordering for Ritz values

      if (sai == 1) {

        VectorXd residual(n_sz);

        for (size_t j = 0; j < nev0; ++j) {
          opsA_.matvec(eigvecs_.col(static_cast<int>(j)), residual);
          residual -= eigvals_(static_cast<int>(j)) * eigvecs_.col(static_cast<int>(j));
          res_iter(static_cast<int>(j)) = residual.norm();
        }
        Ind = sort_indices(eigvals_, "sr"); // resort the transformed Ritz values for reporting convergence
      }

      VectorXd lam_final(static_cast<int>(nev0));
      VectorXd res_final(static_cast<int>(nev0));

      for (size_t j = 0; j < nev0; ++j) {
          lam_final(static_cast<int>(j)) = eigvals_(static_cast<int>(Ind[j]));
          res_final(static_cast<int>(j)) = res_iter(static_cast<int>(Ind[j]));
      }
      if (print_every > 0 && !do_progress) {
        if (log) {
          print_convergence_data(2, *log, lam_final, res_final, static_cast<int>(i + 1));
        }
        if (verbosity >= 2) {
          print_convergence_data(1, std::cout, lam_final, res_final, static_cast<int>(i + 1));
        }
      }
      for (size_t j = 0; j < nev0; ++j) {
          eigvals_(static_cast<int>(j)) = lam_final(static_cast<int>(j));
          eigvecs_.col(static_cast<int>(j)) = Q.block(0, 0, n_sz, static_cast<int>(nev0)).col(static_cast<int>(Ind[j]));
      }
      compute_final_residual_norms();
      set_iterations(iter_done);
      return status_;
    }

    if (which_lower == "sai") {
      KrylovSubspace(f, Q, H, opsSAI_, current_nev, k);
    } else if (!generalized) {
      KrylovSubspace(f, Q, H, opsA_, current_nev, k);
    } else {
      // Update cached BQ for the restarted basis before expansion.
      if (!opsB_.block_matvec) {
        throw std::runtime_error("KrylovSchur: generalized requires opsB.block_matvec.");
      }
      MatrixXd BQtmp(static_cast<int>(n_sz), static_cast<int>(current_nev));
      opsB_.block_matvec(Q.leftCols(static_cast<int>(current_nev)), BQtmp);
      BQ.leftCols(static_cast<int>(current_nev)) = std::move(BQtmp);
      KrylovSubspaceGeneralized(f, Q, H, BQ, opsA_, opsB_, current_nev, k);
    }
  }
  set_iterations(iter_done);
  return status_;
}

}  // namespace fasteig

