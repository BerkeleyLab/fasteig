#include "BlockLanczosSolver.hpp"

#include "fasteig_problem.hpp"
#include "fasteig_logger.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace fasteig {
namespace {

std::string lower_copy(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

}  // namespace

BlockLanczosSolver::BlockLanczosSolver(::fasteig_problem& problem,
                                       const fasteig_options& options)
    : fasteig_solver(problem, options) {}

FASTEIG_INT BlockLanczosSolver::solve() {
  const operations& ops = active_ops();
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
  if (!ops.block_matvec) throw std::invalid_argument("BlockLanczosSolver: ops.block_matvec is not set.");

  const int b = static_cast<int>(blksize);
  const int max_blocks = static_cast<int>(maxiter);
  const int subspace_dim = max_blocks * b;
  if (subspace_dim < static_cast<int>(nev)) {
    throw std::invalid_argument("BlockLanczosSolver: need maxiter*blksize >= nev.");
  }

  const std::string sigma = lower_copy(which);
  if (sigma != "sr" && sigma != "lr" && sigma != "sm" && sigma != "lm" && sigma != "be") {
    throw std::invalid_argument("BlockLanczosSolver: invalid which option (use SR/LR/SM/LM/BE).");
  }

  // print_every is intentionally ignored for BlockLanczos:
  // if verbosity==2, emit a snapshot every block iteration (one block matvec).
  std::ostream* log = problem_.log_stream();
  fasteig_logger stdout_logger;
  if (verbosity >= 2) stdout_logger.attach(&std::cout);

  // Basis blocks Q_0..Q_maxiter (Q_maxiter exists to form beta for the last snapshot).
  // Use the driver's initial block when provided (options_.v0 is initialized by the driver)
  // so all solvers can start from the same vectors.
  // Fall back to random vectors only if v0 is missing or too narrow.
  MatrixXd V0;
  if (options_.v0().size() > 0) {
    const int c = std::min<int>(options_.v0().cols(), b);
    V0.resize(n, b);
    V0.leftCols(c) = options_.v0().leftCols(c);
    if (c < b) {
      V0.rightCols(b - c).setRandom();
    }
  } else {
    V0 = MatrixXd::Random(n, b);
  }
  MatrixXd Q(n, (max_blocks + 1) * b);
  MatrixXd T((max_blocks + 1) * b, (max_blocks + 1) * b);
  Q.setZero();
  T.setZero();

  MatrixXd beta(b, b);

  struct Snapshot {
    VectorXd lam;      // Ritz values (nev_i)
    VectorXd rel_res;  // relative residual estimates (nev_i)
    MatrixXd Csel;     // selected eigenvectors of T_m (m_dim x nev_i)
  };

  auto compute_snapshot = [&](int iter_blocks, const MatrixXd& beta_iter, bool need_vectors) -> Snapshot {
    const int m_dim = iter_blocks * b;
    const int nev_i = std::min(static_cast<int>(nev), m_dim);
    if (nev_i <= 0) return {};

    Eigen::SelfAdjointEigenSolver<MatrixXd> solver(T.topLeftCorner(m_dim, m_dim));
    const auto D = solver.eigenvalues();
    const auto C = solver.eigenvectors();

    std::vector<size_t> Ind(static_cast<size_t>(m_dim), 0);
    if (sigma == "be") {
      int pos = 0;
      for (int i = 0; i < m_dim / 2; ++i) {
        Ind[static_cast<size_t>(pos++)] = static_cast<size_t>(m_dim - 1 - i);
        Ind[static_cast<size_t>(pos++)] = static_cast<size_t>(i);
      }
      if (m_dim % 2 == 1) {
        Ind[static_cast<size_t>(pos)] = static_cast<size_t>(m_dim / 2);
      }
    } else {
      Ind = sort_indices(D, sigma);
    }

    VectorXd lam(nev_i);
    VectorXd rel_res(nev_i);
    MatrixXd Csel;
    if (need_vectors) Csel.resize(m_dim, nev_i);
    for (int j = 0; j < nev_i; ++j) {
      const int idx = static_cast<int>(Ind[static_cast<size_t>(j)]);
      const double lambda = D(idx);
      lam(j) = lambda;

      const VectorXd y_last = C.block(m_dim - b, idx, b, 1);
      const double abs_res = (beta_iter * y_last).norm();
      rel_res(j) = abs_res / (1.0 + std::abs(lambda));

      if (need_vectors) Csel.col(j) = C.col(idx);
    }

    Snapshot out;
    out.lam = std::move(lam);
    out.rel_res = std::move(rel_res);
    out.Csel = std::move(Csel);
    return out;
  };

  auto emit_snapshot = [&](int iter_blocks, const Snapshot& snap) {
    if (snap.lam.size() == 0) return;

    if (log) {
      problem_.logger().print_convergence_data(snap.lam, snap.rel_res, iter_blocks);
    }
    if (verbosity >= 2) {
      stdout_logger.print_convergence_data(snap.lam, snap.rel_res, iter_blocks);
    }
  };

  // Build the block Krylov basis one block at a time using KrylovSubspace.
  // After iter blocks, the current Ritz space is span{Q_0, ..., Q_{iter-1}} (dimension iter*b),
  // and beta corresponds to the coupling to Q_iter for the residual estimate.
  bool converged = false;
  int it_stop = max_blocks;
  int last_emitted_iter = 0;
  Snapshot final;
  for (int iter = 1; iter <= max_blocks; ++iter) {
    if (iter == 1) {
      KrylovSubspace::build(V0, Q, T, ops, /*begin=*/0, /*end=*/1, beta);
    } else {
      KrylovSubspace::build(V0, Q, T, ops, static_cast<size_t>(iter - 1), static_cast<size_t>(iter), beta);
    }

    // Always compute/check convergence once the subspace is large enough to represent nev.
    const bool do_check = (iter * b >= static_cast<int>(nev));
    const bool do_progress = (verbosity >= 2) || (log != nullptr);
    if (!do_check && !do_progress) continue;

    Snapshot snap = compute_snapshot(iter, beta, /*need_vectors=*/false);
    if (do_progress) {
      emit_snapshot(iter, snap);
      last_emitted_iter = iter;
    }

    if (do_check && snap.rel_res.size() > 0 && snap.rel_res.maxCoeff() < tol) {
      converged = true;
      it_stop = iter;
      final = compute_snapshot(iter, beta, /*need_vectors=*/true);
      break;
    }
  }

  if (!converged) {
    final = compute_snapshot(max_blocks, beta, /*need_vectors=*/true);
    // Ensure we have a final progress record even if verbosity==1.
    if (((log != nullptr) || verbosity >= 2) && last_emitted_iter != max_blocks) {
      emit_snapshot(max_blocks, final);
    }
  }

  const int m_dim = it_stop * b;
  eigvals_ = final.lam;
  eigvecs_.noalias() = Q.leftCols(m_dim) * final.Csel;
  compute_final_residual_norms();
  set_iterations(static_cast<long long>(it_stop));
  return status_;
}

}  // namespace fasteig
