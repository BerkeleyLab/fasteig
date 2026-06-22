#include "LOBPCGSolver.hpp"

#include "fasteig.hpp"
#include "fasteig_problem.hpp"
#include "fasteig_logger.hpp"

#include <algorithm>

namespace fasteig {

using namespace Eigen;
using namespace std;

LOBPCGSolver::LOBPCGSolver(::fasteig_problem& problem,
                           const fasteig_options& options)
    : fasteig_solver(problem, options),
      lam(eigvals_),
      V(eigvecs_) {}

void LOBPCGSolver::reset_state(double ctol,
                               double otol,
                               int vb,
                               int pe,
                               std::ostream* logstream) {
  conv_tol = ctol;
  orth_tol = otol;
  verbose = vb;
  print_every = pe;

  iter = -1;
  b_conv = 0;
  use_orth = false;

  const int n_i = V.rows();
  const int b_i = V.cols();
  n = static_cast<size_t>(n_i);
  b = static_cast<size_t>(b_i);

  XP = MatrixXd(n_i, 2 * b_i);
  AXP = MatrixXd(n_i, 2 * b_i);
  W = MatrixXd(n_i, b_i);
  AW = MatrixXd(n_i, b_i);

  StAS = MatrixXd(3 * b_i, 3 * b_i);
  StS = MatrixXd(3 * b_i, 3 * b_i);
  C = MatrixXd(3 * b_i, 2 * b_i);

  resnorms = VectorXd(b_i);
  eigsolver = Eigen::SelfAdjointEigenSolver<MatrixXd>(3 * b_i);

  log = logstream;
  logging = (print_every > 0) && (log != nullptr);
}

FASTEIG_INT LOBPCGSolver::solve() {
  // Respect which: for LR/LM we run on -A (same eigenvectors) and negate eigenvalues back.
  auto lower_copy = [](std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
  };
  const std::string which_l = lower_copy(options_.which());
  const bool flip_sign = (which_l == "lr" || which_l == "lm");

  ops_ = active_ops();
  if (flip_sign) {
    const auto orig_mv = ops_.matvec;
    const auto orig_bmv = ops_.block_matvec;
    if (orig_mv) {
      ops_.matvec = [orig_mv](const VectorXd& v, VectorXd& av) {
        orig_mv(v, av);
        av *= -1.0;
      };
    }
    if (orig_bmv) {
      ops_.block_matvec = [orig_bmv](const MatrixXd& V, MatrixXd& AV) {
        orig_bmv(V, AV);
        AV *= -1.0;
      };
    }
  }

  const FASTEIG_INT nev = options_.nev();
  const FASTEIG_INT maxiter = options_.maxiter();
  const FASTEIG_DOUBLE tol = options_.tol();
  const FASTEIG_DOUBLE orth_tol = 1.0e-10;
  const int verbosity = options_.verbosity();
  const FASTEIG_INT print_every = options_.print_every();

  std::ostream* log = problem_.log_stream();
  reset_state(tol, orth_tol, verbosity, print_every, log);
  fasteig_logger stdout_logger;
  if (verbose >= 2) stdout_logger.attach(&std::cout);

  for (int i = 0; i < maxiter; ++i) {
      /*
      -----------------------------------------------------------------
          Initialize the X and P subspaces if they haven't
          already been given values. This would be indicated
          by the condition 'iter == -1'
      -----------------------------------------------------------------
      */
      if (iter == -1)  {
          if (log) problem_.logger().print_progress_note("Initializing subspaces", -1);
          if (verbose >= 2) stdout_logger.print_progress_note("Initializing subspaces", -1);
          this->initialize_XP_subspaces();
          iter++;
      }

      int k = iter;

      /*
      -----------------------------------------------------------------
          Preconditioning and residual orthogonalization
      -----------------------------------------------------------------
      */

      // Apply custom preconditioner to the residual if one is defined in ops
  //Chao
  //    if (ops.precond) {
  //        ops.precond(W);
  //    // Otherwise, default to a diagonal preconditoned
  //    } else {
  //        // TODO!
  //    }

      // Orthogonalize W against [X P]
      if (use_orth)       ortho(W, XP, orth_tol);

      // Update matrix-vector product A*W
      ops_.block_matvec(W, AW);

      /*
      -----------------------------------------------------------------
          RR Step
      -----------------------------------------------------------------
      */

      // Attempt RR step without orthogonalization
      bool needs_orth = this->modified_rayleigh_ritz();


      // If the RR step failed due to poor conditioning of S'S, orthogonalize W against [X P]
      if (!use_orth && needs_orth) {
          // Orthogonalizing W against [X P]
          string orth_message = "Orthogonalizing W";
          if (log) problem_.logger().print_progress_note(orth_message, k+1);
          if (verbose >= 2) stdout_logger.print_progress_note(orth_message, k+1);
          ortho(W, XP, orth_tol);
          // Recompute AW
          ops_.block_matvec(W, AW);
          // Re-run RR step
          this->modified_rayleigh_ritz();
          // Set use_orth = true so we orthogonalize W against [X P]
          // from here on out. The condition number of S generally grows
          // monotonically, so once it exceeds the threshold for the first
          // time, it will likely remain above the threshold thereafter.
          use_orth = true;
      }

      // Updated residual vectors
      // Recall that after locking, W_k only stores the residual vectors of
      // *unconverged* vectors. This is why we need to take a block of X, AX, and lam,
      // which store both the converged and unconverged vectors.
      auto lam_unconv = lam.tail(b-b_conv);
      auto X_unconv = XP.middleCols(b_conv,b-b_conv);
      auto AX_unconv = AXP.middleCols(b_conv,b-b_conv);
      W.noalias()  = AX_unconv - X_unconv * lam_unconv.asDiagonal();

      /*
      -----------------------------------------------------------------
          Post Processing
      -----------------------------------------------------------------
      */
      // Compute residual norms (used for convergence + optional progress output).
      for (int j = 0; j < b; j++) {
          const double num = (AXP.col(j) - lam(j) * XP.col(j)).norm();
          const double denom = (1.0 + abs(lam(j))) * XP.col(j).norm();
          resnorms(j) = (denom > 0.0) ? (num / denom) : num;
      }

      // Convergence progress update
      const bool do_progress = (print_every > 0) && (((k + 1) % print_every) == 0);
      if (do_progress) {
          // Always write progress to file if a log stream is provided.
          if (log) {
              problem_.logger().print_convergence_data(lam, resnorms, k + 1);
          }
          // Optionally print progress to stdout based on verbosity.
          if (verbose >= 2) {
              stdout_logger.print_convergence_data(lam, resnorms, k + 1);
          }
      }

      // Expose current eigenvector estimates to the caller.
      V.noalias() = XP.leftCols(b);

      // Update iteration counter
      iter++;

      // Stop as soon as all requested eigenpairs are converged.
      const int nev_i = std::min(static_cast<int>(nev), static_cast<int>(b));
      if (nev_i > 0 && resnorms.head(nev_i).maxCoeff() < conv_tol) break;
  }
  set_iterations(iter < 0 ? 0 : static_cast<long long>(iter));
  if (flip_sign) {
    lam *= -1.0;
  }
  compute_final_residual_norms();
  return status_;
}

/*
 * Assigns the initial values to the X and P subspaces.
 * - The Rayleigh Ritz step is applied to the initial X to
 *   orthogonalize it and get an initial set of Ritz values
 * - The P subspace is filled with random Gaussian vectors.
 *   This makes the code more uniform, as we don't need to
 *   treat the first iteration separately, and these random
 *   vectors can only help with convergence.
 */
void LOBPCGSolver::initialize_XP_subspaces() {

    // Block of S and AS storing the eigenvalue estimates
    auto X  =  XP.leftCols(b);
    auto AX = AXP.leftCols(b);

    // Copy V to X
    X.noalias() = V;

    // Initial matrix-vector product A*X
    // Note:    Putting the result in AW initially, as this has type
    //          MatrixXd, whereas AX as defined above has type Block<...>.
    //          block_matvec() expected MatrixXd variables as arguments.
    //          Nothing is stored in AW currently, and it will be overwritten
    //          before we need to use it again.
    ops_.block_matvec(X, AW);
    AX.noalias() = AW;

    // Initial orthogonalization via RR step
    // See modified_rayleigh_ritz() function below for more
    // details of what's happening here.
    // Overlap matrices
    MatrixXd XtX = X.transpose() * X;
    MatrixXd XtAX = X.transpose() * AX;
    // Cholesky decomp of XtX to convert to standard eigenproblem
    LLT<Ref<MatrixXd>,Lower> chol(XtX);
    auto Rt = XtX.triangularView<Lower>();
    Rt.solveInPlace(XtAX);
    Rt.solveInPlace(XtAX.transpose());
    // Solve projected eigenproblem
    eigsolver.compute(XtAX);
    auto C = eigsolver.eigenvectors();
    // Convert back to original system
    Rt.transpose().solveInPlace(C);

    // Initial eigenvalue and eigenvector estimates
    lam.noalias() = eigsolver.eigenvalues();
    X.applyOnTheRight(C);
    AX.applyOnTheRight(C);

    // Initial residual W
    W.noalias() = AX - X * lam.asDiagonal();

    // Initial P --- fill with random unit vectors
    // This makes the subsequent code simpler and easier to read,
    // and these random vectors can only help with convergence.
    XP.rightCols(b).setRandom();
    for (int i = 0; i < b; i++) {
        XP.col(b+i) /= XP.col(b+i).norm();
    }
    // Initial AP
    auto P = XP.rightCols(b);
    ops_.block_matvec( P, AW );
    AXP.rightCols(b).noalias() = AW;

}

/*
 * Locks eigenvectors that have converged during LOBPCG iteration.
 *
 */
void LOBPCGSolver::lock_eigenvectors() {

    int lock = 0;

    // Count the number of new eigenpairs to lock (may be multiple)
    for (int i = 0; i < b - b_conv; i++) {

        int j = i + b_conv;

        // Compute residual norm |Ax_j / lam_j - x_j | / |x_j|
        double num = (AXP.col(j) - lam(j)*XP.col(j)).norm();
        double denom = (1+abs(lam(j)))*XP.col(j).norm();
        resnorms(j) = num/denom;

        // Check for convergence
        if (resnorms(j) < conv_tol) {
            lock++;
        }

        // Break otherwise, as we lock eigenpairs in order
        else {
            break;
        }
    }

    if (lock > 0) {

        // Progress note about locking eigenvectors
        for (int l = 0; l < lock; l++) {
            if (log) problem_.logger().print_locking_note(b_conv+1+l, iter+1);
            if (verbose >= 2) {
                fasteig_logger stdout_logger;
                stdout_logger.attach(&std::cout);
                stdout_logger.print_locking_note(b_conv+1+l, iter+1);
            }
        }

        // Discard columns of P and W corresponding to locked eigenpairs.
        // Only needed if there are still unconverged vectors remaining.
        if (b_conv + lock < b) {
            // Number of unconverged eigenvectors remaining before/after locking update
            int u_pre_lock = b - b_conv;
            int u_post_lock = u_pre_lock - lock;

            // Shifting P & W in memory
            //  Before: S = [ X=S(:,1:b) P(:,:lock) P(:,lock:) W(:,:lock) W(:,lock:) ]
            //   After: S = [ X=S(:,1:b) P(:,lock:) W(:,lock:) ]
            // Shift P/AP (keep only unconverged columns at the end).
            // Note the use of .eval() due to aliasing.
            XP.rightCols(u_post_lock) = XP.rightCols(u_pre_lock).rightCols(u_post_lock).eval();
            AXP.rightCols(u_post_lock) = AXP.rightCols(u_pre_lock).rightCols(u_post_lock).eval();
            // Shift W & AW
            W  = W.rightCols(u_post_lock).eval();
            AW = AW.rightCols(u_post_lock).eval();
        }

        // Officially update b_conv
        b_conv += lock;
    }
}

void LOBPCGSolver::svqb(MatrixXd& U, double tau, char method) {
    MatrixXd overlap = U.transpose() * U;

    VectorXd d = overlap.diagonal();
    d = d.array().sqrt().inverse();
    const auto D = d.asDiagonal();
    overlap.applyOnTheLeft(D);
    overlap.applyOnTheRight(D);

    eigsolver.compute(overlap);
    VectorXd theta = eigsolver.eigenvalues();
    const MatrixXd Z = eigsolver.eigenvectors();

    const double threshold = tau * theta.lpNorm<Infinity>();
    const double newval = (method == 'R') ? threshold : 0.0;
    theta = (theta.array() < threshold).select(newval, theta);
    theta = theta.array().sqrt().inverse();
    theta = theta.array().isInf().select(0, theta);

    U.applyOnTheRight(D);
    U.applyOnTheRight(Z);
    U.applyOnTheRight(theta.asDiagonal());
}

void LOBPCGSolver::ortho(MatrixXd& U,
                         const MatrixXd& V,
                         double tol,
                         int maxiter1,
                         int maxiter2,
                         double tau) {
    const int block_size = U.cols();
    const MatrixXd identity = MatrixXd::Identity(block_size, block_size);

    double normU = U.norm();
    for (int i = 0; i < maxiter1; ++i) {
        U = U - V * (V.transpose() * U);

        for (int j = 0; j < maxiter2; ++j) {
            svqb(U, tau, (j == 0) ? 'R' : 'D');
            normU = U.norm();
            const double orth_err = (U.transpose() * U - identity).norm();
            if (orth_err < tol * normU) break;
        }

        const double orth_err = (V.transpose() * U).norm();
        if (orth_err < tol * normU) break;
    }
}

bool LOBPCGSolver::modified_rayleigh_ritz() {

    /*
    ---------------------------------------------------------------------
        Overlap matrix S'AS = [ [ XP' AXP   XP' AW ]
                                [ W'  AXP    W' AW ] ]

        Assumes AXP and AW have already been computed.
    ---------------------------------------------------------------------
    */

    // Subspace sizes, accounting for locked eigenvectors
    int n_cols_XP = 2*b - b_conv;
    int n_cols_W  =   b - b_conv;

    // Computing the A-overlap matrix block by block
    // XP' AXP
    StAS.topLeftCorner(n_cols_XP,n_cols_XP).noalias() = XP.transpose() * AXP;
    // XP' AW   &   W' AXP = AW' XP
    auto tmp = XP.transpose() * AW;
    StAS.topRightCorner(n_cols_XP, n_cols_W).noalias() = XP.transpose() * AW;
    StAS.bottomLeftCorner(n_cols_W, n_cols_XP).noalias() = StAS.topRightCorner(n_cols_XP, n_cols_W).transpose();
    // W' AW
    StAS.bottomRightCorner(n_cols_W,n_cols_W).noalias() = W.transpose() * AW;

    /*
    ---------------------------------------------------------------------
        Conversion to standard eigenproblem
    ---------------------------------------------------------------------
    */

    VectorXd d;

    if (!use_orth) {

        // Overlap matrix S'S = [ [ XP' XP   XP' W ]
        //                        [ W'  XP    W' W ] ]
        // XP' XP
        StS.topLeftCorner(n_cols_XP,n_cols_XP).noalias() = XP.transpose() * XP;
        // XP' W   &   W' XP  =  XP' W
        StS.topRightCorner(n_cols_XP, n_cols_W).noalias() = XP.transpose() * W;
        StS.bottomLeftCorner(n_cols_W, n_cols_XP).noalias() = StS.topRightCorner(n_cols_XP, n_cols_W).transpose();
        // W' W
        StS.bottomRightCorner(n_cols_W,n_cols_W).noalias() = W.transpose() * W;

        // Rescale by norms of the columns of S, currently stored
        // on the diagonal of StS
        d = StS.diagonal();
        d = d.array().sqrt().inverse();
        auto D = d.asDiagonal();
        StS.applyOnTheLeft(D);
        StS.applyOnTheRight(D);
        // Apply to StAS as well
        StAS.applyOnTheLeft(D);
        StAS.applyOnTheRight(D);

        // In-place Cholesky factorization of Rt * R = StS
        // Note 1: We can do this in-place, editing StS, as we
        //         don't need StS again afterwards
        // Note 2: Using type Ref<MatrixXd> instead of MatrixXd makes
        //         this an inplace operator
        LLT<Ref<MatrixXd>,Lower> chol(StS);

        // Condition number of Rt
        // Because Rt is lower triangular, cond(Rt) is the ratio of the max
        // and min diagonal elements of Rt. We did the cholesky decom in-place,
        // so diag(Rt) is stored in diag(StS) currently.
        double rdmax = StS.diagonal().maxCoeff();
        double rdmin = StS.diagonal().minCoeff();
        double cond = rdmax / rdmin;

        // If Rt is too poorly conditioned, return 'true' immediately to tell
        // calling function lobpcg() to orthogonalize W
        double tau_skip = 1 / orth_tol;
        if (tau_skip > 0 && cond > tau_skip) {
            return true;
        }

        // View for factor L in the decomposition
        auto Rt = StS.triangularView<Lower>();

        // Convert from generalized eigenproblem (S'AS, S'S)
        // to standard eigenproblem (Rt^{-1}) S'AS (R^{-1})
        Rt.solveInPlace(StAS);
        Rt.solveInPlace(StAS.transpose());
    }

    /*
    ---------------------------------------------------------------------
        Solve projected eigenproblem
    ---------------------------------------------------------------------
    */
    eigsolver.compute(StAS);
    auto Z = eigsolver.eigenvectors();

    /*
    ---------------------------------------------------------------------
        Subspace Modification

    Modifying columns Z(:,u:2u) based on Z1_perp = Z(:u,u:) sub-block.
    c.f. Section 4.2 of C. Yang (2018) for more details.


    Z =  | Z1  Z1_perp |            Z1 : b x b      Z1_perp : b  x 2u
         | Z2  Z2_perp |            Z2 : u x b      Z2_perp : 2u x 2u
                                         u = b - b_conv

    We want to modify Z1_perp(:,:u) and Z2_perp(:,:u) based on Z1_perp.
    This implicitly causes the Ritz vectors forming the updated P subspace
    to be orthogonal to the un-updated X subspace.

    ---------------------------------------------------------------------
    */

    int u = b - b_conv;  // Number of unconverged eigenvecs

    // LQ decomposition, Z(:u,u:)' = Q' * L'   (this is a u x 2u block)
    // Note that we can't do this inplace as we'll need the current
    // value of this block in the subsequent computation
    auto Z1_perp = Z.topRightCorner(b, 2*u); // nrows = b, ncols = 2u
    HouseholderQR<MatrixXd> qr(Z1_perp.transpose());


    // Z(:,u:2u) <- Z(:,u:2u) * Q'(:,:u)
    // Note that Q' = qr.householderQ() represents a full 2b x 2b matrix,
    // but this matrix isn't representationed explicitly. Instead, it is
    // represented under-the-hood via Householder reflectors. I'm not sure
    // how to extract apply only the left b columns of Q', so instead I
    // just apply the full Q' to the right of Z(:,u:3u). This gives us
    // the correct result for Z(:,u:2u), but it also overwrites Z(:,2u:3u).
    // Luckily we don't need this anymore, so this is OK. There is some
    // wasted computational overhead though.
    auto Z_perp = Z.rightCols(2*u);
    Z_perp.applyOnTheRight(qr.householderQ());

    /*
    -----------------------------------------------------------------------
        Cleaning up
    -----------------------------------------------------------------------
    */

    // Copy Ritz coefficients to output matrix C
    C.noalias() = Z.leftCols(C.cols());

    // Convert back to original eigensystem (A,B) if the original S wasn't orthogonal
    if (!use_orth) {
        // C = D * (Rt^{-1})' * Z
        auto Rt = StS.triangularView<Lower>();
        Rt.transpose().solveInPlace(C);
        C.applyOnTheLeft(d.asDiagonal());
    }

    // Update subspaces X & P
    XP.applyOnTheRight(C.topRows(n_cols_XP));
    XP.noalias() +=  W * C.bottomRows(n_cols_W);
    // Update AX & AP in the same way
    AXP.applyOnTheRight(C.topRows(n_cols_XP));
    AXP.noalias() += AW * C.bottomRows(n_cols_W);

    // Copy the Ritz values for eigenvalues to lam
    lam.noalias() = eigsolver.eigenvalues().head(b);

    return use_orth;
}

}  // namespace fasteig
