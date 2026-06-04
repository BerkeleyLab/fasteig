#include "RMMDIISSolver.hpp"

#include "fasteig_problem.hpp"

namespace fasteig {

RMMDIISSolver::RMMDIISSolver(::fasteig_problem& problem,
                             const fasteig_options& options)
    : fasteig_solver(problem, options) {}

FASTEIG_INT RMMDIISSolver::solve() {
  const operations& ops = active_ops();
  const FASTEIG_INT n = eigvecs_.rows();
  const FASTEIG_INT nev = options_.nev();
  const FASTEIG_INT maxiter = options_.maxiter();
  const std::string which = options_.which();
  const FASTEIG_DOUBLE tol = options_.tol();

  fasteig_problem ks_prob(ops, n, "KrylovSchur");
  fasteig_options ks_opts(n);
  ks_opts.set("solver", "KrylovSchur");
  ks_opts.set("nev", std::to_string(nev));
  ks_opts.set("maxiter", std::to_string(maxiter));
  ks_opts.set("which", which);
  ks_opts.set("tol", "1e-3");
  ks_opts.set("verbosity", "1");
  ks_opts.set("print_every", "0");
  ks_opts.v0() = MatrixXd::Random(n, nev);

  (void)ks_prob.solve(ks_opts);
  const MatrixXd& ksvecs = ks_prob.eigenvectors();

  for (FASTEIG_INT j = 0; j < nev; ++j) {
    VectorXd rmmvec(n);
    FASTEIG_DOUBLE rmmval;
    rmm_diis(ksvecs.col(j), ops, maxiter, rmmvec, rmmval, tol);
    eigvecs_.col(j) = rmmvec;
    eigvals_(j) = rmmval;
  }
  compute_final_residual_norms();
  return status_;
}

}  // namespace fasteig
