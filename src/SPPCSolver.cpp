#include "SPPCSolver.hpp"

#include "core/Executor.hpp"
#include "core/LinearOperator.hpp"
#include "fasteig_problem.hpp"

#include <memory>

namespace fasteig {

SPPCSolver::SPPCSolver(::fasteig_problem& problem,
                       const fasteig_options& options)
    : fasteig_solver(problem, options) {}

FASTEIG_INT SPPCSolver::solve() {
  const operations& legacy_ops = active_ops();
  std::shared_ptr<core::Executor> exec = problem_.executor(options_);
  std::unique_ptr<core::LinearOperator> op = problem_.make_linear_operator(options_, *exec);
  operations ops = make_host_operations_adapter(*op);
  ops.H0 = legacy_ops.H0;
  ops.V21 = legacy_ops.V21;
  const int n = static_cast<int>(eigvecs_.rows());
  const int ns = static_cast<int>(ops.H0.rows());
  const FASTEIG_INT nev = options_.nev();
  const FASTEIG_INT maxiter = options_.maxiter();
  const std::string which = options_.which();
  const FASTEIG_DOUBLE tol = options_.tol();

  operations ops_H0;
  ops_H0.matvec = [&H0 = ops.H0](const VectorXd& v, VectorXd& av) {
    av = H0.selfadjointView<Eigen::Lower>() * v;
  };

  VectorXd v0_H0 = VectorXd::Ones(ns);
  fasteig_problem ks_prob(ops_H0, ns, "KrylovSchur");
  fasteig_options ks_opts(ns);
  ks_opts.set("solver", "KrylovSchur");
  ks_opts.set("nev", std::to_string(nev));
  ks_opts.set("maxiter", "5000");
  ks_opts.set("which", which);
  ks_opts.set("tol", "1e-14");
  ks_opts.set("verbosity", "1");
  ks_opts.set("print_every", "0");

  ks_opts.v0() = MatrixXd::Random(ns, nev);
  ks_opts.v0().col(0) = v0_H0;

  (void)ks_prob.solve(ks_opts);
  const MatrixXd& ritz_vecs = ks_prob.eigenvectors();
  const VectorXd& ritz_vals = ks_prob.eigenvalues();

  MatrixXd V0 = MatrixXd::Zero(n, nev);
  V0.topRows(ns) = ritz_vecs;
  MatrixXd L0 = MatrixXd::Zero(nev, nev);
  for (FASTEIG_INT i = 0; i < nev; ++i) {
    L0(i, i) = ritz_vals(i);
  }

  MatrixXd EIGS(maxiter + 1, nev);
  MatrixXd RESID(maxiter + 1, nev);
  SPPC_Ind(V0, L0, ops, ns, nev, maxiter, eigvecs_, EIGS, RESID, tol);
  for (FASTEIG_INT j = 0; j < nev; ++j) {
    eigvals_(j) = EIGS(maxiter, j);
  }
  compute_final_residual_norms();
  return status_;
}

}  // namespace fasteig

