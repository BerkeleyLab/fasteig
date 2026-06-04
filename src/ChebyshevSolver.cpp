#include "ChebyshevSolver.hpp"

#include <algorithm>
#include <cctype>

namespace fasteig {

ChebyshevSolver::ChebyshevSolver(::fasteig_problem& problem,
                                 const fasteig_options& options)
    : fasteig_solver(problem, options) {}

FASTEIG_INT ChebyshevSolver::solve() {
  auto lower_copy = [](std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
  };
  const std::string which_l = lower_copy(options_.which());
  const bool flip_sign = (which_l == "lr" || which_l == "lm");

  operations ops_eff = active_ops();
  if (flip_sign) {
    const auto orig_mv = ops_eff.matvec;
    const auto orig_bmv = ops_eff.block_matvec;
    if (orig_mv) {
      ops_eff.matvec = [orig_mv](const VectorXd& v, VectorXd& av) {
        orig_mv(v, av);
        av *= -1.0;
      };
    }
    if (orig_bmv) {
      ops_eff.block_matvec = [orig_bmv](const MatrixXd& V, MatrixXd& AV) {
        orig_bmv(V, AV);
        AV *= -1.0;
      };
    }
  }

  const FASTEIG_INT n = eigvecs_.rows();
  const FASTEIG_INT nev = options_.nev();
  const FASTEIG_INT maxiter = options_.maxiter();
  const FASTEIG_INT degree = options_.degree();

  MatrixXd V0;
  if (options_.v0().size() > 0) {
    V0 = options_.v0().leftCols(static_cast<int>(nev));
  } else {
    V0 = MatrixXd::Random(n, nev);
  }
  MatrixXd residuals(n, nev);
  chebyfsit(ops_eff, V0, maxiter, degree, eigvals_, eigvecs_, residuals);
  if (flip_sign) {
    eigvals_ *= -1.0;
  }
  compute_final_residual_norms();
  set_iterations(static_cast<long long>(maxiter));
  return status_;
}

}  // namespace fasteig
