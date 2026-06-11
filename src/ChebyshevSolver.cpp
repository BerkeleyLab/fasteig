#include "ChebyshevSolver.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <iostream>

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
  filter_subspace(ops_eff, V0, maxiter, degree, eigvals_, eigvecs_, residuals);
  if (flip_sign) {
    eigvals_ *= -1.0;
  }
  compute_final_residual_norms();
  set_iterations(static_cast<long long>(maxiter));
  return status_;
}

void ChebyshevSolver::chebyshev_filter(const operations& ops,
                                       MatrixXd& V,
                                       FASTEIG_INT degree,
                                       FASTEIG_DOUBLE lambda_min,
                                       FASTEIG_DOUBLE lambda_max) {
  MatrixXd V_old = V;
  MatrixXd V_new = V;
  MatrixXd AV = V;

  ops.block_matvec(V, V_new);

  const FASTEIG_DOUBLE rat = 1.0e3; // ratio of Cheby(xlb) and Cheby(lb), hard-coded for now.
  const FASTEIG_DOUBLE xlb = std::cosh(std::acosh(rat) / degree);

  const FASTEIG_DOUBLE ub = lambda_max;
  const FASTEIG_DOUBLE lb = (2.0 * lambda_min + lambda_max * xlb - lambda_max) / (1.0 + xlb);

  const FASTEIG_DOUBLE c = 2.0 / (ub - lb);
  const FASTEIG_DOUBLE d = -(ub + lb) / (ub - lb);

  V_new = c * V_new + d * V;
  for (FASTEIG_INT k = 2; k <= degree; ++k) {
    ops.block_matvec(V_new, AV);
    MatrixXd V_tmp = 2.0 * (c * AV + d * V_new) - V_old;
    V_old = V_new;
    V_new = V_tmp;
  }
  V = V_new;
}

FASTEIG_INT ChebyshevSolver::filter_subspace(const operations& ops,
                                             MatrixXd& V,
                                             FASTEIG_INT maxiter,
                                             FASTEIG_INT degree,
                                             VectorXd& ritz_vals,
                                             MatrixXd& ritz_vecs,
                                             MatrixXd& residuals) {
  FASTEIG_DOUBLE lambda_max;
  FASTEIG_DOUBLE lambda_min;

  FASTEIG_INT info = 0;

  const FASTEIG_INT nrows = static_cast<FASTEIG_INT>(V.rows());
  const FASTEIG_INT ncols = static_cast<FASTEIG_INT>(V.cols());
  MatrixXd X = V;
  MatrixXd eigvecs(ncols, ncols);

  std::cout << "Chebyshev filtering for low the of the spectrum... \n" << std::endl;
  KrylovSubspace::getevbounds(nrows, ops, lambda_min, lambda_max);
  std::cout << "Spectral bounds from Lanczos: " << std::endl;
  std::cout << "lb = " << lambda_min << " ub = " << lambda_max << std::endl;

  for (FASTEIG_INT iter = 0; iter < maxiter; ++iter) {
    std::cout << "Cheby filter iter = " << iter << std::endl;

    chebyshev_filter(ops, X, degree, lambda_min, lambda_max);

    const MatrixXd Gmat = X.transpose() * X;
    Eigen::LLT<MatrixXd> llt(Gmat);
    if (llt.info() == Eigen::Success) {
      const MatrixXd L = llt.matrixL();
      ritz_vecs = X * L.transpose().triangularView<Eigen::Upper>().solve(MatrixXd::Identity(ncols, ncols));
      X = ritz_vecs;
    } else {
      std::cerr << "Decomposition failed. Matrix may not be positive definite." << std::endl;
      info = -1;
      break;
    }
  }
  if (info == 0) {
    RayleighRitz(ops, X, ritz_vals, ritz_vecs, eigvecs, residuals);
  }

  return info;
}

}  // namespace fasteig
