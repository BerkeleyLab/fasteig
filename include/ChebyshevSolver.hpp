#pragma once

#include "fasteig_solver.hpp"

class fasteig_problem;

namespace fasteig {

class ChebyshevSolver final : public fasteig_solver {
public:
  ChebyshevSolver(::fasteig_problem& problem,
                  const fasteig_options& options);
  FASTEIG_INT solve() override;

private:
  static void chebyshev_filter(const operations& ops,
                               MatrixXd& V,
                               FASTEIG_INT degree,
                               FASTEIG_DOUBLE lambda_min,
                               FASTEIG_DOUBLE lambda_max);

  static FASTEIG_INT filter_subspace(const operations& ops,
                                     MatrixXd& V,
                                     FASTEIG_INT maxiter,
                                     FASTEIG_INT degree,
                                     VectorXd& ritz_vals,
                                     MatrixXd& ritz_vecs,
                                     MatrixXd& residuals);
};

}  // namespace fasteig

