#pragma once

#include "fasteig_solver.hpp"

class fasteig_problem;

namespace fasteig {

class ChebyshevSolver final : public fasteig_solver {
public:
  ChebyshevSolver(::fasteig_problem& problem,
                  const fasteig_options& options);
  FASTEIG_INT solve() override;
  static FASTEIG_INT solve(::fasteig_problem& problem,
                           const fasteig_options& options);

private:
  static void chebyshev_filter(const operations& ops,
                               MatrixXd& V,
                               FASTEIG_INT degree,
                               FASTEIG_DOUBLE lambda_min,
                               FASTEIG_DOUBLE lambda_max);
};

}  // namespace fasteig
