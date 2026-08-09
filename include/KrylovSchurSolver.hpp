#pragma once

#include "fasteig_solver.hpp"

class fasteig_problem;

namespace fasteig {

class KrylovSchurSolver final : public fasteig_solver {
public:
  KrylovSchurSolver(::fasteig_problem& problem,
                    const fasteig_options& options);

  FASTEIG_INT solve() override;

private:
  FASTEIG_INT solve_generalized_host();
};

}  // namespace fasteig

