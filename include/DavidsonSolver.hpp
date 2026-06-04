#pragma once

#include "fasteig_solver.hpp"

class fasteig_problem;

namespace fasteig {

class DavidsonSolver final : public fasteig_solver {
public:
  DavidsonSolver(::fasteig_problem& problem,
                 const fasteig_options& options);
  FASTEIG_INT solve() override;
};

}  // namespace fasteig


