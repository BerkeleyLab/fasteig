#pragma once

#include "fasteig_solver.hpp"

class fasteig_problem;

namespace fasteig {

class ChebyshevSolver final : public fasteig_solver {
public:
  ChebyshevSolver(::fasteig_problem& problem,
                  const fasteig_options& options);
  FASTEIG_INT solve() override;
};

}  // namespace fasteig


