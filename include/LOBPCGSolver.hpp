#pragma once

#include "fasteig_solver.hpp"

class fasteig_problem;

namespace fasteig {

// Standard symmetric SR/LR problems; all large working blocks belong to the
// problem's executor. The LOBPCG block width is the number of requested pairs.
class LOBPCGSolver final : public fasteig_solver {
public:
  LOBPCGSolver(::fasteig_problem& problem,
               const fasteig_options& options);
  FASTEIG_INT solve() override;
};

}  // namespace fasteig

