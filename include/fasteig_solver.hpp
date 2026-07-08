//********************************************************************************
//
//  Authors
//    Dong Min Roh
//    Chao Yang
//    Applied Mathematics and Computational Research Division
//    Lawrence Berkeley National Laboratory
//
//  Refactor notes (2025-12-30)
//    - Introduces RAII-friendly parameter and input parsing classes.
//    - Keeps a small C-compatible wrapper function signature: parseinput(char*, fasteig_param*).
//    - Removes C-style heap management (malloc/free) from parseinput implementation.
//
//********************************************************************************

#pragma once

// Keep this header lean: configuration macros, Eigen aliases (VectorXd/MatrixXd),
// and the operations struct come from fasteig.hpp.
#include "fasteig.hpp"
#include "fasteig_options.hpp"

#include <memory>
#include <optional>
#include <string>

void rmm_diis(const VectorXd v0,
              const operations& ops,
              FASTEIG_SIZET max_it,
              VectorXd& eigvec,
              FASTEIG_DOUBLE& eigval,
              FASTEIG_DOUBLE tol);

void SPPC_Ind( const MatrixXd &V0,
               const MatrixXd &L0,
               const operations &ops,
               int ns,
               int m,
               int P,
               MatrixXd &X,
               MatrixXd &EIGS,
               MatrixXd &RESID,
               FASTEIG_DOUBLE tol,
               bool normalize = false );

// =============================
// Solver class hierarchy
// =============================
class fasteig_problem;

namespace fasteig {

// Utility: sort Ritz value indices for solver selection rules.
template <typename T>
std::vector<size_t> sort_indices(const T& v, std::string which) {
  std::vector<size_t> idx(v.size());
  std::iota(idx.begin(), idx.end(), 0);

  if (which == "sr") {
    std::stable_sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) {
      return v(i1) < v(i2);
    });
  } else if (which == "lr") {
    std::stable_sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) {
      return v(i1) > v(i2);
    });
  } else if (which == "sm") {
    std::stable_sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) {
      return std::abs(v(i1)) < std::abs(v(i2));
    });
  } else if (which == "lm" || which == "sai") {
    std::stable_sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) {
      return std::abs(v(i1)) > std::abs(v(i2));
    });
  }

  return idx;
}

class fasteig_solver {
public:
  virtual ~fasteig_solver() = default;
  virtual FASTEIG_INT solve() = 0;

  FASTEIG_INT status() const noexcept { return status_; }
  std::optional<long long> iterations() const noexcept {
    return (iterations_ >= 0) ? std::optional<long long>(iterations_) : std::nullopt;
  }

  static void RayleighRitz(const operations &ops,
                           MatrixXd &V,
                           VectorXd &Ritzvals,
                           MatrixXd &Ritzvecs,
                           MatrixXd &eigenvectors,
                           MatrixXd &residuals);

  static void RayleighRitz(const operations& ops,
                           const Eigen::Ref<const MatrixXd>& V,
                           FASTEIG_SIZET nev,
                           std::string which,
                           VectorXd& Ritzvals,
                           MatrixXd& Ritzvecs,
                           MatrixXd& residuals);

  static VectorXd compute_residual_norms(const operations& ops,
                                         const VectorXd& eigvals,
                                         const MatrixXd& eigvecs);

  static VectorXd compute_residual_norms(const operations& opsA,
                                         const operations* opsB,
                                         const VectorXd& eigvals,
                                         const MatrixXd& eigvecs);

protected:
  fasteig_solver(::fasteig_problem& problem,
                 const fasteig_options& options);
  const operations& active_ops() const noexcept;
  void compute_final_residual_norms();
  void set_iterations(long long it) noexcept { iterations_ = it; }

  ::fasteig_problem& problem_;
  const operations& opsA_;
  const operations& opsB_;
  const operations& opsSAI_;
  VectorXd& eigvals_;
  VectorXd& resnrms_;
  MatrixXd& eigvecs_;
  fasteig_options options_;
  FASTEIG_INT status_ = 0;
  long long iterations_ = -1;
};

// Concrete solvers live in their own headers/cpps:
//   include/*Solver.hpp + src/*Solver.cpp
class BlockLanczosSolver;
class BlockKrylovSchurSolver;
class KrylovSchurSolver;
class DavidsonSolver;
class LOBPCGSolver;
class ChebyshevSolver;
class SPPCSolver;
class RMMDIISSolver;

std::unique_ptr<fasteig_solver> make_fasteig_solver(const std::string& solver,
                                                    ::fasteig_problem& problem,
                                                    const fasteig_options& options);

}  // namespace fasteig
