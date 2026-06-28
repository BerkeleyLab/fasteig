#pragma once

#include "LanczosCommon.hpp"
#include "fasteig.hpp"

#include <memory>
#include <string>

namespace fasteig {
namespace core {

class Executor;
class MultiVector;
class LinearOperator;
class Vector;

class LanczosWorkspace {
public:
  virtual ~LanczosWorkspace() = default;

  virtual void record_step(FASTEIG_INT it,
                           double beta,
                           double alpha,
                           double beta_next) = 0;

  virtual LanczosSnapshot compute_snapshot(FASTEIG_INT m,
                                           FASTEIG_INT nev,
                                           const std::string& which,
                                           bool need_vectors,
                                           const char* solver_name) = 0;

  virtual bool needs_host_basis_for_ritz_vectors() const noexcept = 0;

  virtual void finalize_eigenpairs(const LinearOperator& op,
                                   const MultiVector& basis,
                                   const MatrixXd& host_basis,
                                   FASTEIG_INT active_cols,
                                   const LanczosSnapshot& snapshot,
                                   VectorXd& eigvals,
                                   MatrixXd& eigvecs,
                                   VectorXd& residual_norms) = 0;
};

class CpuLanczosWorkspace final : public LanczosWorkspace {
public:
  explicit CpuLanczosWorkspace(FASTEIG_INT maxiter);

  void record_step(FASTEIG_INT it,
                   double beta,
                   double alpha,
                   double beta_next) override;

  LanczosSnapshot compute_snapshot(FASTEIG_INT m,
                                   FASTEIG_INT nev,
                                   const std::string& which,
                                   bool need_vectors,
                                   const char* solver_name) override;

  bool needs_host_basis_for_ritz_vectors() const noexcept override { return true; }

  void finalize_eigenpairs(const LinearOperator& op,
                           const MultiVector& basis,
                           const MatrixXd& host_basis,
                           FASTEIG_INT active_cols,
                           const LanczosSnapshot& snapshot,
                           VectorXd& eigvals,
                           MatrixXd& eigvecs,
                           VectorXd& residual_norms) override;

private:
  MatrixXd T_;
  double beta_next_ = 0.0;
};

std::unique_ptr<LanczosWorkspace> make_lanczos_workspace(const Executor& exec,
                                                         FASTEIG_INT n,
                                                         FASTEIG_INT maxiter,
                                                         FASTEIG_INT nev);

}  // namespace core
}  // namespace fasteig

