#pragma once

#include "core/RitzExtraction.hpp"
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

  virtual void record_block_step(FASTEIG_INT block_index,
                                 FASTEIG_INT block_size,
                                 const MatrixXd& diagonal_block,
                                 const MatrixXd& beta_block) = 0;

  virtual LanczosSnapshot compute_snapshot(FASTEIG_INT m,
                                           FASTEIG_INT nev,
                                           const std::string& which,
                                           bool need_vectors,
                                           const char* solver_name) = 0;

  virtual LanczosSnapshot compute_block_snapshot(FASTEIG_INT block_count,
                                                 FASTEIG_INT block_size,
                                                 FASTEIG_INT nev,
                                                 const std::string& which,
                                                 bool need_vectors,
                                                 const char* solver_name,
                                                 FASTEIG_INT active_dim = 0) = 0;

  virtual void prepare_krylov_schur_restart(
      FASTEIG_INT retained_cols,
      const LanczosSnapshot& snapshot,
      const char* solver_name) = 0;

  // Seed a retained_dim-by-retained_dim Ritz diagonal and the full
  // block_size-by-retained_dim coupling to the residual block.
  virtual void prepare_block_krylov_schur_restart(
      FASTEIG_INT block_size,
      const VectorXd& retained_values,
      const MatrixXd& coupling_block,
      const char* solver_name) = 0;

  virtual void finalize_eigenpairs(const LinearOperator& op,
                                   const MultiVector& basis,
                                   FASTEIG_INT active_cols,
                                   const LanczosSnapshot& snapshot,
                                   const std::string& which,
                                   FASTEIG_DOUBLE shift,
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

  void record_block_step(FASTEIG_INT block_index,
                         FASTEIG_INT block_size,
                         const MatrixXd& diagonal_block,
                         const MatrixXd& beta_block) override;

  LanczosSnapshot compute_snapshot(FASTEIG_INT m,
                                   FASTEIG_INT nev,
                                   const std::string& which,
                                   bool need_vectors,
                                   const char* solver_name) override;

  LanczosSnapshot compute_block_snapshot(FASTEIG_INT block_count,
                                         FASTEIG_INT block_size,
                                         FASTEIG_INT nev,
                                         const std::string& which,
                                         bool need_vectors,
                                         const char* solver_name,
                                         FASTEIG_INT active_dim = 0) override;

  void prepare_krylov_schur_restart(
      FASTEIG_INT retained_cols,
      const LanczosSnapshot& snapshot,
      const char* solver_name) override;

  void prepare_block_krylov_schur_restart(
      FASTEIG_INT block_size,
      const VectorXd& retained_values,
      const MatrixXd& coupling_block,
      const char* solver_name) override;

  void finalize_eigenpairs(const LinearOperator& op,
                           const MultiVector& basis,
                           FASTEIG_INT active_cols,
                           const LanczosSnapshot& snapshot,
                           const std::string& which,
                           FASTEIG_DOUBLE shift,
                           VectorXd& eigvals,
                           MatrixXd& eigvecs,
                           VectorXd& residual_norms) override;

private:
  MatrixXd T_;
  MatrixXd beta_block_;
  double beta_next_ = 0.0;
  FASTEIG_INT restart_coupling_row_ = -1;
};

std::unique_ptr<LanczosWorkspace> make_lanczos_workspace(const Executor& exec,
                                                         FASTEIG_INT n,
                                                         FASTEIG_INT maxiter,
                                                         FASTEIG_INT nev);

}  // namespace core
}  // namespace fasteig

