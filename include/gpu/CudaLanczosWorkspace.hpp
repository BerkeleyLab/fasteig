#pragma once

#ifdef USE_CUDA

#include "LanczosCommon.hpp"
#include "core/Executor.hpp"
#include "core/LanczosWorkspace.hpp"
#include "fasteig.hpp"

#include <cusolverDn.h>

#include <string>

namespace fasteig {
namespace core {

class MultiVector;

class CudaLanczosWorkspace final : public LanczosWorkspace {
public:
  CudaLanczosWorkspace(const Executor& exec,
                       FASTEIG_INT n,
                       FASTEIG_INT maxiter,
                       FASTEIG_INT nev);
  ~CudaLanczosWorkspace();

  CudaLanczosWorkspace(const CudaLanczosWorkspace&) = delete;
  CudaLanczosWorkspace& operator=(const CudaLanczosWorkspace&) = delete;

  void record_step(FASTEIG_INT it,
                   double beta,
                   double alpha,
                   double beta_next) override;

  LanczosSnapshot compute_snapshot(FASTEIG_INT m,
                                   FASTEIG_INT nev,
                                   const std::string& which,
                                   bool need_vectors,
                                   const char* solver_name) override;

  bool needs_host_basis_for_ritz_vectors() const noexcept override { return false; }

  void finalize_eigenpairs(const LinearOperator& op,
                           const MultiVector& basis,
                           const MatrixXd& host_basis,
                           FASTEIG_INT active_cols,
                           const LanczosSnapshot& snapshot,
                           VectorXd& eigvals,
                           MatrixXd& eigvecs,
                           VectorXd& residual_norms) override;

private:
  const Executor& exec_;
  FASTEIG_INT n_ = 0;
  FASTEIG_INT maxiter_ = 0;
  FASTEIG_INT nev_ = 0;

  double* d_T_ = nullptr;
  double* d_snapshot_T_ = nullptr;
  double* d_W_ = nullptr;
  double* d_work_ = nullptr;
  double* d_Csel_ = nullptr;
  double* d_ritz_ = nullptr;
  int* d_dev_info_ = nullptr;
  int lwork_ = 0;
  cusolverDnHandle_t cusolver_handle_ = nullptr;
  double beta_next_ = 0.0;
};

}  // namespace core
}  // namespace fasteig

#endif  // USE_CUDA

