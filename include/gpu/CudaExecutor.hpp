#pragma once

#ifdef USE_CUDA

#include "core/Executor.hpp"

namespace fasteig::gpu {
class CudaContext;
}

namespace fasteig {
namespace core {

class CudaExecutor final : public Executor {
public:
  using Executor::create_vector;
  using Executor::create_multivector;

  explicit CudaExecutor(int device_id = 0);

  ExecutorKind kind() const noexcept override { return ExecutorKind::Cuda; }
  const char* name() const noexcept override { return "CUDA"; }
  int device_id() const noexcept { return device_id_; }
  gpu::CudaContext& context() const;

  std::unique_ptr<Vector> create_vector(FASTEIG_INT size) const override;
  std::unique_ptr<MultiVector> create_multivector(FASTEIG_INT rows,
                                                  FASTEIG_INT cols) const override;
  std::unique_ptr<CsrMatrix> create_csr_matrix(const CsrMatrixData& data) const override;
  void synchronize() const override;

private:
  int device_id_ = 0;
  mutable std::shared_ptr<gpu::CudaContext> context_;
};

}  // namespace core
}  // namespace fasteig

#endif  // USE_CUDA

