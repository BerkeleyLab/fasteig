#include "gpu/CudaExecutor.hpp"

#ifdef USE_CUDA

#include "gpu/CudaCsrMatrix.hpp"
#include "gpu/CudaMultiVector.hpp"
#include "gpu/CudaVector.hpp"
#include "gpu/CudaContext.hpp"
#include "gpu/CudaCheck.hpp"

#include <cuda_runtime.h>

namespace fasteig {
namespace core {

CudaExecutor::CudaExecutor(int device_id) : device_id_(device_id) {}

gpu::CudaContext& CudaExecutor::context() const {
  if (!context_) {
    FASTEIG_CUDA_CHECK(cudaSetDevice(device_id_));
    context_ = std::make_shared<gpu::CudaContext>();
  }
  return *context_;
}

std::unique_ptr<Vector> CudaExecutor::create_vector(FASTEIG_INT size) const {
  return std::make_unique<CudaVector>(*this, size);
}

std::unique_ptr<MultiVector> CudaExecutor::create_multivector(FASTEIG_INT rows,
                                                              FASTEIG_INT cols) const {
  return std::make_unique<CudaMultiVector>(*this, rows, cols);
}

std::unique_ptr<CsrMatrix> CudaExecutor::create_csr_matrix(const CsrMatrixData& data) const {
  return std::make_unique<CudaCsrMatrix>(*this, data);
}

void CudaExecutor::synchronize() const {
  FASTEIG_CUDA_CHECK(cudaStreamSynchronize(context().stream()));
}

std::shared_ptr<Executor> make_cuda_executor(int device_id) {
  return std::make_shared<CudaExecutor>(device_id);
}

}  // namespace core
}  // namespace fasteig

#endif  // USE_CUDA

