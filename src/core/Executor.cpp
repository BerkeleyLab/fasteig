#include "core/Executor.hpp"

#include "core/CsrMatrix.hpp"
#include "core/MultiVector.hpp"
#include "core/Vector.hpp"

#include <memory>

namespace fasteig {
namespace core {

std::unique_ptr<Vector> CpuExecutor::create_vector(FASTEIG_INT size) const {
  return std::make_unique<CpuVector>(*this, size);
}

std::unique_ptr<MultiVector> CpuExecutor::create_multivector(FASTEIG_INT rows,
                                                             FASTEIG_INT cols) const {
  return std::make_unique<CpuMultiVector>(*this, rows, cols);
}

std::unique_ptr<CsrMatrix> CpuExecutor::create_csr_matrix(const CsrMatrixData& data) const {
  return std::make_unique<CpuCsrMatrix>(*this, data);
}

std::shared_ptr<Executor> make_cpu_executor() {
  return std::make_shared<CpuExecutor>();
}

}  // namespace core
}  // namespace fasteig
