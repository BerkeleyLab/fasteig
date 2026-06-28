#include "core/Executor.hpp"

#include "core/CsrMatrix.hpp"
#include "core/MultiVector.hpp"
#include "core/Vector.hpp"

#include <memory>
#include <random>
#include <stdexcept>

namespace fasteig {
namespace core {

std::unique_ptr<Vector> Executor::create_vector(const Eigen::Ref<const VectorXd>& values) const {
  auto x = create_vector(static_cast<FASTEIG_INT>(values.size()));
  x->copy_from_host(values);
  return x;
}

std::unique_ptr<Vector> Executor::create_random_vector(FASTEIG_INT size,
                                                       std::optional<unsigned int> seed) const {
  if (size <= 0) {
    throw std::invalid_argument("Executor::create_random_vector: size must be positive.");
  }
  constexpr unsigned int kDefaultSeed = 1u;
  std::mt19937 rng(seed.value_or(kDefaultSeed));
  std::uniform_real_distribution<double> dist(-1.0, 1.0);

  VectorXd values(static_cast<int>(size));
  for (int i = 0; i < values.size(); ++i) {
    values(i) = dist(rng);
  }
  return create_vector(values);
}

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

#ifndef USE_CUDA
std::shared_ptr<Executor> make_cuda_executor(int) {
  throw std::invalid_argument("GPU backend requested but FASTEIG was not built with CUDA.");
}
#endif

std::shared_ptr<Executor> make_executor(bool gpu_backend) {
  if (gpu_backend) {
    return make_cuda_executor();
  }
  return make_cpu_executor();
}

}  // namespace core
}  // namespace fasteig

