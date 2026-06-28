#pragma once

#include "fasteig.hpp"

#include <memory>
#include <optional>
#include <string>

namespace fasteig {
namespace core {

class CsrMatrix;
struct CsrMatrixData;
class MultiVector;
class Vector;

enum class ExecutorKind {
  Cpu,
  Cuda
};

class Executor {
public:
  virtual ~Executor() = default;

  virtual ExecutorKind kind() const noexcept = 0;
  virtual const char* name() const noexcept = 0;

  virtual std::unique_ptr<Vector> create_vector(FASTEIG_INT size) const = 0;
  virtual std::unique_ptr<Vector> create_vector(const Eigen::Ref<const VectorXd>& values) const;
  virtual std::unique_ptr<Vector> create_random_vector(
      FASTEIG_INT size,
      std::optional<unsigned int> seed = std::nullopt) const;
  virtual std::unique_ptr<MultiVector> create_multivector(FASTEIG_INT rows,
                                                          FASTEIG_INT cols) const = 0;
  virtual std::unique_ptr<CsrMatrix> create_csr_matrix(const CsrMatrixData& data) const = 0;
  virtual void synchronize() const = 0;

  bool is_cpu() const noexcept { return kind() == ExecutorKind::Cpu; }
  bool is_gpu() const noexcept { return kind() == ExecutorKind::Cuda; }
};

class CpuExecutor final : public Executor {
public:
  using Executor::create_vector;

  ExecutorKind kind() const noexcept override { return ExecutorKind::Cpu; }
  const char* name() const noexcept override { return "CPU"; }

  std::unique_ptr<Vector> create_vector(FASTEIG_INT size) const override;
  std::unique_ptr<MultiVector> create_multivector(FASTEIG_INT rows,
                                                  FASTEIG_INT cols) const override;
  std::unique_ptr<CsrMatrix> create_csr_matrix(const CsrMatrixData& data) const override;
  void synchronize() const override {}
};

std::shared_ptr<Executor> make_cpu_executor();
std::shared_ptr<Executor> make_cuda_executor(int device_id = 0);
std::shared_ptr<Executor> make_executor(bool gpu_backend);

}  // namespace core
}  // namespace fasteig

