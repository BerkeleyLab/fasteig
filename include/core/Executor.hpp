#pragma once

#include "fasteig.hpp"

#include <memory>
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
  virtual std::unique_ptr<MultiVector> create_multivector(FASTEIG_INT rows,
                                                          FASTEIG_INT cols) const = 0;
  virtual std::unique_ptr<CsrMatrix> create_csr_matrix(const CsrMatrixData& data) const = 0;

  bool is_cpu() const noexcept { return kind() == ExecutorKind::Cpu; }
  bool is_gpu() const noexcept { return kind() == ExecutorKind::Cuda; }
};

class CpuExecutor final : public Executor {
public:
  ExecutorKind kind() const noexcept override { return ExecutorKind::Cpu; }
  const char* name() const noexcept override { return "CPU"; }

  std::unique_ptr<Vector> create_vector(FASTEIG_INT size) const override;
  std::unique_ptr<MultiVector> create_multivector(FASTEIG_INT rows,
                                                  FASTEIG_INT cols) const override;
  std::unique_ptr<CsrMatrix> create_csr_matrix(const CsrMatrixData& data) const override;
};

std::shared_ptr<Executor> make_cpu_executor();
#ifdef USE_CUDA
std::shared_ptr<Executor> make_cuda_executor(int device_id = 0);
#endif

}  // namespace core
}  // namespace fasteig
