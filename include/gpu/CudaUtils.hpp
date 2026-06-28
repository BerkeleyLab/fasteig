#pragma once

#ifdef USE_CUDA

#include "core/Executor.hpp"
#include "gpu/CudaExecutor.hpp"

#include <stdexcept>
#include <string>

namespace fasteig {
namespace core {

inline const CudaExecutor& cuda_executor_from(const Executor& exec, const char* where) {
  const auto* cuda_exec = dynamic_cast<const CudaExecutor*>(&exec);
  if (!cuda_exec) {
    throw std::invalid_argument(std::string(where) + ": executor is not a CudaExecutor.");
  }
  return *cuda_exec;
}

}  // namespace core
}  // namespace fasteig

#endif  // USE_CUDA

