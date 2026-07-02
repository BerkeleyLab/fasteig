#pragma once

#if defined(USE_CUDA) && defined(USE_CUDSS)

#include "core/CsrMatrix.hpp"
#include "fasteig.hpp"

#include <cudss.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace fasteig {
namespace core {
class Executor;
}  // namespace core
}  // namespace fasteig

namespace fasteig {
namespace gpu {

class CudaDssSolver {
public:
  struct Timings {
    double setup_sec = 0.0;
    double matrix_setup_sec = 0.0;
    double analysis_sec = 0.0;
    double factorization_sec = 0.0;
  };

  CudaDssSolver(const core::Executor& exec,
                core::CsrMatrixData matrix,
                cudssMatrixType_t matrix_type,
                cudssMatrixViewType_t matrix_view);
  ~CudaDssSolver();

  CudaDssSolver(const CudaDssSolver&) = delete;
  CudaDssSolver& operator=(const CudaDssSolver&) = delete;

  FASTEIG_INT rows() const noexcept { return rows_; }
  FASTEIG_INT cols() const noexcept { return cols_; }
  FASTEIG_INT nonzeros() const noexcept { return nnz_; }
  const Timings& timings() const noexcept { return timings_; }
  long long solve_count() const noexcept { return solve_count_; }

  void solve(const double* d_rhs,
             double* d_solution,
             FASTEIG_INT rhs_cols = 1,
             FASTEIG_INT rhs_ld = 0,
             FASTEIG_INT solution_ld = 0) const;

private:
  void analyze_and_factorize();

  const core::Executor& exec_;
  FASTEIG_INT rows_ = 0;
  FASTEIG_INT cols_ = 0;
  FASTEIG_INT nnz_ = 0;

  int* d_row_offsets32_ = nullptr;
  int* d_colind32_ = nullptr;
  int64_t* d_row_offsets64_ = nullptr;
  int64_t* d_colind64_ = nullptr;
  double* d_values_ = nullptr;
  std::vector<int> h_row_offsets32_;
  std::vector<int> h_colind32_;
  std::vector<int64_t> h_row_offsets64_;
  std::vector<int64_t> h_colind64_;
  std::vector<double> h_values_;

  cudssHandle_t handle_ = nullptr;
  cudssConfig_t config_ = nullptr;
  cudssData_t data_ = nullptr;
  cudssMatrix_t matrix_ = nullptr;

  Timings timings_{};
  mutable long long solve_count_ = 0;
};

}  // namespace gpu
}  // namespace fasteig

#endif  // defined(USE_CUDA) && defined(USE_CUDSS)

