#include "gpu/CudaDssSolver.hpp"

#if defined(USE_CUDA) && defined(USE_CUDSS)

#include "gpu/CudaCheck.hpp"
#include "gpu/CudaContext.hpp"
#include "gpu/CudaExecutor.hpp"
#include "gpu/CudaUtils.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace fasteig {
namespace gpu {

CudaDssSolver::CudaDssSolver(const core::Executor& exec,
                             core::CsrMatrixData matrix,
                             cudssMatrixType_t matrix_type,
                             cudssMatrixViewType_t matrix_view)
    : exec_(exec), rows_(matrix.rows), cols_(matrix.cols), nnz_(matrix.nonzeros()) {
  const auto setup_t0 = std::chrono::steady_clock::now();

  core::validate_csr_matrix_data(matrix);
  if (rows_ != cols_) {
    throw std::invalid_argument("CudaDssSolver: matrix must be square.");
  }
  if (rows_ <= 0) {
    throw std::invalid_argument("CudaDssSolver: matrix dimension must be positive.");
  }

  auto& context = core::cuda_executor_from(exec_, "CudaDssSolver").context();

  h_row_offsets32_ = matrix.rowptr;
  h_row_offsets64_.resize(matrix.rowptr.size());
  for (std::size_t k = 0; k < matrix.rowptr.size(); ++k) {
    h_row_offsets64_[k] = static_cast<int64_t>(matrix.rowptr[k]);
  }
  h_colind32_ = matrix.colind;
  h_colind64_.resize(matrix.colind.size());
  for (std::size_t k = 0; k < matrix.colind.size(); ++k) {
    h_colind64_[k] = static_cast<int64_t>(matrix.colind[k]);
  }
  h_values_ = matrix.values;

  FASTEIG_CUDA_CHECK(cudaMalloc(&d_row_offsets32_, h_row_offsets32_.size() * sizeof(int)));
  FASTEIG_CUDA_CHECK(cudaMalloc(&d_row_offsets64_, h_row_offsets64_.size() * sizeof(int64_t)));
  if (nnz_ > 0) {
    FASTEIG_CUDA_CHECK(cudaMalloc(&d_colind32_, h_colind32_.size() * sizeof(int)));
    FASTEIG_CUDA_CHECK(cudaMalloc(&d_colind64_, h_colind64_.size() * sizeof(int64_t)));
    FASTEIG_CUDA_CHECK(cudaMalloc(&d_values_, h_values_.size() * sizeof(double)));
  }

  FASTEIG_CUDA_CHECK(cudaMemcpyAsync(d_row_offsets32_, h_row_offsets32_.data(),
                                     h_row_offsets32_.size() * sizeof(int),
                                     cudaMemcpyHostToDevice, context.stream()));
  FASTEIG_CUDA_CHECK(cudaMemcpyAsync(d_row_offsets64_, h_row_offsets64_.data(),
                                     h_row_offsets64_.size() * sizeof(int64_t),
                                     cudaMemcpyHostToDevice, context.stream()));
  if (nnz_ > 0) {
    FASTEIG_CUDA_CHECK(cudaMemcpyAsync(d_colind32_, h_colind32_.data(),
                                       h_colind32_.size() * sizeof(int),
                                       cudaMemcpyHostToDevice, context.stream()));
    FASTEIG_CUDA_CHECK(cudaMemcpyAsync(d_colind64_, h_colind64_.data(),
                                       h_colind64_.size() * sizeof(int64_t),
                                       cudaMemcpyHostToDevice, context.stream()));
    FASTEIG_CUDA_CHECK(cudaMemcpyAsync(d_values_, h_values_.data(),
                                       h_values_.size() * sizeof(double),
                                       cudaMemcpyHostToDevice, context.stream()));
  }
  FASTEIG_CUDA_CHECK(cudaStreamSynchronize(context.stream()));

  FASTEIG_CUDSS_CHECK(cudssCreate(&handle_));
  FASTEIG_CUDSS_CHECK(cudssSetStream(handle_, context.stream()));
  FASTEIG_CUDSS_CHECK(cudssConfigCreate(&config_));
  FASTEIG_CUDSS_CHECK(cudssDataCreate(handle_, &data_));

  auto try_create_matrix = [&](const char* label,
                               const void* row_start,
                               const void* row_end,
                               const void* colind,
                               const void* values,
                               cudssDataType_t index_type) {
    matrix_ = nullptr;
    cudssStatus_t status = cudssMatrixCreateCsr(&matrix_,
                                                static_cast<int64_t>(rows_),
                                                static_cast<int64_t>(cols_),
                                                static_cast<int64_t>(nnz_),
                                                row_start,
                                                row_end,
                                                colind,
                                                values,
                                                index_type,
                                                index_type,
                                                CUDSS_R_64F,
                                                matrix_type,
                                                matrix_view,
                                                CUDSS_BASE_ZERO);
    if (status != CUDSS_STATUS_SUCCESS) {
      if (matrix_) {
        cudssMatrixDestroy(matrix_);
        matrix_ = nullptr;
      }
    }
    return std::pair<const char*, cudssStatus_t>{label, status};
  };

  std::vector<std::pair<const char*, cudssStatus_t>> attempts;
  attempts.push_back(try_create_matrix("device int32 row-offset CSR",
                                       d_row_offsets32_, nullptr,
                                       d_colind32_, d_values_,
                                       CUDSS_R_32I));
  if (!matrix_) {
    attempts.push_back(try_create_matrix("device int64 row-offset CSR",
                                         d_row_offsets64_, nullptr,
                                         d_colind64_, d_values_,
                                         CUDSS_R_64I));
  }
  if (!matrix_) {
    attempts.push_back(try_create_matrix("host int32 row-offset CSR",
                                         h_row_offsets32_.data(), nullptr,
                                         h_colind32_.empty() ? nullptr : h_colind32_.data(),
                                         h_values_.empty() ? nullptr : h_values_.data(),
                                         CUDSS_R_32I));
  }
  if (!matrix_) {
    attempts.push_back(try_create_matrix("host int64 row-offset CSR",
                                         h_row_offsets64_.data(), nullptr,
                                         h_colind64_.empty() ? nullptr : h_colind64_.data(),
                                         h_values_.empty() ? nullptr : h_values_.data(),
                                         CUDSS_R_64I));
  }
  if (!matrix_) {
    std::ostringstream ss;
    ss << "CudaDssSolver: cudssMatrixCreateCsr failed for all CSR descriptor variants"
       << " (rows=" << rows_ << ", cols=" << cols_ << ", nnz=" << nnz_ << ").";
    for (const auto& [label, status] : attempts) {
      ss << " " << label << "=" << cudss_status_name(status)
         << "(" << static_cast<int>(status) << ").";
    }
    throw std::runtime_error(ss.str());
  }

  const auto matrix_setup_t1 = std::chrono::steady_clock::now();
  timings_.matrix_setup_sec = std::chrono::duration<double>(matrix_setup_t1 - setup_t0).count();

  analyze_and_factorize();
  const auto setup_t1 = std::chrono::steady_clock::now();
  timings_.setup_sec = std::chrono::duration<double>(setup_t1 - setup_t0).count();
}

CudaDssSolver::~CudaDssSolver() {
  if (matrix_) cudssMatrixDestroy(matrix_);
  if (data_) cudssDataDestroy(handle_, data_);
  if (config_) cudssConfigDestroy(config_);
  if (handle_) cudssDestroy(handle_);
  if (d_row_offsets32_) cudaFree(d_row_offsets32_);
  if (d_colind32_) cudaFree(d_colind32_);
  if (d_row_offsets64_) cudaFree(d_row_offsets64_);
  if (d_colind64_) cudaFree(d_colind64_);
  if (d_values_) cudaFree(d_values_);
}

void CudaDssSolver::analyze_and_factorize() {
  auto& context = core::cuda_executor_from(exec_, "CudaDssSolver::analyze_and_factorize").context();

  double* d_dummy_rhs = nullptr;
  double* d_dummy_solution = nullptr;
  const std::size_t vector_bytes = static_cast<std::size_t>(rows_) * sizeof(double);
  FASTEIG_CUDA_CHECK(cudaMalloc(&d_dummy_rhs, vector_bytes));
  FASTEIG_CUDA_CHECK(cudaMalloc(&d_dummy_solution, vector_bytes));
  FASTEIG_CUDA_CHECK(cudaMemsetAsync(d_dummy_rhs, 0, vector_bytes, context.stream()));
  FASTEIG_CUDA_CHECK(cudaMemsetAsync(d_dummy_solution, 0, vector_bytes, context.stream()));

  cudssMatrix_t rhs = nullptr;
  cudssMatrix_t solution = nullptr;
  FASTEIG_CUDSS_CHECK(cudssMatrixCreateDn(&rhs,
                                          static_cast<int64_t>(rows_),
                                          1,
                                          static_cast<int64_t>(rows_),
                                          d_dummy_rhs,
                                          CUDSS_R_64F,
                                          CUDSS_LAYOUT_COL_MAJOR));
  FASTEIG_CUDSS_CHECK(cudssMatrixCreateDn(&solution,
                                          static_cast<int64_t>(rows_),
                                          1,
                                          static_cast<int64_t>(rows_),
                                          d_dummy_solution,
                                          CUDSS_R_64F,
                                          CUDSS_LAYOUT_COL_MAJOR));

  const auto analysis_t0 = std::chrono::steady_clock::now();
  FASTEIG_CUDSS_CHECK(cudssExecute(handle_, CUDSS_PHASE_ANALYSIS,
                                   config_, data_, matrix_, solution, rhs));
  FASTEIG_CUDA_CHECK(cudaStreamSynchronize(context.stream()));
  const auto analysis_t1 = std::chrono::steady_clock::now();

  const auto factor_t0 = std::chrono::steady_clock::now();
  FASTEIG_CUDSS_CHECK(cudssExecute(handle_, CUDSS_PHASE_FACTORIZATION,
                                   config_, data_, matrix_, solution, rhs));
  FASTEIG_CUDA_CHECK(cudaStreamSynchronize(context.stream()));
  const auto factor_t1 = std::chrono::steady_clock::now();

  timings_.analysis_sec = std::chrono::duration<double>(analysis_t1 - analysis_t0).count();
  timings_.factorization_sec = std::chrono::duration<double>(factor_t1 - factor_t0).count();

  if (solution) cudssMatrixDestroy(solution);
  if (rhs) cudssMatrixDestroy(rhs);
  cudaFree(d_dummy_solution);
  cudaFree(d_dummy_rhs);
}

void CudaDssSolver::solve(const double* d_rhs,
                          double* d_solution,
                          FASTEIG_INT rhs_cols,
                          FASTEIG_INT rhs_ld,
                          FASTEIG_INT solution_ld) const {
  if (!d_rhs || !d_solution) {
    throw std::invalid_argument("CudaDssSolver::solve: null RHS or solution pointer.");
  }
  if (rhs_cols <= 0) {
    throw std::invalid_argument("CudaDssSolver::solve: rhs_cols must be positive.");
  }
  if (rhs_ld == 0) rhs_ld = rows_;
  if (solution_ld == 0) solution_ld = rows_;
  if (rhs_ld < rows_ || solution_ld < rows_) {
    throw std::invalid_argument("CudaDssSolver::solve: leading dimension is too small.");
  }

  auto& context = core::cuda_executor_from(exec_, "CudaDssSolver::solve").context();

  cudssMatrix_t rhs = nullptr;
  cudssMatrix_t solution = nullptr;
  FASTEIG_CUDSS_CHECK(cudssMatrixCreateDn(&rhs,
                                          static_cast<int64_t>(rows_),
                                          static_cast<int64_t>(rhs_cols),
                                          static_cast<int64_t>(rhs_ld),
                                          d_rhs,
                                          CUDSS_R_64F,
                                          CUDSS_LAYOUT_COL_MAJOR));
  FASTEIG_CUDSS_CHECK(cudssMatrixCreateDn(&solution,
                                          static_cast<int64_t>(rows_),
                                          static_cast<int64_t>(rhs_cols),
                                          static_cast<int64_t>(solution_ld),
                                          d_solution,
                                          CUDSS_R_64F,
                                          CUDSS_LAYOUT_COL_MAJOR));

  FASTEIG_CUDSS_CHECK(cudssExecute(handle_, CUDSS_PHASE_SOLVE,
                                   config_, data_, matrix_, solution, rhs));
  ++solve_count_;

  if (solution) cudssMatrixDestroy(solution);
  if (rhs) cudssMatrixDestroy(rhs);
}

}  // namespace gpu
}  // namespace fasteig

#endif  // defined(USE_CUDA) && defined(USE_CUDSS)

