#include "gpu/CudaLanczosKernels.hpp"
#ifdef USE_CUDA
#include "gpu/CudaCheck.hpp"
namespace fasteig {
namespace gpu {
namespace {
__global__ void set_lanczos_projected_matrix_entry_kernel(double* T,
                                                          int ld,
                                                          int it,
                                                          double beta,
                                                          double alpha) {
  T[it + it * ld] = alpha;
  if (it != 0) {
    T[it + (it - 1) * ld] = beta;
    T[(it - 1) + it * ld] = beta;
  }
}
__global__ void set_lanczos_projected_matrix_entry_from_device_kernel(double* T,
                                                                      int ld,
                                                                      int it,
                                                                      const double* beta,
                                                                      const double* alpha) {
  T[it + it * ld] = *alpha;
  if (it != 0) {
    T[it + (it - 1) * ld] = *beta;
    T[(it - 1) + it * ld] = *beta;
  }
}
__global__ void normalize_column_from_device_norm_kernel(const double* x,
                                                         const double* norm,
                                                         double* y,
                                                         int n) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    y[idx] = x[idx] / (*norm);
  }
}
}  // namespace
void launch_set_lanczos_projected_matrix_entry(double* d_T,
                                               int ld,
                                               int it,
                                               double beta,
                                               double alpha,
                                               cudaStream_t stream) {
  set_lanczos_projected_matrix_entry_kernel<<<1, 1, 0, stream>>>(d_T, ld, it, beta, alpha);
  FASTEIG_CUDA_CHECK(cudaGetLastError());
}
void launch_set_lanczos_projected_matrix_entry_from_device(double* d_T,
                                                           int ld,
                                                           int it,
                                                           const double* d_beta,
                                                           const double* d_alpha,
                                                           cudaStream_t stream) {
  set_lanczos_projected_matrix_entry_from_device_kernel<<<1, 1, 0, stream>>>(
      d_T, ld, it, d_beta, d_alpha);
  FASTEIG_CUDA_CHECK(cudaGetLastError());
}
void launch_normalize_column_from_device_norm(const double* d_x,
                                              const double* d_norm,
                                              double* d_y,
                                              int n,
                                              cudaStream_t stream) {
  constexpr int block = 256;
  const int grid = (n + block - 1) / block;
  normalize_column_from_device_norm_kernel<<<grid, block, 0, stream>>>(d_x, d_norm, d_y, n);
  FASTEIG_CUDA_CHECK(cudaGetLastError());
}
}  // namespace gpu
}  // namespace fasteig
#endif  // USE_CUDA

