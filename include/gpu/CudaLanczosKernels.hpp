#pragma once

#ifdef USE_CUDA

#include <cuda_runtime.h>

namespace fasteig {
namespace gpu {

void launch_set_lanczos_projected_matrix_entry(double* d_T,
                                               int ld,
                                               int it,
                                               double beta,
                                               double alpha,
                                               cudaStream_t stream);

void launch_set_lanczos_projected_matrix_entry_from_device(double* d_T,
                                                           int ld,
                                                           int it,
                                                           const double* d_beta,
                                                           const double* d_alpha,
                                                           cudaStream_t stream);

void launch_normalize_column_from_device_norm(const double* d_x,
                                              const double* d_norm,
                                              double* d_y,
                                              int n,
                                              cudaStream_t stream);

}  // namespace gpu
}  // namespace fasteig

#endif  // USE_CUDA

