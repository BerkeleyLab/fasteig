#include "core/Orthogonalization.hpp"

#include "core/Executor.hpp"
#include "core/MultiVector.hpp"

#ifdef USE_CUDA
#include "gpu/CudaCheck.hpp"
#include "gpu/CudaContext.hpp"
#include "gpu/CudaExecutor.hpp"
#include "gpu/CudaMultiVector.hpp"
#include "gpu/CudaUtils.hpp"

#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace fasteig {
namespace core {

namespace {

void validate_tall_skinny(const Eigen::Ref<const MatrixXd>& X, const char* where) {
  if (X.rows() <= 0 || X.cols() <= 0) {
    throw std::invalid_argument(std::string(where) + ": input matrix must be nonempty.");
  }
  if (X.rows() < X.cols()) {
    throw std::invalid_argument(std::string(where) + ": row count must be >= column count.");
  }
}

void validate_multivector_qr(const MultiVector& X, const char* where) {
  if (X.rows() <= 0 || X.cols() <= 0) {
    throw std::invalid_argument(std::string(where) + ": input multivector must be nonempty.");
  }
  if (X.rows() < X.cols()) {
    throw std::invalid_argument(std::string(where) + ": row count must be >= column count.");
  }
}

FASTEIG_INT host_thin_qr(const Eigen::Ref<const MatrixXd>& X,
                         MatrixXd& Q,
                         MatrixXd& R,
                         QrMethod method) {
  switch (method) {
    case QrMethod::Householder:
      return householder_qr(X, Q, R);
    case QrMethod::Cholesky:
      return cholesky_qr(X, Q, R);
    case QrMethod::Cholesky2:
      return cholesky_qr2(X, Q, R);
    case QrMethod::GramSchmidt:
      return gram_schmidt_qr(X, Q, R);
    case QrMethod::Svqb: {
      MatrixXd original = X;
      Q = original;
      const FASTEIG_INT info = svqb(Q);
      if (info != 0) return info;
      R = Q.transpose() * original;
      return 0;
    }
  }
  return -1;
}

#ifdef USE_CUDA
int checked_int(FASTEIG_INT value, const char* where) {
  if (value < 0 || value > static_cast<FASTEIG_INT>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument(std::string(where) + ": value does not fit in int.");
  }
  return static_cast<int>(value);
}

struct DeviceDoubleBuffer {
  double* ptr = nullptr;
  std::size_t capacity = 0;

  ~DeviceDoubleBuffer() {
    if (ptr) cudaFree(ptr);
  }

  void ensure(std::size_t count) {
    if (count <= capacity) return;
    if (ptr) {
      FASTEIG_CUDA_CHECK(cudaFree(ptr));
      ptr = nullptr;
      capacity = 0;
    }
    if (count > 0) {
      FASTEIG_CUDA_CHECK(cudaMalloc(&ptr, count * sizeof(double)));
      capacity = count;
    }
  }
};

struct DeviceIntBuffer {
  int* ptr = nullptr;

  ~DeviceIntBuffer() {
    if (ptr) cudaFree(ptr);
  }

  void allocate() {
    if (!ptr) FASTEIG_CUDA_CHECK(cudaMalloc(&ptr, sizeof(int)));
  }
};

void check_cusolver_info(const gpu::CudaContext& context, int* d_info, const char* where) {
  int info = 0;
  FASTEIG_CUDA_CHECK(cudaMemcpyAsync(&info, d_info, sizeof(int),
                                     cudaMemcpyDeviceToHost, context.stream()));
  FASTEIG_CUDA_CHECK(cudaStreamSynchronize(context.stream()));
  if (info != 0) {
    throw std::runtime_error(std::string(where) + ": cuSOLVER reported info=" +
                             std::to_string(info) + ".");
  }
}

FASTEIG_INT cuda_householder_qr_in_place(CudaMultiVector& X, MatrixXd& R) {
  validate_multivector_qr(X, "cuda_householder_qr_in_place");

  const int n = checked_int(X.rows(), "cuda_householder_qr_in_place(rows)");
  const int b = checked_int(X.cols(), "cuda_householder_qr_in_place(cols)");
  auto& context = cuda_executor_from(X.executor(), "cuda_householder_qr_in_place").context();

  DeviceDoubleBuffer d_tau;
  DeviceDoubleBuffer d_work;
  DeviceIntBuffer d_info;
  d_tau.ensure(static_cast<std::size_t>(b));
  d_info.allocate();

  int lwork_geqrf = 0;
  int lwork_orgqr = 0;
  FASTEIG_CUSOLVER_CHECK(cusolverDnDgeqrf_bufferSize(context.cusolver_handle(),
                                                     n, b, X.data(), n, &lwork_geqrf));
  FASTEIG_CUSOLVER_CHECK(cusolverDnDorgqr_bufferSize(context.cusolver_handle(),
                                                     n, b, b, X.data(), n,
                                                     d_tau.ptr, &lwork_orgqr));
  d_work.ensure(static_cast<std::size_t>(std::max(lwork_geqrf, lwork_orgqr)));

  FASTEIG_CUSOLVER_CHECK(cusolverDnDgeqrf(context.cusolver_handle(),
                                          n, b, X.data(), n, d_tau.ptr,
                                          d_work.ptr, lwork_geqrf, d_info.ptr));

  MatrixXd raw_r(static_cast<int>(b), static_cast<int>(b));
  FASTEIG_CUDA_CHECK(cudaMemcpy2DAsync(raw_r.data(),
                                       static_cast<std::size_t>(b) * sizeof(double),
                                       X.data(),
                                       static_cast<std::size_t>(n) * sizeof(double),
                                       static_cast<std::size_t>(b) * sizeof(double),
                                       static_cast<std::size_t>(b),
                                       cudaMemcpyDeviceToHost,
                                       context.stream()));
  check_cusolver_info(context, d_info.ptr, "cuda_householder_qr_in_place(geqrf)");

  FASTEIG_CUSOLVER_CHECK(cusolverDnDorgqr(context.cusolver_handle(),
                                          n, b, b, X.data(), n, d_tau.ptr,
                                          d_work.ptr, lwork_orgqr, d_info.ptr));
  check_cusolver_info(context, d_info.ptr, "cuda_householder_qr_in_place(orgqr)");

  R = MatrixXd::Zero(b, b);
  R.template triangularView<Eigen::Upper>() =
      raw_r.template triangularView<Eigen::Upper>();
  return 0;
}

FASTEIG_INT cuda_cholesky_qr_in_place(CudaMultiVector& X, MatrixXd& R) {
  validate_multivector_qr(X, "cuda_cholesky_qr_in_place");

  const int n = checked_int(X.rows(), "cuda_cholesky_qr_in_place(rows)");
  const int b = checked_int(X.cols(), "cuda_cholesky_qr_in_place(cols)");
  auto& context = cuda_executor_from(X.executor(), "cuda_cholesky_qr_in_place").context();

  DeviceDoubleBuffer d_gram;
  DeviceDoubleBuffer d_work;
  DeviceIntBuffer d_info;
  d_gram.ensure(static_cast<std::size_t>(b) * static_cast<std::size_t>(b));
  d_info.allocate();

  const double one = 1.0;
  const double zero = 0.0;

  FASTEIG_CUBLAS_CHECK(cublasDgemm(context.cublas_handle(),
                                   CUBLAS_OP_T, CUBLAS_OP_N,
                                   b, b, n,
                                   &one,
                                   X.data(), n,
                                   X.data(), n,
                                   &zero,
                                   d_gram.ptr, b));

  int lwork = 0;
  FASTEIG_CUSOLVER_CHECK(cusolverDnDpotrf_bufferSize(context.cusolver_handle(),
                                                     CUBLAS_FILL_MODE_UPPER,
                                                     b, d_gram.ptr, b, &lwork));
  d_work.ensure(static_cast<std::size_t>(lwork));
  FASTEIG_CUSOLVER_CHECK(cusolverDnDpotrf(context.cusolver_handle(),
                                          CUBLAS_FILL_MODE_UPPER,
                                          b, d_gram.ptr, b,
                                          d_work.ptr, lwork, d_info.ptr));
  check_cusolver_info(context, d_info.ptr, "cuda_cholesky_qr_in_place(potrf)");

  FASTEIG_CUBLAS_CHECK(cublasDtrsm(context.cublas_handle(),
                                   CUBLAS_SIDE_RIGHT,
                                   CUBLAS_FILL_MODE_UPPER,
                                   CUBLAS_OP_N,
                                   CUBLAS_DIAG_NON_UNIT,
                                   n, b,
                                   &one,
                                   d_gram.ptr, b,
                                   X.data(), n));

  R.resize(b, b);
  FASTEIG_CUDA_CHECK(cudaMemcpyAsync(R.data(), d_gram.ptr,
                                     static_cast<std::size_t>(b) * static_cast<std::size_t>(b) *
                                         sizeof(double),
                                     cudaMemcpyDeviceToHost,
                                     context.stream()));
  FASTEIG_CUDA_CHECK(cudaStreamSynchronize(context.stream()));
  R.template triangularView<Eigen::StrictlyLower>().setZero();
  return 0;
}

FASTEIG_INT cuda_gram_schmidt_qr_in_place(CudaMultiVector& X, MatrixXd& R, int passes = 2) {
  validate_multivector_qr(X, "cuda_gram_schmidt_qr_in_place");
  if (passes <= 0) {
    throw std::invalid_argument("cuda_gram_schmidt_qr_in_place: passes must be positive.");
  }

  const int n = checked_int(X.rows(), "cuda_gram_schmidt_qr_in_place(rows)");
  const int b = checked_int(X.cols(), "cuda_gram_schmidt_qr_in_place(cols)");
  auto& context = cuda_executor_from(X.executor(), "cuda_gram_schmidt_qr_in_place").context();

  R.setZero(b, b);
  for (int j = 0; j < b; ++j) {
    double* d_col_j = X.data() + static_cast<std::size_t>(j) * static_cast<std::size_t>(n);
    for (int pass = 0; pass < passes; ++pass) {
      for (int i = 0; i < j; ++i) {
        const double* d_col_i = X.data() + static_cast<std::size_t>(i) * static_cast<std::size_t>(n);
        double h = 0.0;
        FASTEIG_CUBLAS_CHECK(cublasDdot(context.cublas_handle(), n,
                                        d_col_i, 1, d_col_j, 1, &h));
        R(i, j) += h;
        const double minus_h = -h;
        FASTEIG_CUBLAS_CHECK(cublasDaxpy(context.cublas_handle(), n,
                                         &minus_h, d_col_i, 1, d_col_j, 1));
      }
    }

    double r = 0.0;
    FASTEIG_CUBLAS_CHECK(cublasDnrm2(context.cublas_handle(), n, d_col_j, 1, &r));
    if (!std::isfinite(r) || r <= std::numeric_limits<double>::epsilon()) {
      return -1;
    }
    R(j, j) = r;
    const double inv_r = 1.0 / r;
    FASTEIG_CUBLAS_CHECK(cublasDscal(context.cublas_handle(), n, &inv_r, d_col_j, 1));
  }
  FASTEIG_CUDA_CHECK(cudaStreamSynchronize(context.stream()));
  return 0;
}
#endif

}  // namespace

FASTEIG_INT cholesky_qr(const Eigen::Ref<const MatrixXd>& X,
                        MatrixXd& Q,
                        MatrixXd& R) {
  validate_tall_skinny(X, "cholesky_qr");

  const int ncols = static_cast<int>(X.cols());
  const MatrixXd gram = X.transpose() * X;
  Eigen::LLT<MatrixXd> llt(gram);
  if (llt.info() != Eigen::Success) {
    return -1;
  }

  const MatrixXd L = llt.matrixL();
  R = L.transpose();
  Q = X * R.triangularView<Eigen::Upper>().solve(MatrixXd::Identity(ncols, ncols));
  return 0;
}

FASTEIG_INT cholesky_qr2(const Eigen::Ref<const MatrixXd>& X,
                         MatrixXd& Q,
                         MatrixXd& R) {
  validate_tall_skinny(X, "cholesky_qr2");

  MatrixXd q1;
  MatrixXd r1;
  FASTEIG_INT info = cholesky_qr(X, q1, r1);
  if (info != 0) return info;

  MatrixXd r2;
  info = cholesky_qr(q1, Q, r2);
  if (info != 0) return info;

  R = r2 * r1;
  return 0;
}

FASTEIG_INT gram_schmidt_qr(const Eigen::Ref<const MatrixXd>& X,
                            MatrixXd& Q,
                            MatrixXd& R,
                            int passes) {
  validate_tall_skinny(X, "gram_schmidt_qr");
  if (passes <= 0) {
    throw std::invalid_argument("gram_schmidt_qr: passes must be positive.");
  }

  const int ncols = static_cast<int>(X.cols());
  Q.resize(X.rows(), X.cols());
  R.setZero(ncols, ncols);

  for (int j = 0; j < ncols; ++j) {
    VectorXd v = X.col(j);
    for (int pass = 0; pass < passes; ++pass) {
      for (int i = 0; i < j; ++i) {
        const double h = Q.col(i).dot(v);
        R(i, j) += h;
        v.noalias() -= h * Q.col(i);
      }
    }

    const double r = v.norm();
    if (!std::isfinite(r) || r <= std::numeric_limits<double>::epsilon()) {
      return -1;
    }
    R(j, j) = r;
    Q.col(j).noalias() = v / r;
  }

  return 0;
}

FASTEIG_INT householder_qr(const Eigen::Ref<const MatrixXd>& X,
                           MatrixXd& Q,
                           MatrixXd& R) {
  validate_tall_skinny(X, "householder_qr");

  const int nrows = static_cast<int>(X.rows());
  const int ncols = static_cast<int>(X.cols());
  Eigen::HouseholderQR<MatrixXd> qr(X);
  Q = qr.householderQ() * MatrixXd::Identity(nrows, ncols);
  R = MatrixXd::Zero(ncols, ncols);
  R.template triangularView<Eigen::Upper>() =
      qr.matrixQR().topLeftCorner(ncols, ncols)
          .template triangularView<Eigen::Upper>();
  return 0;
}

FASTEIG_INT svqb(MatrixXd& U, double tau, char method) {
  if (U.rows() <= 0 || U.cols() <= 0) {
    throw std::invalid_argument("svqb: input matrix must be nonempty.");
  }
  if (tau <= 0.0) {
    throw std::invalid_argument("svqb: tau must be positive.");
  }
  if (method != 'R' && method != 'D') {
    throw std::invalid_argument("svqb: method must be 'R' or 'D'.");
  }

  MatrixXd W = U.transpose() * U;
  VectorXd d = W.diagonal();
  if ((d.array() <= 0.0).any()) return -1;
  d = d.array().sqrt().inverse();
  const auto D = d.asDiagonal();
  W.applyOnTheLeft(D);
  W.applyOnTheRight(D);

  Eigen::SelfAdjointEigenSolver<MatrixXd> solver(W);
  if (solver.info() != Eigen::Success) return -1;

  VectorXd theta = solver.eigenvalues();
  const MatrixXd Z = solver.eigenvectors();
  const double threshold = tau * theta.lpNorm<Eigen::Infinity>();
  const double replacement = (method == 'R') ? threshold : 0.0;
  theta = (theta.array() < threshold).select(replacement, theta);
  theta = theta.array().sqrt().inverse();
  theta = theta.array().isInf().select(0.0, theta);

  U.applyOnTheRight(D);
  U.applyOnTheRight(Z);
  U.applyOnTheRight(theta.asDiagonal());
  return 0;
}

FASTEIG_INT thin_qr_in_place(MultiVector& X, MatrixXd& R, QrMethod method) {
  validate_multivector_qr(X, "thin_qr_in_place");

  if (auto* cpu_x = dynamic_cast<CpuMultiVector*>(&X)) {
    MatrixXd Q;
    const FASTEIG_INT info = host_thin_qr(cpu_x->values(), Q, R, method);
    if (info == 0) cpu_x->values() = std::move(Q);
    return info;
  }

#ifdef USE_CUDA
  if (auto* cuda_x = dynamic_cast<CudaMultiVector*>(&X)) {
    if (method == QrMethod::Householder) {
      return cuda_householder_qr_in_place(*cuda_x, R);
    }
    if (method == QrMethod::Cholesky) {
      return cuda_cholesky_qr_in_place(*cuda_x, R);
    }
    if (method == QrMethod::Cholesky2) {
      MatrixXd r1;
      MatrixXd r2;
      FASTEIG_INT info = cuda_cholesky_qr_in_place(*cuda_x, r1);
      if (info != 0) return info;
      info = cuda_cholesky_qr_in_place(*cuda_x, r2);
      if (info != 0) return info;
      R = r2 * r1;
      return 0;
    }
    if (method == QrMethod::GramSchmidt) {
      return cuda_gram_schmidt_qr_in_place(*cuda_x, R);
    }

    MatrixXd host_values = cuda_x->copy_to_host();
    MatrixXd Q;
    const FASTEIG_INT info = host_thin_qr(host_values, Q, R, method);
    if (info == 0) cuda_x->copy_from_host(Q);
    return info;
  }
#endif

  throw std::invalid_argument("thin_qr_in_place: unsupported multivector backend or QR method.");
}

FASTEIG_INT block_gram_schmidt(const MultiVector& basis,
                               FASTEIG_INT active_cols,
                               const MultiVector& source,
                               MultiVector& residual,
                               MatrixXd& H) {
  if (&basis.executor() != &source.executor() ||
      &basis.executor() != &residual.executor()) {
    throw std::invalid_argument("block_gram_schmidt: executor mismatch.");
  }
  if (active_cols < 0 || active_cols > basis.cols()) {
    throw std::invalid_argument("block_gram_schmidt: invalid active column count.");
  }
  if (source.rows() != basis.rows() || residual.rows() != basis.rows() ||
      source.cols() != residual.cols()) {
    throw std::invalid_argument("block_gram_schmidt: dimension mismatch.");
  }

  H.setZero(static_cast<int>(active_cols), static_cast<int>(source.cols()));
  residual.copy_from(source);
  if (active_cols == 0) return 0;

  if (const auto* cpu_basis = dynamic_cast<const CpuMultiVector*>(&basis)) {
    const auto* cpu_source = dynamic_cast<const CpuMultiVector*>(&source);
    auto* cpu_residual = dynamic_cast<CpuMultiVector*>(&residual);
    if (!cpu_source || !cpu_residual) {
      throw std::invalid_argument("block_gram_schmidt: CPU object mismatch.");
    }

    const auto V = cpu_basis->values().leftCols(static_cast<int>(active_cols));
    H.noalias() = V.transpose() * cpu_source->values();
    cpu_residual->values().noalias() = cpu_source->values() - V * H;

    MatrixXd S = V.transpose() * cpu_residual->values();
    H.noalias() += S;
    cpu_residual->values().noalias() -= V * S;
    return 0;
  }

#ifdef USE_CUDA
  if (const auto* cuda_basis = dynamic_cast<const CudaMultiVector*>(&basis)) {
    const auto* cuda_source = dynamic_cast<const CudaMultiVector*>(&source);
    auto* cuda_residual = dynamic_cast<CudaMultiVector*>(&residual);
    if (!cuda_source || !cuda_residual) {
      throw std::invalid_argument("block_gram_schmidt: CUDA object mismatch.");
    }

    const int n = checked_int(basis.rows(), "block_gram_schmidt(rows)");
    const int k = checked_int(active_cols, "block_gram_schmidt(active_cols)");
    const int b = checked_int(source.cols(), "block_gram_schmidt(block_cols)");
    auto& context = cuda_executor_from(basis.executor(), "block_gram_schmidt").context();

    DeviceDoubleBuffer d_h;
    DeviceDoubleBuffer d_s;
    const std::size_t coeff_count =
        static_cast<std::size_t>(k) * static_cast<std::size_t>(b);
    d_h.ensure(coeff_count);
    d_s.ensure(coeff_count);

    const double one = 1.0;
    const double zero = 0.0;
    const double minus_one = -1.0;

    FASTEIG_CUBLAS_CHECK(cublasDgemm(context.cublas_handle(),
                                     CUBLAS_OP_T, CUBLAS_OP_N,
                                     k, b, n,
                                     &one,
                                     cuda_basis->data(), n,
                                     cuda_source->data(), n,
                                     &zero,
                                     d_h.ptr, k));
    FASTEIG_CUBLAS_CHECK(cublasDcopy(context.cublas_handle(),
                                     n * b,
                                     cuda_source->data(), 1,
                                     cuda_residual->data(), 1));
    FASTEIG_CUBLAS_CHECK(cublasDgemm(context.cublas_handle(),
                                     CUBLAS_OP_N, CUBLAS_OP_N,
                                     n, b, k,
                                     &minus_one,
                                     cuda_basis->data(), n,
                                     d_h.ptr, k,
                                     &one,
                                     cuda_residual->data(), n));
    FASTEIG_CUBLAS_CHECK(cublasDgemm(context.cublas_handle(),
                                     CUBLAS_OP_T, CUBLAS_OP_N,
                                     k, b, n,
                                     &one,
                                     cuda_basis->data(), n,
                                     cuda_residual->data(), n,
                                     &zero,
                                     d_s.ptr, k));
    FASTEIG_CUBLAS_CHECK(cublasDaxpy(context.cublas_handle(),
                                     k * b,
                                     &one,
                                     d_s.ptr, 1,
                                     d_h.ptr, 1));
    FASTEIG_CUBLAS_CHECK(cublasDgemm(context.cublas_handle(),
                                     CUBLAS_OP_N, CUBLAS_OP_N,
                                     n, b, k,
                                     &minus_one,
                                     cuda_basis->data(), n,
                                     d_s.ptr, k,
                                     &one,
                                     cuda_residual->data(), n));

    H.resize(k, b);
    FASTEIG_CUDA_CHECK(cudaMemcpyAsync(H.data(), d_h.ptr,
                                       coeff_count * sizeof(double),
                                       cudaMemcpyDeviceToHost,
                                       context.stream()));
    FASTEIG_CUDA_CHECK(cudaStreamSynchronize(context.stream()));
    return 0;
  }
#endif

  throw std::invalid_argument("block_gram_schmidt: unsupported multivector backend.");
}

FASTEIG_INT orthogonalize_block_against_basis(MultiVector& basis,
                                              FASTEIG_INT active_cols,
                                              const MultiVector& source,
                                              MultiVector& residual,
                                              MatrixXd& H,
                                              MatrixXd& R,
                                              QrMethod qr_method) {
  FASTEIG_INT info = block_gram_schmidt(basis, active_cols, source, residual, H);
  if (info != 0) return info;
  return thin_qr_in_place(residual, R, qr_method);
}

}  // namespace core
}  // namespace fasteig

