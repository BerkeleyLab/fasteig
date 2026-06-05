#pragma once

#include "fasteig.hpp"
#include "fasteig_logger.hpp"
#include "fasteig_options.hpp"
#include "fasteig_solver.hpp"

#include <functional>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

class fasteig_problem {
public:
  // Construct from an existing sparse matrix
  explicit fasteig_problem(SpMat H, std::string solver = "Lanczos");

  // Construct from user-provided operators (matrix-free)
  explicit fasteig_problem(operations ops, FASTEIG_INT n, std::string solver = "Lanczos");

  // Construct from parsed input parameters
  explicit fasteig_problem(const fasteig_params& p);

  // Solve with default options
  FASTEIG_INT solve();

  // Solve with user-provided options
  FASTEIG_INT solve(const fasteig_options& opts);

  // Results access
  VectorXd& eigenvalues() noexcept { return eigvals_; }
  const VectorXd& eigenvalues() const noexcept { return eigvals_; }
  VectorXd& resnormual_norms() noexcept { return resnrms_; }
  const VectorXd& resnormual_norms() const noexcept { return resnrms_; }
  VectorXd& residual_norms() noexcept { return resnrms_; }
  const VectorXd& residual_norms() const noexcept { return resnrms_; }
  MatrixXd& eigenvectors() noexcept { return eigvecs_; }
  const MatrixXd& eigenvectors() const noexcept { return eigvecs_; }
  FASTEIG_INT status() const noexcept { return status_; }

  FASTEIG_INT n() const noexcept { return n_; }
  const std::string& solver() const noexcept { return solver_; }

  operations& ops() noexcept { return opsA_; }
  const operations& ops() const noexcept { return opsA_; }

  operations& opsA() noexcept { return opsA_; }
  const operations& opsA() const noexcept { return opsA_; }
  operations& opsB() noexcept { return opsB_; }
  const operations& opsB() const noexcept { return opsB_; }
  operations& opsSAI() noexcept { return opsSAI_; }
  const operations& opsSAI() const noexcept { return opsSAI_; }
  operations& solver_ops(const fasteig_options& opts) noexcept;
  const operations& solver_ops(const fasteig_options& opts) const noexcept;

  const SpMat* matrix() const noexcept { return H_.has_value() ? &(*H_) : nullptr; }
  const SpMat* bmatrix() const noexcept { return B_.has_value() ? &(*B_) : nullptr; }
  const fasteig_params* params() const noexcept { return params_.has_value() ? &params_->get() : nullptr; }
  bool is_generalized() const noexcept { return B_.has_value(); }

  // Generalized eigenproblem setup (currently: factorize B once for KrylovSchur).
  void prepare(const fasteig_options& opts);
  std::optional<double> b_factorization_sec() const noexcept { return b_factorization_sec_; }

  // Logging/progress output.
  void open_log(const fasteig_options& opts, const std::string& solver_name = std::string(), bool append = false);
  void set_log_path(const std::string& path, bool append = false);
  void set_log_stream(std::ostream* os) noexcept { logger_.attach(os); }
  void close_log() { logger_.close(); }
  fasteig_logger& logger() noexcept { return logger_; }
  const fasteig_logger& logger() const noexcept { return logger_; }
  std::ostream* log_stream() noexcept { return logger_.stream(); }
  std::ostream* log_stream() const noexcept { return const_cast<fasteig_logger&>(logger_).stream(); }

  // Best-effort iteration count from the most recent solve().
  std::optional<long long> last_iterations() const noexcept { return last_iterations_; }

  // (For SPPC)
  void set_sppc_split(FASTEIG_INT ns);

private:
  static SpMat build_sparse_matrix_from_triplets(const FASTEIG_INT n,
                                                 const FASTEIG_INT nnz,
                                                 const std::vector<FASTEIG_INT>& rowind,
                                                 const std::vector<FASTEIG_INT>& colind,
                                                 const std::vector<FASTEIG_DOUBLE>& nzvals);

  static SpMat build_sparse_matrix_from_RSA(const FASTEIG_INT n,
                                            const FASTEIG_INT nnz,
                                            const std::vector<FASTEIG_INT>& colptr,
                                            const std::vector<FASTEIG_INT>& rowind,
                                            const std::vector<FASTEIG_DOUBLE>& nzvals);

  void build_default_ops_from_H(operations& target_ops);
  void build_default_ops_from_B(operations& target_ops);
  void build_sai_ops_from_H(const fasteig_params& p);
  static bool is_sai_which(const std::string& which);
  void set_sppc_split(operations& target_ops, FASTEIG_INT ns);

private:
  Eigen::MINRES<SpMat> minresSolver_;
  Eigen::ConjugateGradient<SpMat, Eigen::Lower | Eigen::Upper> cg_;

  std::optional<std::reference_wrapper<const fasteig_params>> params_;
  std::optional<SpMat> H_;
  std::optional<SpMat> B_;
  operations opsA_{};
  operations opsB_{};
  operations opsSAI_{};
  FASTEIG_INT n_ = 0;

  FASTEIG_INT ns_ = 0;

  std::string solver_ = "Lanczos";
  bool b_triangle_upper_for_ops_ = true;  // effective stored triangle for B when using selfadjointView
  bool b_factorized_ = false;
  std::optional<double> b_factorization_sec_;

#ifdef EIGEN_USE_MKL_ALL
  Eigen::PardisoLDLT<SpMat, Eigen::Lower> ldlt_;  // MKL PARDISO LDLT
  Eigen::PardisoLDLT<SpMat, Eigen::Lower> ldltB_;  // MKL PARDISO LDLT (for B)
#else
  Eigen::SimplicialLDLT<SpMat, Eigen::Lower> ldlt_;  // Eigen's default LDLT
  Eigen::SimplicialLDLT<SpMat, Eigen::Lower> ldltB_;  // Eigen's default LDLT (for B)
#endif

  VectorXd eigvals_;
  VectorXd resnrms_;
  MatrixXd eigvecs_;
  FASTEIG_INT status_ = 0;

  fasteig_logger logger_;
  std::optional<long long> last_iterations_;
};
