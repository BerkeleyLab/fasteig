#pragma once

#include "fasteig.hpp"
#include "fasteig_options.hpp"

#include <fstream>
#include <optional>
#include <ostream>
#include <string>
#include <string_view>

class fasteig_problem;

struct fasteig_gpu_lanczos_timings {
  bool use_gpu_snapshot = true;
  bool use_cpu_orthog = false;
  long long matvec_calls = 0;
  double setup_sec = 0.0;
  double context_sec = 0.0;
  double matrix_setup_sec = 0.0;
  double device_alloc_sec = 0.0;
  double snapshot_setup_sec = 0.0;
  double initial_h2d_sec = 0.0;
  double spmv_sec = 0.0;
  double orthog_sec = 0.0;
  double t_update_sec = 0.0;
  double cpu_orthog_transfer_sec = 0.0;
  double krylov_total_sec = 0.0;
  double gpu_loop_event_sec = 0.0;
  double snapshot_sec = 0.0;
  double snapshot_h2d_sec = 0.0;
  double snapshot_t_d2h_sec = 0.0;
  double snapshot_t_d2d_sec = 0.0;
  double snapshot_kernel_sec = 0.0;
  double snapshot_d2h_sec = 0.0;
  double ritz_h2d_sec = 0.0;
  double ritz_gemm_sec = 0.0;
  double ritz_d2h_sec = 0.0;
  double eigvec_sec = 0.0;
  double transfer_sec = 0.0;
  double residual_sec = 0.0;
};

class fasteig_logger {
public:
  fasteig_logger() = default;
  explicit fasteig_logger(const std::string& path, bool append = false);
  explicit fasteig_logger(std::ostream* os) { attach(os); }

  fasteig_logger(const fasteig_logger&) = delete;
  fasteig_logger& operator=(const fasteig_logger&) = delete;

  fasteig_logger(fasteig_logger&&) noexcept = default;
  fasteig_logger& operator=(fasteig_logger&&) noexcept = default;

  void open(const std::string& path, bool append = false);
  void attach(std::ostream* os) noexcept;
  void close();

  bool enabled() const noexcept;
  const std::string& path() const noexcept { return path_; }

  std::ostream* stream() noexcept;
  const std::ostream* stream() const noexcept;

  void line(std::string_view text = {});
  void section(std::string_view title);
  void flush();

  void print_progress_note(const std::string& message, int iter = -1);
  void print_locking_note(int eig_idx, int iter);

  void print_convergence_data(double ritz_val,
                              double residual,
                              int iter);

  void print_convergence_data(const Eigen::Ref<const VectorXd> ritz_vals,
                              const Eigen::Ref<const VectorXd> residuals,
                              int iter);

  void print_eigenpairs_table(const VectorXd& eigvals,
                              const VectorXd& residual_norms);

  void print_run_header(const fasteig_options& opts,
                        const std::string& solver_name,
                        std::optional<double> runtime_sec = std::nullopt);

  void print_matrix_summary(const fasteig_problem& problem);

  void print_gpu_lanczos_timings(const fasteig_gpu_lanczos_timings& timings);

  void print_final_results(FASTEIG_INT status,
                           const VectorXd& eigvals,
                           const VectorXd& residual_norms,
                           std::optional<double> b_factorization_sec = std::nullopt,
                           std::optional<double> runtime_sec = std::nullopt,
                           std::optional<long long> iterations = std::nullopt,
                           bool include_status = false,
                           bool include_raw_results = false);

  void write_results_text(FASTEIG_INT status,
                          const VectorXd& eigvals,
                          const VectorXd& residual_norms,
                          std::optional<double> b_factorization_sec = std::nullopt,
                          std::optional<double> runtime_sec = std::nullopt,
                          std::optional<long long> iterations = std::nullopt);

  void write_results_text(const fasteig_options& opts,
                          const std::string& solver_name,
                          FASTEIG_INT status,
                          const VectorXd& eigvals,
                          const VectorXd& residual_norms,
                          std::optional<double> b_factorization_sec = std::nullopt,
                          std::optional<double> runtime_sec = std::nullopt,
                          std::optional<long long> iterations = std::nullopt);

private:
  std::ofstream file_;
  std::ostream* external_ = nullptr;
  std::string path_;
};

