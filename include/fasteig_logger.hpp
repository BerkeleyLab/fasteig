#pragma once

#include "fasteig.hpp"
#include "fasteig_options.hpp"

#include <fstream>
#include <optional>
#include <ostream>
#include <string>
#include <string_view>

class fasteig_problem;

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

  // Command-line reports use the same attached stream as solver output.
  void print_version();
  void print_build_info();
  void print_usage(std::string_view program);

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

  void print_factorization_setup(const std::string& backend,
                                 const std::string& matrix,
                                 std::optional<double> shift,
                                 double analysis_sec,
                                 double factorization_sec,
                                 double setup_sec,
                                 std::optional<double> matrix_setup_sec = std::nullopt);

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

