#pragma once

#include "fasteig.hpp"
#include "fasteig_options.hpp"

#include <optional>
#include <ostream>
#include <string>

class fasteig_problem;

VectorXd fasteig_compute_residual_norms(const operations& ops,
                                       const VectorXd& eigvals,
                                       const MatrixXd& eigvecs);

// Generalized residual norms for Ax = λBx:
//   r_i = ||A v_i - λ_i B v_i|| / ((1+|λ_i|) ||B v_i||)
// If opsB is null (or has no matvec), this reduces to the standard form with B=I.
VectorXd fasteig_compute_residual_norms(const operations& opsA,
                                       const operations* opsB,
                                       const VectorXd& eigvals,
                                       const MatrixXd& eigvecs);

// Stdout printing:
//   verbosity=1: eigenpairs table
//   verbosity=2: iteration summary (during solve) + eigenpairs table
void fasteig_print_eigenpairs_table(std::ostream& os,
                                   const VectorXd& eigvals,
                                   const VectorXd& residual_norms);

void fasteig_print_run_header(std::ostream& os,
                             const fasteig_options& opts,
                             const std::string& solver_name,
                             std::optional<double> runtime_sec = std::nullopt);

void fasteig_print_matrix_summary(std::ostream& os,
                                 const fasteig_problem& problem);

// File writing: always writes full information (header + table) whenever called.
void fasteig_write_results_text(const fasteig_options& opts,
                               const std::string& solver_name,
                               FASTEIG_INT status,
                               const VectorXd& eigvals,
                               const VectorXd& residual_norms,
                               std::optional<double> b_factorization_sec = std::nullopt,
                               std::optional<double> runtime_sec = std::nullopt,
                               std::optional<long long> iterations = std::nullopt);

// Append final results to an existing output stream (no file open/close here).
void fasteig_write_results_text(std::ostream& os,
                               FASTEIG_INT status,
                               const VectorXd& eigvals,
                               const VectorXd& residual_norms,
                               std::optional<double> b_factorization_sec = std::nullopt,
                               std::optional<double> runtime_sec = std::nullopt,
                               std::optional<long long> iterations = std::nullopt);

