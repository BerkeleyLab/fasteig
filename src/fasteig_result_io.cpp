#include "fasteig_result_io.hpp"

#include "fasteig_problem.hpp"

#include <algorithm>
#include <cctype>
#include <complex>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <type_traits>

VectorXd fasteig_compute_residual_norms(const operations& ops,
                                       const VectorXd& eigvals,
                                       const MatrixXd& eigvecs) {
  return fasteig_compute_residual_norms(ops, /*opsB=*/nullptr, eigvals, eigvecs);
}

VectorXd fasteig_compute_residual_norms(const operations& opsA,
                                       const operations* opsB,
                                       const VectorXd& eigvals,
                                       const MatrixXd& eigvecs) {
  const int nev = static_cast<int>(eigvals.size());
  if (eigvecs.cols() < nev) {
    throw std::invalid_argument("fasteig_compute_residual_norms: eigvecs has fewer columns than eigvals.");
  }
  if (!opsA.matvec && !opsA.block_matvec) {
    throw std::invalid_argument("fasteig_compute_residual_norms: opsA has no matvec or block_matvec.");
  }

  VectorXd residuals(nev);
  VectorXd av(eigvecs.rows());
  VectorXd bv(eigvecs.rows());

  for (int i = 0; i < nev; ++i) {
    const auto v = eigvecs.col(i);
    if (opsA.matvec) {
      opsA.matvec(v, av);
    } else {
      MatrixXd V(v.rows(), 1);
      V.col(0) = v;
      MatrixXd AV(v.rows(), 1);
      opsA.block_matvec(V, AV);
      av = AV.col(0);
    }

    if (opsB && opsB->matvec) {
      opsB->matvec(v, bv);
      const double num = (av - eigvals(i) * bv).norm();
      const double denom = (1.0 + std::abs(eigvals(i))) * bv.norm();
      residuals(i) = (denom > 0.0) ? (num / denom) : num;
    } else {
      const double num = (av - eigvals(i) * v).norm();
      const double denom = (1.0 + std::abs(eigvals(i))) * v.norm();
      residuals(i) = (denom > 0.0) ? (num / denom) : num;
    }
  }
  return residuals;
}

namespace {

std::string which_pairs_pretty(std::string which) {
  std::transform(which.begin(), which.end(), which.begin(),
                 [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
  if (which == "SR") return "Smallest Real";
  if (which == "LR") return "Largest Real";
  if (which == "SM") return "Smallest Magnitude";
  if (which == "LM") return "Largest Magnitude";
  if (which == "BE") return "Both Ends";
  return which;
}

}  // namespace

void fasteig_print_run_header(std::ostream& os,
                             const fasteig_options& opts,
                             const std::string& solver_name,
                             std::optional<double> runtime_sec) {
  auto print_kv = [&](const char* key, const std::string& value) {
    constexpr int kKeyWidth = 22;
    os << "  " << std::left << std::setw(kKeyWidth) << key << ": " << value << "\n";
  };

  auto lower_copy = [](std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
  };

  const std::string solver_l = lower_copy(solver_name);

  os << "\n";
  os << "RUN CONFIGURATION\n";
  os << "=================\n";
  os << "\n";

  print_kv("Eigensolver", solver_name);
  print_kv("Number of Eigenpairs", std::to_string(opts.nev()));
  print_kv("Which Pairs", which_pairs_pretty(opts.which()));
  {
    std::ostringstream ss;
    ss << std::setprecision(16) << opts.tol();
    print_kv("Tolerance", ss.str());
  }

  // Common configuration knobs (shown when meaningful).
  if (opts.maxiter() > 0) print_kv("Max Iterations", std::to_string(opts.maxiter()));
  if (opts.print_every() > 0) print_kv("Print Every", std::to_string(opts.print_every()));
  print_kv("Verbosity", std::to_string(opts.verbosity()));

  if (opts.blksize() > 1) print_kv("Block Size", std::to_string(opts.blksize()));
  if (opts.maxdim() > 0) print_kv("Max Subspace Dim", std::to_string(opts.maxdim()));
  if (solver_l == "davidson" && opts.maxblks() > 0) {
    print_kv("Max Blocks", std::to_string(opts.maxblks()));
  }
  if (solver_l == "chebyshev" && opts.degree() > 0) {
    print_kv("Degree", std::to_string(opts.degree()));
  }
  if (opts.shift() != 0.0) {
    std::ostringstream ss;
    ss << std::setprecision(16) << opts.shift();
    print_kv("Shift", ss.str());
  }

  if (runtime_sec.has_value()) {
    os << "\n";
    os << "Runtime (sec): " << std::fixed << std::setprecision(6) << *runtime_sec << "\n";
  }

  os << "\n";
}

namespace {

const char* matrix_format_name(MatrixFormat fmt) {
  switch (fmt) {
    case MatrixFormat::MTX: return "MTX";
    case MatrixFormat::RSA: return "RSA";
    case MatrixFormat::FEI: return "FEI";
    case MatrixFormat::HDF: return "HDF";
    case MatrixFormat::UNKNOWN: return "UNKNOWN";
  }
  return "UNKNOWN";
}

}  // namespace

void fasteig_print_matrix_summary(std::ostream& os,
                                 const fasteig_problem& problem) {
  const fasteig_params* p = problem.params();
  const SpMat* A = problem.matrix();
  const SpMat* B = problem.bmatrix();

  auto print_kv = [&](const char* key, const std::string& value) {
    constexpr int kKeyWidth = 22;
    os << "  " << std::left << std::setw(kKeyWidth) << key << ": " << value << "\n";
  };

  os << "\n";
  os << "PROBLEM SUMMARY\n";
  os << "===============\n";
  os << "\n";

  if (p) {
    print_kv("Matrix Format", matrix_format_name(p->matrixformat));

    if (B) {
      print_kv("Problem Type", "Generalized (A,B)");
    } else {
      print_kv("Problem Type", "Standard (B=I)");
    }

    if (!p->amatfname.empty()) {
      const std::filesystem::path ap(p->amatfname);
      const std::string name = ap.filename().empty() ? p->amatfname : ap.filename().string();
      print_kv("A Matrix Name", name);
    }

    if (B) {
      if (!p->bmatfname.empty()) {
        const std::filesystem::path bp(p->bmatfname);
        const std::string name = bp.filename().empty() ? p->bmatfname : bp.filename().string();
        print_kv("B Matrix Name", name);
      } else {
        print_kv("B Matrix Name", "IN_MEMORY");
      }
    }
  } else {
    print_kv("Matrix Format", (A ? "IN_MEMORY" : "MATRIX_FREE"));
    if (B) {
      print_kv("Problem Type", "Generalized (A,B)");
    } else {
      print_kv("Problem Type", "Standard (B=I)");
    }
    if (A) print_kv("A Matrix Name", "IN_MEMORY");
    if (B) print_kv("B Matrix Name", "IN_MEMORY");
  }

  constexpr bool is_complex =
      std::is_same_v<FASTEIG_DOUBLE, std::complex<float>> ||
      std::is_same_v<FASTEIG_DOUBLE, std::complex<double>> ||
      std::is_same_v<FASTEIG_DOUBLE, std::complex<long double>>;
  print_kv("Scalar Type", (is_complex ? "Complex" : "Real"));

  // Symmetry (best-effort; only meaningful for an explicit square matrix).
  std::string symmetric = "No";
  if (p && p->matrixformat == MatrixFormat::MTX && p->mtx_symmetric) {
    symmetric = "Yes";
  } else if (A && A->rows() == A->cols() && A->rows() > 0) {
    // Use a modest tolerance to account for roundoff in stored values.
    constexpr double kSymTol = 1e-12;
    symmetric = A->isApprox(A->transpose(), kSymTol) ? "Yes" : "No";
  }
  print_kv("Symmetric", symmetric);

  // Prefer explicit matrix dimension when available.
  if (A) {
    print_kv("Dimension", std::to_string(A->rows()));
  } else {
    // Fall back to params if a matrix object is not built.
    if (p && p->n > 0) print_kv("Dimension", std::to_string(p->n));
  }

  // Nonzeros (if known).
  std::optional<long long> nnz_full;
  std::optional<long long> nnz_stored;
  if (p && p->matrixformat == MatrixFormat::MTX && (p->mtx_symmetric || p->mtx_skew_symmetric)) {
    nnz_stored = static_cast<long long>(p->annz);
    nnz_full = static_cast<long long>(p->mtx_full_nnz());
  } else if (A) {
    nnz_full = static_cast<long long>(A->nonZeros());
  } else if (p && p->annz > 0) {
    nnz_full = static_cast<long long>(p->annz);
  }

  if (nnz_full.has_value()) {
    print_kv("Number of Nonzeros", std::to_string(*nnz_full));
    if (nnz_stored.has_value() && *nnz_stored != *nnz_full) {
      print_kv("Stored Nonzeros", std::to_string(*nnz_stored));
    }
  }

  // Density (if we know both n and nnz).
  if (p && p->n > 0 && nnz_full.has_value()) {
    const double n = static_cast<double>(p->n);
    const double nnz = static_cast<double>(*nnz_full);
    std::ostringstream ss;
    ss << std::scientific << std::setprecision(6) << (nnz / (n * n));
    print_kv("Density", ss.str());
  } else if (A && A->rows() > 0) {
    const double n = static_cast<double>(A->rows());
    const double nnz = static_cast<double>(A->nonZeros());
    std::ostringstream ss;
    ss << std::scientific << std::setprecision(6) << (nnz / (n * n));
    print_kv("Density", ss.str());
  }

  os << "\n";
}

void fasteig_print_eigenpairs_table(std::ostream& os,
                                   const VectorXd& eigvals,
                                   const VectorXd& residual_norms) {
  const int nev = static_cast<int>(eigvals.size());

  const int eig_w = 7;     // "Index"
  const int val_w = 18;
  const int res_w = 14;

  os << std::left << std::setw(eig_w) << "Index"
     << "  " << std::right << std::setw(val_w) << "Eigenvalue"
     << "  " << std::right << std::setw(res_w) << "Residual Norm"
     << "\n";

  os << std::setprecision(6) << std::scientific;
  for (int i = 0; i < nev; ++i) {
    const double r = (i < residual_norms.size()) ? residual_norms(i) : 0.0;
    os << std::left << std::setw(eig_w) << (i + 1)
       << "  " << std::right << std::setw(val_w) << eigvals(i)
       << "  " << std::right << std::setw(res_w) << r
       << "\n";
  }
}

void fasteig_write_results_text(const fasteig_options& opts,
                               const std::string& solver_name,
                               FASTEIG_INT status,
                               const VectorXd& eigvals,
                               const VectorXd& residual_norms,
                               std::optional<double> b_factorization_sec,
                               std::optional<double> runtime_sec,
                               std::optional<long long> iterations) {
  if (opts.output_format() != "text") {
    throw std::invalid_argument("fasteig_write_results_text: output_format must be 'text'.");
  }
  if (opts.output_prefix().empty()) return;

  const std::string path = opts.output_path(solver_name);
  std::ofstream f(path, std::ios::out | std::ios::app);
  if (!f) throw std::runtime_error("Failed to open output file: " + path);
  fasteig_write_results_text(f, status, eigvals, residual_norms, b_factorization_sec, runtime_sec, iterations);
}

void fasteig_write_results_text(std::ostream& os,
                               FASTEIG_INT status,
                               const VectorXd& eigvals,
                               const VectorXd& residual_norms,
                               std::optional<double> b_factorization_sec,
                               std::optional<double> runtime_sec,
                               std::optional<long long> iterations) {
  // The run header and progress header are written by the driver.
  // Append only end-of-run diagnostics and the final eigenpairs table here.
  os << "\n";
  os << "FINAL RESULTS\n";
  os << "=============\n";
  os << "\n";
  fasteig_print_eigenpairs_table(os, eigvals, residual_norms);

  // Machine-friendly block with full precision (used by the testing harness).
  // Keep the human-readable table above at 6 digits for readability.
  os << "\n";
  os << "RAW RESULTS (FULL PRECISION)\n";
  os << "============================\n";
  os << "\n";
  os << "Index Eigenvalue ResidualNorm\n";
  os << std::setprecision(std::numeric_limits<double>::max_digits10) << std::scientific;
  const int nev = static_cast<int>(eigvals.size());
  for (int i = 0; i < nev; ++i) {
    const double r = (i < residual_norms.size()) ? residual_norms(i) : 0.0;
    os << (i + 1) << " " << eigvals(i) << " " << r << "\n";
  }

  os << "\n";
  os << "DIAGNOSTICS\n";
  os << "===========\n";
  os << "\n";
  os << "  Status               : " << status << "\n";
  if (iterations.has_value()) {
    os << "  Iterations           : " << *iterations << "\n";
  }
  if (b_factorization_sec.has_value()) {
    os << "  B Factorization (sec): " << std::fixed << std::setprecision(6) << *b_factorization_sec << "\n";
  }
  if (runtime_sec.has_value()) {
    os << "  Runtime (sec)        : " << std::fixed << std::setprecision(6) << *runtime_sec << "\n";
  }
}

