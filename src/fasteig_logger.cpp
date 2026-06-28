#include "fasteig_logger.hpp"

#include "fasteig_problem.hpp"

#include <algorithm>
#include <cctype>
#include <complex>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <type_traits>

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

std::string lower_copy(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

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

fasteig_logger::fasteig_logger(const std::string& path, bool append) {
  open(path, append);
}

void fasteig_logger::attach(std::ostream* os) noexcept {
  if (file_.is_open()) file_.close();
  path_.clear();
  external_ = os;
}

void fasteig_logger::open(const std::string& path, bool append) {
  close();
  path_ = path;
  if (path_.empty()) return;

  const std::filesystem::path p(path_);
  const auto parent = p.parent_path();
  if (!parent.empty()) {
    std::error_code ec;
    std::filesystem::create_directories(parent, ec);
  }

  const auto mode = append ? (std::ios::out | std::ios::app) : std::ios::out;
  file_.open(path_, mode);
  if (!file_.is_open()) {
    throw std::runtime_error("fasteig_logger: unable to open log file: " + path_);
  }
}

void fasteig_logger::close() {
  if (file_.is_open()) file_.close();
  external_ = nullptr;
  path_.clear();
}

void fasteig_logger::flush() {
  if (auto* os = stream()) os->flush();
}

bool fasteig_logger::enabled() const noexcept {
  return external_ != nullptr || file_.is_open();
}

std::ostream* fasteig_logger::stream() noexcept {
  if (external_) return external_;
  return file_.is_open() ? &file_ : nullptr;
}

const std::ostream* fasteig_logger::stream() const noexcept {
  if (external_) return external_;
  return file_.is_open() ? &file_ : nullptr;
}

void fasteig_logger::line(std::string_view text) {
  if (auto* os = stream()) {
    *os << text << "\n";
  }
}

void fasteig_logger::section(std::string_view title) {
  if (auto* os = stream()) {
    *os << "\n" << title << "\n";
    *os << std::string(title.size(), '=') << "\n\n";
  }
}

void fasteig_logger::print_progress_note(const std::string& message, int iter) {
  auto* os = stream();
  if (!os) return;
  if (iter >= 0) {
    *os << "Iteration " << iter << " | ";
  }
  *os << message << "\n";
}

void fasteig_logger::print_locking_note(int eig_idx, int iter) {
  print_progress_note("Locking Eigenvector " + std::to_string(eig_idx), iter);
}

void fasteig_logger::print_matrix_summary(const fasteig_problem& problem) {
  auto* os = stream();
  if (!os) return;

  const fasteig_params* p = problem.params();
  const SpMat* A = problem.matrix();
  const SpMat* B = problem.bmatrix();

  auto print_kv = [&](const char* key, const std::string& value) {
    constexpr int kKeyWidth = 22;
    *os << "  " << std::left << std::setw(kKeyWidth) << key << ": " << value << "\n";
  };

  *os << "\n";
  *os << "PROBLEM SUMMARY\n";
  *os << "===============\n";
  *os << "\n";

  if (p) {
    print_kv("Matrix Format", matrix_format_name(p->matrixformat));
    print_kv("Problem Type", B ? "Generalized (A,B)" : "Standard (B=I)");

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
    print_kv("Problem Type", B ? "Generalized (A,B)" : "Standard (B=I)");
    if (A) print_kv("A Matrix Name", "IN_MEMORY");
    if (B) print_kv("B Matrix Name", "IN_MEMORY");
  }

  constexpr bool is_complex =
      std::is_same_v<FASTEIG_DOUBLE, std::complex<float>> ||
      std::is_same_v<FASTEIG_DOUBLE, std::complex<double>> ||
      std::is_same_v<FASTEIG_DOUBLE, std::complex<long double>>;
  print_kv("Scalar Type", (is_complex ? "Complex" : "Real"));

  std::string symmetric = "No";
  if (p && p->matrixformat == MatrixFormat::MTX && p->mtx_symmetric) {
    symmetric = "Yes";
  } else if (A && A->rows() == A->cols() && A->rows() > 0) {
    constexpr double kSymTol = 1e-12;
    symmetric = A->isApprox(A->transpose(), kSymTol) ? "Yes" : "No";
  }
  print_kv("Symmetric", symmetric);

  if (A) {
    print_kv("Dimension", std::to_string(A->rows()));
  } else if (p && p->n > 0) {
    print_kv("Dimension", std::to_string(p->n));
  }

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

  *os << "\n";
}

void fasteig_logger::print_run_header(const fasteig_options& opts,
                                      const std::string& solver_name,
                                      std::optional<double> runtime_sec) {
  auto* os = stream();
  if (!os) return;

  auto print_kv = [&](const char* key, const std::string& value) {
    constexpr int kKeyWidth = 22;
    *os << "  " << std::left << std::setw(kKeyWidth) << key << ": " << value << "\n";
  };

  const std::string solver_l = lower_copy(solver_name);

  *os << "\n";
  *os << "RUN CONFIGURATION\n";
  *os << "=================\n";
  *os << "\n";

  print_kv("Eigensolver", solver_name);
  print_kv("Number of Eigenpairs", std::to_string(opts.nev()));
  print_kv("Which Pairs", which_pairs_pretty(opts.which()));
  {
    std::ostringstream ss;
    ss << std::setprecision(16) << opts.tol();
    print_kv("Tolerance", ss.str());
  }

  if (opts.maxiter() > 0) print_kv("Max Iterations", std::to_string(opts.maxiter()));
  if (opts.print_every() > 0) print_kv("Print Every", std::to_string(opts.print_every()));
  print_kv("Verbosity", std::to_string(opts.verbosity()));
  if (opts.gpu_backend()) print_kv("GPU Backend", "CUDA Executor");

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
    *os << "\n";
    *os << "Runtime (sec): " << std::fixed << std::setprecision(6) << *runtime_sec << "\n";
  }

  *os << "\n";
}

void fasteig_logger::print_convergence_data(double ritz_val,
                                            double residual,
                                            int iter) {
  std::ostringstream ss;
  ss << "Ritz Value = "
     << std::setw(22) << std::scientific << std::setprecision(15) << ritz_val
     << "\tResidual = "
     << std::setw(10) << std::setprecision(3) << residual;
  print_progress_note(ss.str(), iter);
}

void fasteig_logger::print_convergence_data(const Eigen::Ref<const VectorXd> ritz_vals,
                                            const Eigen::Ref<const VectorXd> residuals,
                                            int iter) {
  auto* os = stream();
  if (!os) return;
  const int n = static_cast<int>(ritz_vals.size());
  if (n != residuals.size()) {
    throw std::invalid_argument("print_convergence_data: Ritz values and residuals have different lengths.");
  }

  print_progress_note("Ritz Values & Residuals", iter);
  for (int i = 0; i < n; ++i) {
    std::ostringstream ss;
    ss << "Eig " << (i + 1) << " | "
       << std::setw(22) << std::scientific << std::setprecision(15) << ritz_vals(i)
       << "\t"
       << std::setw(10) << std::setprecision(3) << residuals(i);
    print_progress_note(ss.str(), -1);
  }
}

void fasteig_logger::print_eigenpairs_table(const VectorXd& eigvals,
                                            const VectorXd& residual_norms) {
  auto* os = stream();
  if (!os) return;

  const int nev = static_cast<int>(eigvals.size());
  const int eig_w = 7;
  const int val_w = 18;
  const int res_w = 14;

  *os << std::left << std::setw(eig_w) << "Index"
      << "  " << std::right << std::setw(val_w) << "Eigenvalue"
      << "  " << std::right << std::setw(res_w) << "Residual Norm"
      << "\n";

  *os << std::setprecision(6) << std::scientific;
  for (int i = 0; i < nev; ++i) {
    const double r = (i < residual_norms.size()) ? residual_norms(i) : 0.0;
    *os << std::left << std::setw(eig_w) << (i + 1)
        << "  " << std::right << std::setw(val_w) << eigvals(i)
        << "  " << std::right << std::setw(res_w) << r
        << "\n";
  }
}

void fasteig_logger::write_results_text(FASTEIG_INT status,
                                        const VectorXd& eigvals,
                                        const VectorXd& residual_norms,
                                        std::optional<double> b_factorization_sec,
                                        std::optional<double> runtime_sec,
                                        std::optional<long long> iterations) {
  print_final_results(status, eigvals, residual_norms,
                      b_factorization_sec, runtime_sec, iterations,
                      /*include_status=*/true,
                      /*include_raw_results=*/true);
}

void fasteig_logger::write_results_text(const fasteig_options& opts,
                                        const std::string& solver_name,
                                        FASTEIG_INT status,
                                        const VectorXd& eigvals,
                                        const VectorXd& residual_norms,
                                        std::optional<double> b_factorization_sec,
                                        std::optional<double> runtime_sec,
                                        std::optional<long long> iterations) {
  if (opts.output_format() != "text") {
    throw std::invalid_argument("fasteig_logger::write_results_text: output_format must be 'text'.");
  }
  if (opts.output_prefix().empty()) return;

  fasteig_logger file_logger(opts.output_path(solver_name), /*append=*/true);
  file_logger.write_results_text(status, eigvals, residual_norms,
                                 b_factorization_sec, runtime_sec, iterations);
}

void fasteig_logger::print_final_results(FASTEIG_INT status,
                                         const VectorXd& eigvals,
                                         const VectorXd& residual_norms,
                                         std::optional<double> b_factorization_sec,
                                         std::optional<double> runtime_sec,
                                         std::optional<long long> iterations,
                                         bool include_status,
                                         bool include_raw_results) {
  auto* os = stream();
  if (!os) return;

  *os << "\n";
  *os << "FINAL RESULTS\n";
  *os << "=============\n";
  *os << "\n";
  print_eigenpairs_table(eigvals, residual_norms);

  if (include_raw_results) {
    *os << "\n";
    *os << "RAW RESULTS (FULL PRECISION)\n";
    *os << "============================\n";
    *os << "\n";
    *os << "Index Eigenvalue ResidualNorm\n";
    *os << std::setprecision(std::numeric_limits<double>::max_digits10) << std::scientific;
    const int nev = static_cast<int>(eigvals.size());
    for (int i = 0; i < nev; ++i) {
      const double r = (i < residual_norms.size()) ? residual_norms(i) : 0.0;
      *os << (i + 1) << " " << eigvals(i) << " " << r << "\n";
    }
  }

  *os << "\n";
  *os << "DIAGNOSTICS\n";
  *os << "===========\n";
  *os << "\n";
  if (include_status) {
    *os << "  Status               : " << status << "\n";
  }
  if (iterations.has_value()) {
    *os << "  Iterations           : " << *iterations << "\n";
  }
  if (b_factorization_sec.has_value()) {
    *os << "  B Factorization (sec): " << std::fixed << std::setprecision(6)
         << *b_factorization_sec << "\n";
  }
  if (runtime_sec.has_value()) {
    *os << "  Runtime (sec)        : " << std::fixed << std::setprecision(6)
         << *runtime_sec << "\n";
  }
  *os << "\n";
}

