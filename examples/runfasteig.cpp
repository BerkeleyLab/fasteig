#include "fasteig.hpp"
#include "fasteig_options.hpp"
#include "fasteig_problem.hpp"
//DMR
#include "fasteig_result_io.hpp"
#include "print_util.hpp"

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>

int main(int argc, const char* argv[])
{
  if (argc < 2) {
    std::cerr << "[ERROR] Usage: " << argv[0] << " <input filename>\n";
    return 1;
  }

  // Parse the input deck (C ABI retained: parseinput(char*, fasteig_params*)).
  fasteig_params params;

  std::string input_path = argv[1];
  std::vector<char> input_cstr(input_path.begin(), input_path.end());
  input_cstr.push_back('\0');

  const FASTEIG_INT ierr = parseinput(input_cstr.data(), &params);
  if (ierr != 0) {
    std::cerr << "Error in the input file: " << input_path << " ierr = " << ierr << "\n";
    return 1;
  }

  // Create a problem object from params (builds SpMat + default ops internally).
  fasteig_problem prob(params);

  // Option A: solve with defaults derived from params.
  //FASTEIG_INT status = prob.solve();

  // Option B: override/extend defaults with an explicit options object.
  fasteig_options opts;
  opts.configure_from(params);
  // Example overrides:
  // opts.set("nev", "20");
  // opts.set("solver", "Chebyshev");
  // opts.set("tol", "1e-10");

  // Stdout verbosity (1 = final table, 2 = per-iteration + final table).
  const int verbosity = opts.verbosity();

  const std::string solver_name = opts.solver().empty() ? prob.solver() : opts.solver();

  // Output file always contains the "verbosity=2" style information: header, progress (if enabled),
  // runtime, and the final eigenpairs table.
  std::ofstream out = open_log(opts.output_path(solver_name));
  prob.set_log_stream(&out);

  fasteig_print_matrix_summary(out, prob);
  fasteig_print_run_header(out, opts, solver_name);

  if (verbosity >= 1) {
    fasteig_print_matrix_summary(std::cout, prob);
    fasteig_print_run_header(std::cout, opts, solver_name);
  }

  // Generalized setup (e.g., B factorization) should happen once up-front, before timing the solver.
  prob.prepare(opts);

  out << "\n";
  out << "PROGRESS\n";
  out << "========\n";
  out << "\n";
  if (verbosity >= 2) {
    std::cout << "\n";
    std::cout << "PROGRESS\n";
    std::cout << "========\n";
    std::cout << "\n";
  }

  const auto t0 = std::chrono::steady_clock::now();
  const FASTEIG_INT status = prob.solve(opts);
  const auto t1 = std::chrono::steady_clock::now();
  const double runtime_sec = std::chrono::duration<double>(t1 - t0).count();
  const auto iterations = prob.last_iterations();
  const auto b_fact_sec = prob.b_factorization_sec();

  //DMR
  VectorXd residual_norms;
  residual_norms = fasteig_compute_residual_norms(prob.opsA(),
                                                  prob.bmatrix() ? &prob.opsB() : nullptr,
                                                  prob.eigenvalues(),
                                                  prob.eigenvectors());

  // Always write full results to files (output_prefix/logfile controls naming).
  fasteig_write_results_text(out, status, prob.eigenvalues(), residual_norms, b_fact_sec, runtime_sec, iterations);
  out.flush();
  prob.set_log_stream(nullptr);

  // verbosity controls stdout only.
  if (verbosity >= 1) {
    std::cout << "\n";
    std::cout << "FINAL RESULTS\n";
    std::cout << "=============\n";
    std::cout << "\n";
    fasteig_print_eigenpairs_table(std::cout, prob.eigenvalues(), residual_norms);
    std::cout << "\n";

    std::cout << "DIAGNOSTICS\n";
    std::cout << "===========\n";
    std::cout << "\n";
    if (iterations.has_value()) {
      std::cout << "  Iterations           : " << *iterations << "\n";
    }
    if (b_fact_sec.has_value()) {
      std::cout << "  B Factorization (sec): " << std::fixed << std::setprecision(6) << *b_fact_sec << "\n";
    }
    std::cout << "  Runtime (sec)        : " << std::fixed << std::setprecision(6) << runtime_sec << "\n";
    std::cout << "\n";
  }
  //std::cout << "solve status = " << status << "\n";
  //if (status == 0) {
  //  std::cout << "Computed " << prob.eigenvalues().size() << " Ritz values.\n";
  //}
  return (status == 0) ? 0 : 2;
}

