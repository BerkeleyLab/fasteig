#include "fasteig.hpp"
#include "core/Executor.hpp"
#include "fasteig_options.hpp"
#include "parseinput.hpp"
#include "fasteig_problem.hpp"
//DMR

#include <algorithm>
#include <cctype>
#include <iostream>
#include <string>
#include <vector>

namespace {

std::string lower_copy(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

}  // namespace

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
  const std::string exe_l = lower_copy(argv[0]);
#ifdef FASTEIG_GPU_EXEC
  const bool gpu_exec = true;
#else
  const bool gpu_exec = (exe_l.find("gpu") != std::string::npos);
#endif
  const std::string solver_l = lower_copy(opts.solver());
  if (gpu_exec && (solver_l == "lanczos" || solver_l == "gpulanczos" ||
                   solver_l == "blocklanczos" || solver_l == "gpublocklanczos" ||
                   solver_l == "krylovschur" || solver_l == "blockkrylovschur")) {
    if (solver_l == "gpulanczos") {
      opts.set("solver", "Lanczos");
    } else if (solver_l == "gpublocklanczos") {
      opts.set("solver", "BlockLanczos");
    }
    opts.set_gpu_backend(true);
  }
  prob.set_executor(fasteig::core::make_executor(opts.gpu_backend()));
  // Example overrides:
  // opts.set("nev", "20");
  // opts.set("solver", "Chebyshev");
  // opts.set("tol", "1e-10");

  const FASTEIG_INT status = prob.solve(opts);
  return (status == 0) ? 0 : 2;
}

