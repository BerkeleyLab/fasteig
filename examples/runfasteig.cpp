#include "fasteig.hpp"
#include "fasteig_options.hpp"
#include "parseinput.hpp"
#include "fasteig_problem.hpp"

#include <iostream>
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

  const FASTEIG_INT status = prob.solve(opts);
  return (status == 0) ? 0 : 2;
}
