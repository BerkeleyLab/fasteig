#pragma once

#include "fasteig.hpp"

#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

class InputParser {
public:
  // Parses a key-value file and loads matrices referenced by keys.
  // Supported keys: amatrix, bmatrix, logfile, nev, solver, which, maxiter, maxdim,
  //                 degree, blksize, tol, shift, seed, verbosity/verbose, print_every
  static FASTEIG_INT parse_file(const std::string& filename, fasteig_params& params);

private:
  static std::optional<std::pair<std::string, std::string>> read_kv_line(std::istream& in);
  static void load_matrix_market_coordinate(const std::string& mtx_path,
                                            FASTEIG_INT& n_out,
                                            std::vector<FASTEIG_INT>& row,
                                            std::vector<FASTEIG_INT>& col,
                                            std::vector<FASTEIG_DOUBLE>& val,
                                            bool& symmetric_out,
                                            bool& skew_symmetric_out,
                                            bool& triangle_upper_out,
                                            FASTEIG_INT& diag_nnz_out,
                                            FASTEIG_INT& offdiag_nnz_out);

  static bool iequals(std::string_view a, std::string_view b);
};

// Small C-compatible wrapper retained for existing callers.
// NOTE: This is still C++ (it returns/accepts C types), but avoids forcing a rewrite elsewhere.
extern "C" FASTEIG_INT parseinput(char* fname, fasteig_params* params);

