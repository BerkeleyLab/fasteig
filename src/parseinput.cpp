#include "parseinput.hpp"

#include "mmio.hpp"  // existing Matrix Market helper in the project

namespace {

struct MmFileCloser {
  void operator()(FILE* f) const noexcept {
    if (f) std::fclose(f);
  }
};

using UniqueFILE = std::unique_ptr<FILE, MmFileCloser>;

inline UniqueFILE open_c_file(const std::string& path, const char* mode) {
  FILE* fp = std::fopen(path.c_str(), mode);
  return UniqueFILE(fp);
}

inline bool starts_with_key(std::string_view key, std::string_view token_prefix) {
  if (key.size() < token_prefix.size()) return false;
  return std::equal(token_prefix.begin(), token_prefix.end(), key.begin(),
                    [](char a, char b) { return std::tolower(static_cast<unsigned char>(a)) ==
                                               std::tolower(static_cast<unsigned char>(b)); });
}

bool readRSA(const std::string& filename,
             FASTEIG_INT& nRows,
             FASTEIG_INT& nCols,
             FASTEIG_INT& nNonZeros,
             std::vector<FASTEIG_INT>& colPtr,
             std::vector<FASTEIG_INT>& rowInd,
             std::vector<FASTEIG_DOUBLE>& values) {
  std::ifstream file(filename);
  if (!file.is_open()) return false;

  std::string line1, line2, line3, line4, line5;
  std::getline(file, line1);
  std::getline(file, line2);
  std::getline(file, line3);
  std::getline(file, line4);
  std::getline(file, line5);

  std::stringstream ss4(line3);
  std::string type;
  int dummy = 0;
  ss4 >> type >> nRows >> nCols >> nNonZeros >> dummy;

  colPtr.resize(static_cast<size_t>(nCols) + 1);
  for (int i = 0; i <= nCols; ++i) {
    file >> colPtr[static_cast<size_t>(i)];
    colPtr[static_cast<size_t>(i)]--;
  }

  rowInd.resize(static_cast<size_t>(nNonZeros));
  for (int i = 0; i < nNonZeros; ++i) {
    file >> rowInd[static_cast<size_t>(i)];
    rowInd[static_cast<size_t>(i)]--;
  }

  values.resize(static_cast<size_t>(nNonZeros));
  for (int i = 0; i < nNonZeros; ++i) {
    file >> values[static_cast<size_t>(i)];
  }

  return static_cast<bool>(file);
}

bool readFEI(const std::string& filename,
             FASTEIG_INT& nRows,
             FASTEIG_INT& nCols,
             FASTEIG_INT& nNonZeros,
             std::vector<FASTEIG_INT>& colPtr,
             std::vector<FASTEIG_INT>& rowInd,
             std::vector<FASTEIG_DOUBLE>& values) {
  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open()) return false;

  file.read(reinterpret_cast<char*>(&nRows), sizeof(FASTEIG_INT));
  file.read(reinterpret_cast<char*>(&nCols), sizeof(FASTEIG_INT));
  file.read(reinterpret_cast<char*>(&nNonZeros), sizeof(FASTEIG_INT));

  if (!file) {
    throw std::runtime_error("Failed reading matrix header.");
  }
  if (nRows <= 0 || nCols <= 0 || nNonZeros < 0) {
    throw std::runtime_error("Invalid matrix sizes in binary file.");
  }

  colPtr.resize(static_cast<size_t>(nCols) + 1);
  rowInd.resize(static_cast<size_t>(nNonZeros));
  values.resize(static_cast<size_t>(nNonZeros));

  file.read(reinterpret_cast<char*>(colPtr.data()),
            sizeof(FASTEIG_INT) * colPtr.size());
  file.read(reinterpret_cast<char*>(rowInd.data()),
            sizeof(FASTEIG_INT) * rowInd.size());
  file.read(reinterpret_cast<char*>(values.data()),
            sizeof(FASTEIG_DOUBLE) * values.size());

  if (!file) {
    throw std::runtime_error("Failed reading CSC arrays.");
  }
  if (colPtr.front() != 0 || colPtr.back() != nNonZeros) {
    throw std::runtime_error("Invalid column pointer array.");
  }

  return true;
}

} // namespace


MatrixFormat detect_format(const std::string& filename)
{
    namespace fs = std::filesystem;
    fs::path p(filename);

    if (!p.has_extension()) {
        return MatrixFormat::UNKNOWN;
    }

    std::string ext = p.extension().string();
    // normalize
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    if (ext == ".mtx") {
        return MatrixFormat::MTX;
    } else if (ext == ".rsa") {
        return MatrixFormat::RSA;
    } else if (ext == ".fei") {
        return MatrixFormat::FEI;
    } else if (ext == ".hdf") {
        return MatrixFormat::HDF;
    } else {
        return MatrixFormat::UNKNOWN;
    }
}


bool InputParser::iequals(std::string_view a, std::string_view b) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i) {
    if (std::tolower(static_cast<unsigned char>(a[i])) !=
        std::tolower(static_cast<unsigned char>(b[i]))) {
      return false;
    }
  }
  return true;
}

std::optional<std::pair<std::string, std::string>> InputParser::read_kv_line(std::istream& in) {
  std::string key;
  std::string value;

  while (in >> key) {
    // allow comment lines starting with # or %
    if (!key.empty() && (key[0] == '#' || key[0] == '%')) {
      in.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
      continue;
    }

    if (!(in >> value)) {
      return std::nullopt;
    }
    return std::make_pair(std::move(key), std::move(value));
  }
  return std::nullopt;
}

void InputParser::load_matrix_market_coordinate(const std::string& mtx_path,
                                                FASTEIG_INT& n_out,
                                                std::vector<FASTEIG_INT>& row,
                                                std::vector<FASTEIG_INT>& col,
                                                std::vector<FASTEIG_DOUBLE>& val,
                                                bool& symmetric_out,
                                                bool& skew_symmetric_out,
                                                bool& triangle_upper_out,
                                                FASTEIG_INT& diag_nnz_out,
                                                FASTEIG_INT& offdiag_nnz_out) {
  auto f = open_c_file(mtx_path, "r");
  if (!f) {
    throw std::runtime_error("cannot open MatrixMarket file: " + mtx_path);
  }

  MM_typecode matcode;
  if (mm_read_banner(f.get(), &matcode) != 0) {
    throw std::runtime_error("could not process Matrix Market banner in: " + mtx_path);
  }
  const bool is_symmetric = mm_is_symmetric(matcode);
  const bool is_skew = mm_is_skew(matcode);
  if (is_symmetric && is_skew) {
    throw std::runtime_error("invalid MatrixMarket banner (both symmetric and skew): " + mtx_path);
  }

  FASTEIG_INT M = 0, N = 0, nz = 0;
  const FASTEIG_INT ret = mm_read_mtx_crd_size(f.get(), &M, &N, &nz);
  if (ret != 0) {
    throw std::runtime_error("failed to read MatrixMarket coordinate size in: " + mtx_path);
  }
  if (M != N) {
    throw std::runtime_error("MatrixMarket file is not square: " + mtx_path);
  }

  n_out = N;
  row.clear();
  col.clear();
  val.clear();
  row.reserve(static_cast<size_t>(nz));
  col.reserve(static_cast<size_t>(nz));
  val.reserve(static_cast<size_t>(nz));

  symmetric_out = is_symmetric;
  skew_symmetric_out = is_skew;
  triangle_upper_out = true;
  diag_nnz_out = 0;
  offdiag_nnz_out = 0;

  // For symmetric/skew-symmetric MatrixMarket files, the data is typically stored
  // in one triangle (upper or lower) + diagonal. We DO NOT expand to a full matrix.
  // Instead, we keep one triangle and let ops.matvec handle symmetry.
  FASTEIG_INT upper_off = 0;
  FASTEIG_INT lower_off = 0;

  for (FASTEIG_INT i = 0; i < nz; ++i) {
    int r = 0, c = 0;
    double x = 0.0;
    if (std::fscanf(f.get(), "%d %d %lg\n", &r, &c, &x) != 3) {
      throw std::runtime_error("unexpected EOF while reading: " + mtx_path);
    }
    // Adjust 1-based -> 0-based
    const FASTEIG_INT rr = static_cast<FASTEIG_INT>(r - 1);
    const FASTEIG_INT cc = static_cast<FASTEIG_INT>(c - 1);
    const FASTEIG_DOUBLE xx = static_cast<FASTEIG_DOUBLE>(x);

    row.push_back(rr);
    col.push_back(cc);
    val.push_back(xx);

    if (rr == cc) {
      ++diag_nnz_out;
    } else if (rr < cc) {
      ++upper_off;
    } else {
      ++lower_off;
    }
  }

  if (is_symmetric || is_skew) {
    // Decide which triangle we *treat* as the stored triangle and filter accordingly.
    // This protects against files that accidentally contain entries from both triangles.
    triangle_upper_out = (upper_off >= lower_off);

    std::vector<FASTEIG_INT> r2;
    std::vector<FASTEIG_INT> c2;
    std::vector<FASTEIG_DOUBLE> x2;
    r2.reserve(row.size());
    c2.reserve(col.size());
    x2.reserve(val.size());

    FASTEIG_INT off = 0;
    for (size_t k = 0; k < row.size(); ++k) {
      const FASTEIG_INT rr = row[k];
      const FASTEIG_INT cc = col[k];
      const bool keep = (rr == cc) || (triangle_upper_out ? (rr < cc) : (rr > cc));
      if (!keep) continue;
      r2.push_back(rr);
      c2.push_back(cc);
      x2.push_back(val[k]);
      if (rr != cc) ++off;
    }

    offdiag_nnz_out = off;
    row = std::move(r2);
    col = std::move(c2);
    val = std::move(x2);
  } else {
    offdiag_nnz_out = static_cast<FASTEIG_INT>(row.size()) - diag_nnz_out;
  }
}

FASTEIG_INT InputParser::parse_file(const std::string& filename, fasteig_params& params) {
  std::ifstream in(filename);
  if (!in) {
    std::cerr << "failed to open " << filename << "\n";
    return -5;
  }

  // defaults are already set in fasteig_params; nothing needed here.

  bool amatrix_ready = false;
  bool bmatrix_ready = false;
  //DMR
  bool verbosity_explicit = false;

  try {
    while (true) {
      const auto kv = read_kv_line(in);
      if (!kv) break;

      const auto& key = kv->first;
      const auto& value = kv->second;

      if (starts_with_key(key, "amatrix")) {
        params.amatfname = value;

        params.matrixformat = detect_format(params.amatfname);
        switch (params.matrixformat) {
        case MatrixFormat::MTX:
           {
           FASTEIG_INT nA = 0;
           std::vector<FASTEIG_INT> r, c;
           std::vector<FASTEIG_DOUBLE> x;
           bool mtx_sym = false;
           bool mtx_skew = false;
           bool mtx_upper = true;
           FASTEIG_INT mtx_diag = 0;
           FASTEIG_INT mtx_off = 0;
           load_matrix_market_coordinate(params.amatfname, nA, r, c, x,
                                         mtx_sym, mtx_skew, mtx_upper, mtx_diag, mtx_off);

           params.n = nA;
           params.annz = static_cast<FASTEIG_INT>(x.size());
           params.arowind = std::move(r);
           params.acolind = std::move(c);
           params.anzvals = std::move(x);
           params.mtx_symmetric = mtx_sym;
           params.mtx_skew_symmetric = mtx_skew;
           params.mtx_triangle_upper = mtx_upper;
           params.mtx_diag_nnz = mtx_diag;
           params.mtx_offdiag_nnz = mtx_off;
           break;
           }
        case MatrixFormat::RSA:
           {
           FASTEIG_INT nrows = 0, ncols = 0, nnz = 0;
           std::vector<FASTEIG_INT> colPtr, rowInd;
           std::vector<FASTEIG_DOUBLE> values;
           if (!readRSA(params.amatfname, nrows, ncols, nnz,
                        colPtr, rowInd, values) )
              throw std::runtime_error(
                "Could not load matrix " + params.amatfname
              );
           params.n = nrows;
           params.annz = nnz;
           params.acolptr = std::move(colPtr);
           params.arowind = std::move(rowInd);
           params.anzvals = std::move(values);
           break;
           }
        case MatrixFormat::FEI:
           {
           FASTEIG_INT nrows = 0, ncols = 0, nnz = 0;
           std::vector<FASTEIG_INT> colPtr, rowInd;
           std::vector<FASTEIG_DOUBLE> values;
           if (!readFEI(params.amatfname, nrows, ncols, nnz,
                        colPtr, rowInd, values) )
              throw std::runtime_error(
                "Could not load matrix " + params.amatfname
              );
           params.n = nrows;
           params.annz = nnz;
           params.acolptr = std::move(colPtr);
           params.arowind = std::move(rowInd);
           params.anzvals = std::move(values);
           break;
           }
        case MatrixFormat::HDF:
           //
           break;
        case MatrixFormat::UNKNOWN:
           //
           break;
        default:
            throw std::runtime_error(
                "Unsupported matrix format for file: " + params.amatfname
            );
        } // end switch

        amatrix_ready = true;
      } else if (starts_with_key(key, "bmatrix")) {
        params.bmatfname = value;
        const MatrixFormat bfmt = detect_format(params.bmatfname);
        if (bfmt == MatrixFormat::UNKNOWN) {
          throw std::runtime_error("Unsupported matrix format for file: " + params.bmatfname);
        }

        // For now, require A and B to be the same format family to keep the input deck simple.
        if (params.matrixformat != MatrixFormat::UNKNOWN && bfmt != params.matrixformat) {
          throw std::runtime_error("bmatrix format must match amatrix format for now.");
        }

        switch (bfmt) {
          case MatrixFormat::MTX: {
            FASTEIG_INT nB = 0;
            std::vector<FASTEIG_INT> brow, bcol;
            std::vector<FASTEIG_DOUBLE> bval;
            bool symB = false, skewB = false, upperB = true;
            FASTEIG_INT diagB = 0, offdiagB = 0;
            load_matrix_market_coordinate(params.bmatfname, nB, brow, bcol, bval,
                                          symB, skewB, upperB, diagB, offdiagB);
            if (amatrix_ready && nB != params.n) {
              throw std::runtime_error("bmatrix dimension does not match amatrix dimension.");
            }
            params.n = nB;
            params.bnnz = static_cast<FASTEIG_INT>(brow.size());
            params.browind = std::move(brow);
            params.bcolind = std::move(bcol);
            params.bnzvals = std::move(bval);
            params.b_mtx_symmetric = symB;
            params.b_mtx_skew_symmetric = skewB;
            params.b_mtx_triangle_upper = upperB;
            params.b_mtx_diag_nnz = diagB;
            params.b_mtx_offdiag_nnz = offdiagB;
            break;
          }
          case MatrixFormat::RSA: {
            FASTEIG_INT nrows = 0, ncols = 0, nnz = 0;
            std::vector<FASTEIG_INT> colPtr, rowInd;
            std::vector<FASTEIG_DOUBLE> values;
            if (!readRSA(params.bmatfname, nrows, ncols, nnz, colPtr, rowInd, values)) {
              throw std::runtime_error("Could not load matrix " + params.bmatfname);
            }
            if (nrows != ncols) throw std::runtime_error("bmatrix must be square.");
            if (amatrix_ready && nrows != params.n) {
              throw std::runtime_error("bmatrix dimension does not match amatrix dimension.");
            }
            params.n = nrows;
            params.bnnz = nnz;
            params.bcolptr = std::move(colPtr);
            params.browind = std::move(rowInd);
            params.bnzvals = std::move(values);
            break;
          }
          case MatrixFormat::FEI: {
            FASTEIG_INT nrows = 0, ncols = 0, nnz = 0;
            std::vector<FASTEIG_INT> colPtr, rowInd;
            std::vector<FASTEIG_DOUBLE> values;
            if (!readFEI(params.bmatfname, nrows, ncols, nnz, colPtr, rowInd, values)) {
              throw std::runtime_error("Could not load matrix " + params.bmatfname);
            }
            if (nrows != ncols) throw std::runtime_error("bmatrix must be square.");
            if (amatrix_ready && nrows != params.n) {
              throw std::runtime_error("bmatrix dimension does not match amatrix dimension.");
            }
            params.n = nrows;
            params.bnnz = nnz;
            params.bcolptr = std::move(colPtr);
            params.browind = std::move(rowInd);
            params.bnzvals = std::move(values);
            break;
          }
          case MatrixFormat::HDF:
          case MatrixFormat::UNKNOWN:
          default:
            throw std::runtime_error("Unsupported matrix format for file: " + params.bmatfname);
        }

        bmatrix_ready = true;
      } else if (starts_with_key(key, "logfile")) {
        params.logprefix = value;
      } else if (starts_with_key(key, "nev")) {
        params.nev = static_cast<FASTEIG_INT>(std::stoi(value));
      } else if (starts_with_key(key, "solver")) {
        params.solver = value;
      } else if (starts_with_key(key, "which")) {
        params.which = value;
      } else if (starts_with_key(key, "maxiter")) {
        params.maxiter = static_cast<FASTEIG_INT>(std::stoi(value));
      } else if (starts_with_key(key, "maxdim")) {
        params.maxdim = static_cast<FASTEIG_INT>(std::stoi(value));
      } else if (starts_with_key(key, "blksize")) {
        params.blksize = static_cast<FASTEIG_INT>(std::stoi(value));
      //DMR
      } else if (starts_with_key(key, "verbosity")) {
        params.verbosity = static_cast<FASTEIG_INT>(std::stoi(value));
        if (params.verbosity <= 1) params.verbosity = 1;
        if (params.verbosity >= 2) params.verbosity = 2;
        verbosity_explicit = true;
      } else if (starts_with_key(key, "print_every")) {
        params.print_every = static_cast<FASTEIG_INT>(std::stoi(value));
      } else if (starts_with_key(key, "output_prefix")) {
        params.logprefix = value;
      } else if (starts_with_key(key, "output_format")) {
        params.output_format = value;
      } else if (starts_with_key(key, "verbose")) {
        // Backwards-compatible boolean form; enable stdout and/or file depending on prefix.
        const std::string s = value;
        const auto lower = [](std::string x) {
          std::transform(x.begin(), x.end(), x.begin(),
                         [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
          return x;
        };
        const std::string v = lower(s);
        const bool on = (v == "1" || v == "true" || v == "yes" || v == "on");
        params.verbosity = 1;
        verbosity_explicit = true;
      } else if (starts_with_key(key, "tol")) {
        params.tol = static_cast<FASTEIG_DOUBLE>(std::stod(value));
      } else if (starts_with_key(key, "shift")) {
        params.shift = static_cast<FASTEIG_DOUBLE>(std::stod(value));
      } else if (starts_with_key(key, "seed")) {
        params.seed = static_cast<FASTEIG_INT>(std::stoi(value));
      } else {
        // unknown key: ignore for forward compatibility
      }
    }
  } catch (const std::exception& e) {
    std::cerr << "parseinput error: " << e.what() << "\n";
    return -1;
  }

  if (!amatrix_ready) {
    std::cerr << "parseinput: no matrix specified\n";
    return -1;
  }

  // bmatrix is optional in original code; keep that behavior.
  (void)bmatrix_ready;

  //DMR
  // Default output behavior:
  // - If user didn't specify verbosity, print to stdout (files are always written).
  if (params.verbosity < 0) params.verbosity = 1;
  if (params.verbosity > 2) params.verbosity = 2;

  return 0;
}

extern "C" FASTEIG_INT parseinput(char* fname, fasteig_params* params) {
  if (!fname || !params) return -10;
  return InputParser::parse_file(fname, *params);
}

