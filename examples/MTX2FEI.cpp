// MTX2FEI.cpp
//
// Read a sparse matrix in Matrix Market (coordinate) format (.mtx),
// convert to CSC (compressed sparse column) arrays: colptr, rowind, values,
// then write a simple binary file containing:
//   FASTEIG_INT nrows, FASTEIG_INT ncols, FASTEIG_INT nnz
//   FASTEIG_INT colptr[ncols+1]
//   FASTEIG_INT rowind[nnz]
//   FASTEIG_DOUBLE  values[nnz]
//
// Notes:
// - Supports real/integer/pattern coordinate matrices (not dense array format).
// - Handles symmetric/skew-symmetric/hermitian by expanding to full matrix.
// - Converts 1-based indices in MM file to 0-based.
// - Sorts entries by (col,row) and sums duplicates.
// - Uses little-endian as produced by the host (common on x86_64).
//
// Build (example):
//   g++ -O2 -std=c++17 -o mm_to_csc_binary mm_to_csc_binary.cpp
//
// Usage:
//   ./mm_to_csc_binary input.mtx output.bin

#include "fasteig.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

struct COOEntry {
    FASTEIG_INT r;
    FASTEIG_INT c;
    FASTEIG_DOUBLE  v;
};

static inline std::string to_lower(std::string s) {
    for (char& ch : s) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    return s;
}

struct MMHeader {
    bool coordinate = false;     // coordinate vs array
    bool pattern = false;        // pattern vs having values
    bool real = false;
    bool integer = false;
    bool complex = false;
    std::string symmetry;        // general/symmetric/skew-symmetric/hermitian
};

static MMHeader parse_mm_banner(const std::string& line) {
    // Expected: %%MatrixMarket matrix coordinate real general
    std::istringstream iss(line);
    std::string banner, object, format, field, symmetry;
    iss >> banner >> object >> format >> field >> symmetry;
    if (!iss || to_lower(banner) != "%%matrixmarket") {
        throw std::runtime_error("Not a MatrixMarket banner line.");
    }
    if (to_lower(object) != "matrix") {
        throw std::runtime_error("MatrixMarket object is not 'matrix'.");
    }

    MMHeader h;
    std::string fmt = to_lower(format);
    h.coordinate = (fmt == "coordinate");

    if (!h.coordinate) {
        throw std::runtime_error("Only MatrixMarket 'coordinate' format is supported.");
    }

    std::string fld = to_lower(field);
    h.pattern = (fld == "pattern");
    h.real    = (fld == "real");
    h.integer = (fld == "integer");
    h.complex = (fld == "complex");

    if (!(h.pattern || h.real || h.integer)) {
        // complex not supported in this simple example
        throw std::runtime_error("Only field types: real, integer, pattern are supported.");
    }

    h.symmetry = to_lower(symmetry);
    if (h.symmetry != "general" &&
        h.symmetry != "symmetric" &&
        h.symmetry != "skew-symmetric" &&
        h.symmetry != "hermitian")
    {
        throw std::runtime_error("Unsupported symmetry: " + h.symmetry);
    }
    if (h.symmetry == "hermitian") {
        throw std::runtime_error("Hermitian not supported here (complex required).");
    }
    return h;
}

static void read_matrix_market_to_coo(const std::string& path,
                                      FASTEIG_INT& nrows, FASTEIG_INT& ncols,
                                      std::vector<COOEntry>& entries)
{
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Cannot open input file: " + path);

    std::string line;
    if (!std::getline(in, line)) throw std::runtime_error("Empty file: " + path);

    MMHeader hdr = parse_mm_banner(line);

    // Skip comments
    while (std::getline(in, line)) {
        if (!line.empty() && line[0] != '%') break;
    }
    if (!in) throw std::runtime_error("Unexpected EOF while reading size line.");

    // Size line: nrows ncols nnz
    {
        std::istringstream iss(line);
        FASTEIG_INT nnz = 0;
        iss >> nrows >> ncols >> nnz;
        if (!iss) throw std::runtime_error("Failed to parse MatrixMarket size line.");
        if (nrows <= 0 || ncols <= 0 || nnz < 0) throw std::runtime_error("Invalid matrix dimensions/nnz.");
        entries.clear();
        entries.reserve(static_cast<size_t>(nnz) * (hdr.symmetry == "general" ? 1 : 2));
    }

    // Read entries
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '%') continue;

        std::istringstream iss(line);
        FASTEIG_INT i1 = 0, j1 = 0;
        FASTEIG_DOUBLE v = 1.0;

        if (hdr.pattern) {
            iss >> i1 >> j1;
            v = 1.0;
        } else if (hdr.integer) {
            FASTEIG_INT iv = 0;
            iss >> i1 >> j1 >> iv;
            v = static_cast<FASTEIG_DOUBLE>(iv);
        } else if (hdr.real) {
            iss >> i1 >> j1 >> v;
        }
        if (!iss) throw std::runtime_error("Failed to parse entry line: " + line);

        // Convert 1-based -> 0-based
        FASTEIG_INT i = i1 - 1;
        FASTEIG_INT j = j1 - 1;
        if (i < 0 || i >= nrows || j < 0 || j >= ncols) {
            throw std::runtime_error("Index out of range in input: (" + std::to_string(i1) + "," + std::to_string(j1) + ")");
        }

        entries.push_back({i, j, v});

        // Expand symmetry if needed
        if (hdr.symmetry == "symmetric") {
            if (i != j) entries.push_back({j, i, v});
        } else if (hdr.symmetry == "skew-symmetric") {
            if (i == j) {
                // diagonal must be zero; ignore or enforce
                // enforce by storing 0; easiest is ignore
            } else {
                entries.push_back({j, i, -v});
            }
        }
    }
}

static void coo_to_csc(FASTEIG_INT nrows, FASTEIG_INT ncols,
                       std::vector<COOEntry>& entries,
                       std::vector<FASTEIG_INT>& colptr,
                       std::vector<FASTEIG_INT>& rowind,
                       std::vector<FASTEIG_DOUBLE>& values)
{
    // Sort by (col,row)
    std::sort(entries.begin(), entries.end(),
              [](const COOEntry& a, const COOEntry& b) {
                  if (a.c != b.c) return a.c < b.c;
                  return a.r < b.r;
              });

    // Sum duplicates
    std::vector<COOEntry> uniq;
    uniq.reserve(entries.size());
    for (const auto& e : entries) {
        if (!uniq.empty() && uniq.back().r == e.r && uniq.back().c == e.c) {
            uniq.back().v += e.v;
        } else {
            uniq.push_back(e);
        }
    }
    entries.swap(uniq);

    const FASTEIG_INT nnz = static_cast<FASTEIG_INT>(entries.size());
    colptr.assign(static_cast<size_t>(ncols) + 1, 0);
    rowind.resize(static_cast<size_t>(nnz));
    values.resize(static_cast<size_t>(nnz));

    // Count nnz per column
    for (const auto& e : entries) {
        if (e.c < 0 || e.c >= ncols) throw std::runtime_error("Column index out of range during CSC build.");
        colptr[static_cast<size_t>(e.c) + 1] += 1;
    }

    // Prefix sum to get column pointers
    for (FASTEIG_INT j = 0; j < ncols; ++j) {
        colptr[static_cast<size_t>(j) + 1] += colptr[static_cast<size_t>(j)];
    }

    // Fill rowind/values by scanning entries (already sorted by col,row)
    // colptr currently holds cumulative counts; we'll use a write cursor copy.
    std::vector<FASTEIG_INT> next = colptr;
    for (const auto& e : entries) {
        FASTEIG_INT pos = next[static_cast<size_t>(e.c)]++;
        rowind[static_cast<size_t>(pos)] = e.r;
        values[static_cast<size_t>(pos)] = e.v;
    }

    // Optional: verify each column's row indices are sorted (they should be)
    // and within bounds.
    for (FASTEIG_INT j = 0; j < ncols; ++j) {
        FASTEIG_INT start = colptr[static_cast<size_t>(j)];
        FASTEIG_INT end   = colptr[static_cast<size_t>(j + 1)];
        FASTEIG_INT prev = -1;
        for (FASTEIG_INT k = start; k < end; ++k) {
            FASTEIG_INT r = rowind[static_cast<size_t>(k)];
            if (r < 0 || r >= nrows) throw std::runtime_error("Row index out of range in CSC.");
            if (r < prev) throw std::runtime_error("Row indices not sorted within a column (unexpected).");
            prev = r;
        }
    }
}

static void write_csc_binary(const std::string& outpath,
                             FASTEIG_INT nrows, FASTEIG_INT ncols, FASTEIG_INT nnz,
                             const std::vector<FASTEIG_INT>& colptr,
                             const std::vector<FASTEIG_INT>& rowind,
                             const std::vector<FASTEIG_DOUBLE>& values)
{
    if (static_cast<FASTEIG_INT>(colptr.size()) != ncols + 1) throw std::runtime_error("colptr size mismatch.");
    if (static_cast<FASTEIG_INT>(rowind.size()) != nnz) throw std::runtime_error("rowind size mismatch.");
    if (static_cast<FASTEIG_INT>(values.size()) != nnz) throw std::runtime_error("values size mismatch.");

    std::ofstream out(outpath, std::ios::binary);
    if (!out) throw std::runtime_error("Cannot open output file: " + outpath);

    out.write(reinterpret_cast<const char*>(&nrows), sizeof(FASTEIG_INT));
    out.write(reinterpret_cast<const char*>(&ncols), sizeof(FASTEIG_INT));
    out.write(reinterpret_cast<const char*>(&nnz),   sizeof(FASTEIG_INT));

    out.write(reinterpret_cast<const char*>(colptr.data()), sizeof(FASTEIG_INT) * colptr.size());
    out.write(reinterpret_cast<const char*>(rowind.data()), sizeof(FASTEIG_INT) * rowind.size());
    out.write(reinterpret_cast<const char*>(values.data()), sizeof(FASTEIG_DOUBLE)  * values.size());

    if (!out) throw std::runtime_error("Failed while writing binary output: " + outpath);
}

int main(int argc, char** argv)
{
    try {
        if (argc != 3) {
            std::cerr << "Usage: " << argv[0] << " input.mtx output.fei\n";
            return 2;
        }

        const std::string inpath  = argv[1];
        const std::string outpath = argv[2];

        FASTEIG_INT nrows = 0, ncols = 0;
        std::vector<COOEntry> entries;

        read_matrix_market_to_coo(inpath, nrows, ncols, entries);

        std::vector<FASTEIG_INT> colptr, rowind;
        std::vector<FASTEIG_DOUBLE> values;

        coo_to_csc(nrows, ncols, entries, colptr, rowind, values);

        const FASTEIG_INT nnz = static_cast<FASTEIG_INT>(values.size());
        write_csc_binary(outpath, nrows, ncols, nnz, colptr, rowind, values);

        std::cout << "Wrote CSC binary: " << outpath
                  << " (nrows=" << nrows
                  << ", ncols=" << ncols
                  << ", nnz=" << nnz << ")\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}

