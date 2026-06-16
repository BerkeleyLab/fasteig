//********************************************************************************
//
//  Authors
//    Dong Min Roh
//    Chao Yang
//    Applied Mathematics and Computational Research Division
//    Lawrence Berkeley National Laboratory
//
//  Refactor notes (2025-12-30)
//    - Introduces RAII-friendly parameter and input parsing classes.
//    - Keeps a small C-compatible wrapper function signature: parseinput(char*, fasteig_param*).
//    - Removes C-style heap management (malloc/free) from parseinput implementation.
//
//********************************************************************************

#pragma once

// =============================
// Configuration
// =============================
#ifndef FASTEIG_DOUBLE
#define FASTEIG_DOUBLE double
#endif

#ifndef FASTEIG_INT
#define FASTEIG_INT int
#endif

#ifndef FASTEIG_SIZET
#define FASTEIG_SIZET size_t
#endif

// C headers (only where truly needed)
#include <cstdio>
#include <cstdlib>
#include <cstring>

// C++ standard includes
#include <algorithm>
#include <cassert>
#include <cctype>
#include <exception>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <filesystem>

// Parallelism
#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef USE_MPI
#include <mpi.h>
#endif

// Eigen core
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <Eigen/SparseCholesky>
#ifdef EIGEN_USE_MKL_ALL
#include <Eigen/PardisoSupport>
#endif

#include <unsupported/Eigen/IterativeSolvers>

// Precision and matrix types
enum class MatrixFormat {
    MTX,
    RSA,
    FEI,
    HDF,
    UNKNOWN
};

// Precision and matrix types
using SpMat = Eigen::SparseMatrix<FASTEIG_DOUBLE>;
using Perm  = Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic>;
using VectorXd = Eigen::VectorXd;
using MatrixXd = Eigen::MatrixXd;

// =============================
// Linear operator callbacks
// =============================
struct operations {
  std::function<void(const VectorXd&, VectorXd&)> matvec;
  std::function<void(const MatrixXd&, MatrixXd&)> block_matvec;
  // Optional linear solve hooks (e.g., apply B^{-1} for generalized eigenproblems).
  std::function<void(const VectorXd&, VectorXd&)> solve;
  std::function<void(const MatrixXd&, MatrixXd&)> block_solve;
  std::function<void(const VectorXd&, const FASTEIG_DOUBLE&, VectorXd&)> CG;
  std::function<void(const SpMat&, const VectorXd&, VectorXd&)> minres;

  SpMat H0;
  SpMat V21;
};

// =============================
// Parameters (C++ / RAII)
// =============================
class fasteig_params {
public:
  // Files / logging
  std::string amatfname;
  std::string bmatfname;
  std::string logprefix = "logfile";
  std::string output_format = "text";

  // Matrix type
  MatrixFormat matrixformat;

  // Problem sizes
  FASTEIG_INT n    = 0;
  FASTEIG_INT annz = 0;
  FASTEIG_INT bnnz = 0;

  // Sparse triplets for A (0-based)
  std::vector<FASTEIG_INT> arowind;
  std::vector<FASTEIG_INT> acolind;
  std::vector<FASTEIG_DOUBLE> anzvals;

  // MatrixMarket metadata (only meaningful when matrixformat == MTX).
  // For symmetric/skew-symmetric MatrixMarket matrices that store only one triangle,
  // we keep the stored triangle and let ops.matvec handle symmetry.
  bool mtx_symmetric = false;
  bool mtx_skew_symmetric = false;
  bool mtx_triangle_upper = true;  // which triangle is stored (upper vs lower)
  FASTEIG_INT mtx_diag_nnz = 0;     // diagonal entries stored
  FASTEIG_INT mtx_offdiag_nnz = 0;  // off-diagonal entries stored (single triangle)

  // Optional MatrixMarket metadata for B (only meaningful when bmatfname is set and format is MTX).
  bool b_mtx_symmetric = false;
  bool b_mtx_skew_symmetric = false;
  bool b_mtx_triangle_upper = true;
  FASTEIG_INT b_mtx_diag_nnz = 0;
  FASTEIG_INT b_mtx_offdiag_nnz = 0;

  // Optional CSC for A/B (retained for compatibility with old code paths)
  std::vector<FASTEIG_INT> acolptr;
  std::vector<FASTEIG_INT> bcolptr;
  std::vector<FASTEIG_INT> browind;
  std::vector<FASTEIG_INT> bcolind;
  std::vector<FASTEIG_DOUBLE> bnzvals;

  // Solver controls
  FASTEIG_INT nev     = 100;
  std::string which   = "LM";
  std::string solver  = "Lanczos";
  FASTEIG_INT maxiter = 10;
  FASTEIG_INT maxdim  = 20;
  FASTEIG_INT degree  = 10;
  FASTEIG_INT blksize = 5;
  FASTEIG_DOUBLE shift = 0.0;
  FASTEIG_DOUBLE tol   = 1.0e-6;
  FASTEIG_INT seed = -1;  // <0 => unspecified (implementation-defined default)

  // Output controls
  // verbosity levels (stdout only; files are always written when output prefix is set):
  //   1 = eigenpairs table
  //   2 = per-iteration summary + eigenpairs table
  FASTEIG_INT verbosity = -1;    // -1 => unspecified (driver chooses defaults)
  FASTEIG_INT print_every = 0;   // 0 => no per-iteration progress

  void clear_matrices() {
    n = 0;
    annz = 0;
    bnnz = 0;
    arowind.clear();
    acolind.clear();
    anzvals.clear();
    mtx_symmetric = false;
    mtx_skew_symmetric = false;
    mtx_triangle_upper = true;
    mtx_diag_nnz = 0;
    mtx_offdiag_nnz = 0;
    acolptr.clear();
    bcolptr.clear();
    browind.clear();
    bcolind.clear();
    bnzvals.clear();
  }

  FASTEIG_INT mtx_stored_nnz() const noexcept { return mtx_diag_nnz + mtx_offdiag_nnz; }
  FASTEIG_INT mtx_full_nnz() const noexcept {
    return mtx_symmetric || mtx_skew_symmetric ? (mtx_diag_nnz + 2 * mtx_offdiag_nnz) : mtx_stored_nnz();
  }

  FASTEIG_INT b_mtx_stored_nnz() const noexcept { return b_mtx_diag_nnz + b_mtx_offdiag_nnz; }
  FASTEIG_INT b_mtx_full_nnz() const noexcept {
    return b_mtx_symmetric || b_mtx_skew_symmetric ? (b_mtx_diag_nnz + 2 * b_mtx_offdiag_nnz) : b_mtx_stored_nnz();
  }
};


