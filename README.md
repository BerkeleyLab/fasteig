# fasteig

Fast eigenvalue solver library, with CPU and NVIDIA GPU backends.

`fasteig` builds either with the classic `make.inc` workflow or with CMake. The
two are independent — pick whichever suits your environment. CMake is the
better choice if you want dependencies fetched automatically or plan to install
the library for other projects to consume.

---

## Table of contents

- [Building with CMake](#building-with-cmake)
  - [Quick start](#quick-start)
  - [Configure options](#configure-options)
  - [BLAS/LAPACK backends](#blaslapack-backends)
  - [CUDA and cuDSS](#cuda-and-cudss)
  - [Installing](#installing)
  - [Using fasteig from another CMake project](#using-fasteig-from-another-cmake-project)
  - [Installing with Spack](#installing-with-spack)
- [Building with make.inc](#building-with-makeinc)
  - [Perlmutter notes: MKL vs Eigen](#perlmutter-notes-mkl-vs-eigen)
  - [Optional project-space cuDSS install](#optional-project-space-cudss-install)
- [Running the examples](#running-the-examples)
  - [Input file format](#input-file-format)
  - [Solver and backend support](#solver-and-backend-support)
- [Version and build information](#version-and-build-information)
- [AI-Assisted Development Notice]
- [Scope of usage]
- [Support](#support)

---

## Building with CMake

Requires CMake 3.20 or newer and a C++17 compiler.

### Quick start

```bash
# CPU build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=gcc-14 -DCMAKE_CXX_COMPILER=g++-14
cmake --build build -j
```

```bash
# GPU build (separate build directory — see the note below)
cmake -S . -B build-cuda \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=gcc-14 -DCMAKE_CXX_COMPILER=g++-14 \
    -DENABLE_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build-cuda -j
```

The static library lands in `<build-dir>/lib/libfasteig_<PLAT>.a`, or
`libfasteig_<PLAT>_cuda.a` when CUDA is enabled. Example executables go to
`<build-dir>/examples/`.

> **Use a separate build directory for CPU and GPU builds.** Enabling CUDA
> changes which sources are compiled and writes the CUDA compiler into the
> cache. Reconfiguring in place leaves stale objects that link successfully but
> misbehave. The same applies to changing compilers: delete the build directory
> or create a new one.

Pass the CUDA architecture explicitly rather than relying on autodetection. On
a login node with no visible GPU, `native` fails at configure time. Use `70`
for V100, `80` for A100, `90` for H100, or a list such as `"80;90"`.

### Configure options

| Option | Default | Description |
| --- | --- | --- |
| `CMAKE_BUILD_TYPE` | `Release` | Only `Release` and `Debug` are accepted. |
| `PLAT` | `Generic` | Platform name; becomes part of the library filename. |
| `ENABLE_CUDA` | `AUTO` with tests; otherwise `OFF` | `AUTO` enables CUDA if a compiler is found; explicit `ON`/`OFF` override detection. Cached choices are preserved. |
| `USE_CUDSS` | `ON` | Use the NVIDIA cuDSS sparse direct solver. Inert unless `ENABLE_CUDA=ON`. |
| `USE_MKL` | `OFF` | Use Intel MKL for BLAS/LAPACK. |
| `USE_LIBSCI` | `OFF` | Use Cray LibSci for BLAS/LAPACK. |
| `BUILD_EXAMPLES` | `ON` | Build example programs. |
| `BUILD_TESTS` | `OFF` | Build and register CTest tests. Requires `BUILD_EXAMPLES=ON`. |
| `FASTEIG_ARCH_NATIVE` | `ON` | Add `-march=native` to Release builds. |
| `EIGEN_DIR` | *(empty)* | Path to an existing Eigen installation. Fetched automatically if unset. |
| `EIGEN_TAG` | `3.4.0` | Eigen git tag to fetch when `EIGEN_DIR` is unset. |
| `CUDSS_DIR` | *(empty)* | Path to an existing cuDSS install. Downloaded automatically if unset. |
| `CUDSS_VERSION` | `0.8.0.10` | cuDSS redistributable version to download. |
| `CUDSS_URL_HASH` | *(empty)* | SHA256 of the cuDSS tarball. Recommended. |
| `FASTEIG_BLAS_LIBRARIES` | *(empty)* | Explicit BLAS link line; bypasses `find_package`. |
| `FASTEIG_LAPACK_LIBRARIES` | *(empty)* | Explicit LAPACK link line; bypasses `find_package`. |

Turn off `FASTEIG_ARCH_NATIVE` on HPC systems where the login node CPU differs
from the compute nodes, and whenever a package manager is controlling the
target microarchitecture.

All downloaded dependencies are placed under `<build-dir>/_deps`. Nothing is
written into the source tree. To share a download cache across build
directories, set `FETCHCONTENT_BASE_DIR` to a common path.

### BLAS/LAPACK backends

The build selects a backend in this order: MKL if `USE_MKL=ON`, Cray LibSci if
`USE_LIBSCI=ON`, an explicit link line if `FASTEIG_BLAS_LIBRARIES` or
`FASTEIG_LAPACK_LIBRARIES` is set, otherwise `find_package(BLAS)` and
`find_package(LAPACK)`. If none of those succeed, OpenBLAS is fetched and built
from source as a last resort.

```bash
# Intel MKL — source setvars.sh first, or pass -DMKLROOT explicitly
module load PrgEnv-intel
cmake -S . -B build -DUSE_MKL=ON

# Cray LibSci via the compiler wrappers
module load PrgEnv-gnu
cmake -S . -B build \
    -DCMAKE_C_COMPILER=cc -DCMAKE_CXX_COMPILER=CC \
    -DUSE_LIBSCI=ON -DPLAT=Perlmutter -DFASTEIG_ARCH_NATIVE=OFF
```

MKL is always configured with the **LP64** interface. Eigen's Pardiso wrapper
uses `int` throughout, so ILP64 would change `MKL_INT` to `long long` and break
the `pardiso()` signature. The threading layer is chosen to match the
compiler's OpenMP runtime — `gnu_thread` with `libgomp` for GCC and Clang,
`intel_thread` with `libiomp5` for the Intel compilers. Mixing the two in one
process produces wrong results or deadlocks.

### CUDA and cuDSS

`ENABLE_CUDA=ON` compiles `src/gpu` in addition to `src/core` and defines
`FASTEIG_USE_CUDA` for the library and everything that links it. The CPU and
GPU implementations live in different namespaces and are both present in the
archive.

cuDSS is downloaded automatically into `<build-dir>/_deps/cudss` unless you
point at an existing install:

```bash
# use a site-provided cuDSS module
module load cudss
cmake -S . -B build-cuda -DENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80 \
    -DCUDSS_DIR=$CUDSS_ROOT
```

```bash
# pin the version and verify the download
cmake -S . -B build-cuda -DENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80 \
    -DCUDSS_VERSION=0.8.0.10 \
    -DCUDSS_URL_HASH=<sha256>
```

The cuDSS build must match your CUDA major version. NVIDIA publishes one
tarball per CUDA major, and older cuDSS releases have no build for newer
toolkits. If the download fails with an HTTP error, list the available releases:

```bash
curl -s https://developer.download.nvidia.com/compute/cudss/redist/redistrib_0.8.0.json \
  | python3 -m json.tool
```

That JSON also carries the SHA256 for each artifact, which is what you want for
`CUDSS_URL_HASH`.

Because cuDSS is a shared library outside the default loader path, the build
bakes an RPATH into every executable. Verify it resolves:

```bash
ldd build-cuda/examples/runfasteig_gpu | grep cudss
```

If it reports `not found`, add the cuDSS `lib` directory to `LD_LIBRARY_PATH`.

Note that AddressSanitizer is disabled automatically in CUDA `Debug` builds —
it interacts badly with device code. Use `compute-sanitizer` instead.

### Installing

```bash
cmake --install build --prefix /path/to/install
```

This installs the static library into `lib/`, public headers into `include/`,
the example executables into `bin/`, and a CMake package config into
`lib/cmake/fasteig/`. A bundled OpenBLAS, if one was built, is installed
alongside the library.

### Using fasteig from another CMake project

```cmake
find_package(fasteig REQUIRED)
target_link_libraries(myapp PRIVATE fasteig::fasteig)
```

```bash
cmake -S . -B build -DCMAKE_PREFIX_PATH=/path/to/install
```

The imported target carries the include directories, the compile definitions
the library was built with (`EIGEN_USE_BLAS`, `FASTEIG_USE_CUDA`, and so on),
and the full math and CUDA link line.

### Installing with Spack

A `package.py` is provided for Spack-managed installs. Add the repo, then:

```bash
spack install fasteig                          # CPU, default BLAS
spack install fasteig +mkl                     # CPU with MKL
spack install fasteig +cuda cuda_arch=80       # GPU
spack install fasteig +cuda cuda_arch=80 +cudss
```

Check the concretization before committing to a build:

```bash
spack spec -I fasteig +cuda cuda_arch=80 +cudss
```

Confirm that exactly one `cuda` node appears and that its major version matches
the cuDSS release being installed. `cuda_arch` is mandatory for `+cuda` builds
because Spack normally builds on a login node where no GPU is visible.

---

## Building with make.inc

1. Modify the `HOME` macro in `make.inc` to point to where `crd-fasteig` is
   installed.
2. Define the locations of MKL and Eigen.
3. Set the compiler and compiler flags.

Then:

```bash
make lib
```

This generates `libfasteig_$(PLAT).a`, where `PLAT` is a user-defined platform
name such as `Perlmutter`.

### Perlmutter notes: MKL vs Eigen

By default, `make.inc` enables Eigen's MKL integration (`-DEIGEN_USE_MKL_ALL`)
and links MKL. To build **without** MKL/Pardiso, using Eigen's own sparse
factorization instead, override `USE_MKL` and avoid linking MKL when building
the examples:

```bash
make -C src clean
make -C src MAKEINC=../make.inc USE_MKL=""
make -C examples clean
make -C examples MAKEINC=../make.inc USE_MKL="" MKL=""
```

When running a generalized problem, the log prints the factorization backend in
use — `MKL PardisoLDLT` when MKL is enabled, `Eigen SimplicialLDLT` when it is
not.

### Optional project-space cuDSS install

If your system does not provide a cuDSS module, install cuDSS into the
checkout:

```bash
tools/install_cudss.sh
make -C examples USE_CUDA=1
```

If pip cannot download cuDSS, download the Linux cuDSS archive from NVIDIA and
extract it with:

```bash
tools/install_cudss.sh --tarball /path/to/cudss-linux-*.tar.*
make -C examples USE_CUDA=1
```

The default makefile checks for the checkout-local install automatically. On
Perlmutter, a shared project-space install such as
`/global/cfs/cdirs/m1027/Dong/cudss` is also auto-detected when the checkout is
under `/global/cfs/cdirs/m1027/Dong/`. For any other custom cuDSS location,
source the generated `cudss-env.sh` or pass `CUDSS_PATH` and `CUDSS_LIBDIR`
explicitly.

---

## Running the examples

With the Make build:

```bash
cd examples
make
```

This generates a CPU executable called `runfasteig_cpu`. With CUDA enabled
(`make USE_CUDA=1` and `make -C examples USE_CUDA=1`), the examples build also
generates `runfasteig_gpu`.

```bash
./runfasteig_cpu SiO.input
./runfasteig_gpu SiO.input
```

For the executor-aware solvers, the CPU and GPU executables use the same input
deck. Keep the solver name in the input file and select the backend by
executable.

Both Make and CMake build `runfasteig_cpu`, plus `runfasteig_gpu` when CUDA is
enabled. CMake places them in `<build-dir>/examples/`. A CUDA-enabled build
contains both executables, so the same build can test both executors.
See [the testing guide](tests/README.md) for default SiO/SR reference tests,
optional generalized Kuu/Muu tests, CTest, and Perlmutter launch settings.

### Input file format

`SiO.input` contains a list of problem descriptions:

```
amatrix ./Sparse_Matrices/SiO.mtx
logfile SiO
nev     5
which   SR
solver  Chebyshev
maxiter 10
maxdim  20
degree  8
tol     1e-6
```

### Solver and backend support

The GPU executable dispatches `Lanczos`, `BlockLanczos`, `KrylovSchur`,
`BlockKrylovSchur`, `Davidson`, `Chebyshev`, and `LOBPCG` through the CUDA executor. These CUDA paths
support standard stored-matrix problems. `Lanczos`, `BlockLanczos`, `KrylovSchur`,
and `BlockKrylovSchur` also support generalized `A*x = lambda*B*x` on CPU and CUDA
for `SR`, `LR`, `SM`, `LM`, `BE`, and `SAI`, with symmetric A and symmetric
positive-definite B. Other solvers still require standard problems.

`which SAI` is available for `Lanczos`, `BlockLanczos`, `KrylovSchur`, and
`BlockKrylovSchur` on standard problems. It targets the eigenvalues nearest
`shift` by applying `(A - shift*I)^-1`; the factorization is reused throughout
the solve, including restarts. CUDA requires NVIDIA cuDSS; CPU uses the existing
MKL PardisoLDLT or Eigen SimplicialLDLT backend. Davidson, Chebyshev, and LOBPCG
do not support `SAI`.

For any of these generalized Krylov solvers, add `bmatrix path/to/B.mtx`. Ordinary selection
uses `B^-1 A` with one B factorization. For all four solvers, `which SAI` uses
`(A - shift*B)^-1 B`, factors only `A - shift*B`, and returns eigenvalues nearest
the shift. CUDA requires cuDSS for **all** generalized choices, not only SAI.
The shifted matrix may be indefinite; the CUDA path reuses cuDSS General/Full.
A singular shift is not supported. Indefinite or singular B is not supported.

Both executors use B-inner-product orthogonalization and rotate Q and BQ at
restart. The basis, BQ, projected matrix, and candidate eigenvectors stay on the
selected executor. Setup diagnostics precede progress, and total runtime includes
factorization. Progress residuals are projected operator estimates (in the B
inner product for generalized problems); convergence is verified using the original
relative residual `||A*x-lambda*B*x|| / ((1+|lambda|)*||B*x||)`. The same formula
is used in the final table. Very tight requested tolerances may be unattainable
for ill-conditioned B or a shift very close to an eigenvalue. If generalized
Krylov solvers reach their limit without satisfying `tol`, they return status 1
(the driver exits with code 2) and still print the best available eigenpairs.

Generalized block solvers use B-weighted block Gram–Schmidt and Cholesky QR:
`G = Z^T B Z = R^T R`, `Q = Z R^-1`. Rank loss triggers reorthogonalization and
deterministic completion of missing basis directions. Restarted methods retain
both Q and BQ. At full dimension, padded columns of a partial last block are excluded from
the projected eigensolve. Large blocks remain on the executor; small Gram,
projection, and rank-decision data may be copied to the host.

`maxdim` can enlarge the Krylov–Schur restart window above its minimum of
`max(20, 2*nev)` (capped at the matrix dimension). Direct `SM` is supported but
may converge slowly for interior eigenvalues; SAI near zero often helps.

For example, keep the same deck for `runfasteig_cpu` and `runfasteig_gpu`:

```text
amatrix Sparse_Matrices/SiO.mtx
solver BlockLanczos
which SAI
shift 2
nev 5
blksize 5
maxdim 20
maxiter 100
tol 1e-12
seed 1
```

Switch `solver` to `BlockKrylovSchur` to use restarts. Block Lanczos uses
`blksize` and counts block expansions in `maxiter`; Block Krylov–Schur also
uses `blksize`, independently of `nev`, but counts restart cycles. Its effective
cycle dimension is `maxdim` rounded down to a multiple of `blksize` (unless
`maxdim >= n`, which uses the full space, including a partial final block).
For a restarted run, provide at least `ceil(nev/blksize) + 1` blocks so the
requested Ritz vectors fit and a full block remains for expansion. It retains
the requested pairs plus guard Ritz vectors up to roughly half the cycle,
rounded to whole blocks. A very small cycle can converge slowly: for SiO with
`nev 5`, `blksize 5`, and `tol 1e-8`, prefer trying `maxdim 40` over `maxdim 10`.
Block Krylov–Schur verifies the original-problem residuals before accepting
convergence and returns a nonzero status if the tolerance is not met.
Lanczos and both block solvers report backtransformed eigenvalues in SAI progress; the
progress residuals remain relative estimates for the inverse operator. Final
eigenpairs are sorted algebraically and final residuals are evaluated against
the original problem (`A*x-lambda*x` or `A*x-lambda*B*x`).

Chebyshev supports `SR` and `LR` for standard problems on both executors. Its
filter recurrence, trial blocks, Cholesky QR, deflation, Ritz-vector rotation,
and residual evaluation use executor memory. The small projected eigensolve
uses the shared host Rayleigh–Ritz path, as in Davidson; only small matrices
and convergence data cross to the host during iteration. The final requested
eigenvectors are copied back once. Use the same `seed` and input deck when
comparing CPU and GPU runs. `print_every` controls progress output, while
convergence is checked after every filter cycle.

LOBPCG supports `SR` and `LR` for standard problems on both executors. Its
block width is `nev`. The current vectors `X`, residual directions `W`, previous
search directions `P`, trial basis, and operator images stay in executor memory.
The shared block Gram--Schmidt/Householder QR path removes dependent directions;
the small projected eigensolve uses the same host Ritz extraction as Davidson
and Chebyshev. No preconditioner is currently applied (`W` starts from the
residuals). Convergence is checked each iteration, and `print_every` controls
progress output. Use the same input deck and `seed` for CPU/CUDA comparisons.
`SM`, `LM`, `BE`, `SAI`, and generalized problems are not supported by LOBPCG.

---

## Version and build information

Every fasteig library records its release version, Git commit, source-tree
state, build system, configure options, compiler and compiler flags, selected
math backends, and CUDA configuration. The example driver exposes that metadata
without requiring an input deck:

```bash
./runfasteig_cpu --version
./runfasteig_cpu --build-info
```

For CMake builds, use `build/examples/runfasteig_cpu` instead. Development builds
from a Git checkout append the abbreviated commit to the release version and
mark builds made from a modified source tree as `dirty`.

Applications can query the same information through the installed public API:

```cpp
#include <version.hpp>

std::cout << fasteig::version() << '\n';
const fasteig::BuildInfo& info = fasteig::build_info();
std::cout << info.git_commit << '\n';
```

Numeric macros such as `FASTEIG_VERSION_MAJOR` and comparison helpers such as
`FASTEIG_VERSION_GE(1, 0, 0)` are also available for compile-time checks.
When configuring from a source archive without `.git`, package maintainers can
preserve the revision with `-DFASTEIG_GIT_COMMIT=<full-sha>` and optionally
`-DFASTEIG_GIT_DIRTY=0`.

---

## AI-Assisted Development Notice

This software contains components generated by Large Language Model (LLM) or machine intelligence platforms. This generated code may consist of entire code blocks, code reviews, or in some cases, entire applications.

The purpose of this document is to provide transparency regarding the use of artificial intelligence in the creation and maintenance of this project.

---

## Scope of usage

AI tools have been used to assist in various stages of the development lifecycle. 

---

## Human oversight and verification

AI tools were employed to accelerate development and identify potential issues, and a human reviewed AI-generated content. 

---

## Reliability

Code generated by LLMs may occasionally contain subtle logical errors or inefficiencies that were not detected during review.

---

## Support

For installation help, usage questions, and bug-reporting instructions, see
[SUPPORT.md](SUPPORT.md).

