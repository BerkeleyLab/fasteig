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
| `ENABLE_CUDA` | `OFF` | Build the NVIDIA GPU backend from `src/gpu`. |
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
ldd build-cuda/examples/runfasteig | grep cudss
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

> **Naming difference between build systems.** The CMake build produces a single
> `runfasteig` executable per build directory rather than the `_cpu`/`_gpu`
> pair, since the backend is fixed at configure time. Run
> `build/examples/runfasteig` for the CPU build and
> `build-cuda/examples/runfasteig` for the GPU build.

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
`BlockKrylovSchur`, and `Davidson` through the CUDA executor. These CUDA paths
currently support standard stored-matrix problems; generalized Krylov–Schur
remains host-only.

CUDA `which SAI` is available for scalar Lanczos and Krylov–Schur and requires
NVIDIA cuDSS. The block methods and Davidson do not support `SAI`.