# Tests (xSDK M2-oriented)

This directory provides a small, user-runnable test workflow that does **not** require commercial software.

## Scope (current)

- Uses `SiO.mtx` as the single integration-test matrix (select via `--matrix SiO`).
- Uses a *reference eigenvalue file* generated offline (currently via ScaLAPACK on Perlmutter),
  stored as `tests/reference/SiO.out.121mpi`.
- Validates FASTEIG by checking:
  - eigenvalues match the reference eigenvalues (within a configurable tolerance), and
  - FASTEIG-reported **final residual norms** are <= the requested tolerance (`--check-tol`).

Eigenvectors can be added later (e.g., to compute true residuals `||Av - λv||`).

## Prerequisites

- A FASTEIG executable must be built. The harness auto-detects `examples/runfasteig`
  first, then `examples/runfasteig_cpu`; override with `--runfasteig` if needed.
- Python 3 (no SciPy required).
- A reference eigenvalue file for the matrix under test:
  - default: `tests/reference/<matrix_stem>.out.121mpi`
  - or pass `--ref-file /path/to/ref.out`

Example (local):

```sh
make -C src MAKEINC=../make.inc.mac
make -C examples MAKEINC=../make.inc.mac
```

If your executable is named differently, pass it explicitly:

```sh
python3 tests/scripts/run_suite.py --matrix SiO --runfasteig examples/runfasteig
python3 tests/scripts/run_suite.py --matrix SiO --runfasteig runfasteig
```

## Run smoke tests (k=5)

Run all supported solvers (Lanczos, BlockLanczos, KrylovSchur, LOBPCG, Davidson, Chebyshev):

```sh
python3 tests/scripts/run_suite.py --matrix SiO
```

Smallest and largest algebraic eigenvalues:

```sh
python3 tests/scripts/run_suite.py --solver Lanczos --matrix SiO
python3 tests/scripts/run_suite.py --solver KrylovSchur --matrix SiO
```

Or run a single case:

```sh
python3 tests/scripts/run_case.py --solver Lanczos --which SR --matrix SiO
python3 tests/scripts/run_case.py --solver Lanczos --which LR --matrix SiO
```

Chebyshev uses the same reference checks for CPU and CUDA. After building with
`make -C examples USE_CUDA=1`, run these from the repository root on a GPU node:

```sh
python3 tests/scripts/run_suite.py --solver Chebyshev --matrix SiO --runfasteig examples/runfasteig_cpu
python3 tests/scripts/run_suite.py --solver Chebyshev --matrix SiO --runfasteig examples/runfasteig_gpu
```

Each command checks both `SR` and `LR`. For a CMake CUDA build, pass
`--runfasteig build-cuda/examples/runfasteig` instead. Chebyshev now uses the
same residual normalization for progress, convergence, and final results:
`||Ax - lambda*x|| / ((1 + |lambda|) * ||x||)`.

LOBPCG uses the same CPU/CUDA SR/LR reference checks and residual normalization:

```sh
python3 tests/scripts/run_suite.py --solver LOBPCG --matrix SiO --runfasteig examples/runfasteig_cpu
python3 tests/scripts/run_suite.py --solver LOBPCG --matrix SiO --runfasteig examples/runfasteig_gpu
```

Use the GPU executable on a GPU node after building with `USE_CUDA=1`. Both
commands use the same seed by default. Large LOBPCG blocks stay in executor
memory; the small projected eigensolve runs on the host. LOBPCG currently
supports only standard `SR`/`LR` problems, without preconditioning.

Specify a different `.mtx` file (MatrixMarket coordinate) by passing a path to `--matrix`:

```sh
python3 tests/scripts/run_case.py --solver Lanczos --which SR \\
  --matrix examples/Sparse_Matrices/SiO.mtx
```

Notes:
- `SR` tests the smallest (algebraic) `k` eigenvalues from the reference file.
- `LR` tests the largest (algebraic) `k` eigenvalues from the reference file.
- By default the test runs FASTEIG with `--fasteig-tol 1e-14`, checks residuals with `--check-tol 1e-10`,
  and checks eigenvalues with the mixed criterion `|Δ| <= eig_atol + eig_rtol*|ref|` using defaults
  `eig_atol=eig_rtol=1e-10`.
- Default FASTEIG `maxiter` used by the harness is `2000` (override with `--maxiter`).
- The harness passes a fixed `seed` into FASTEIG (default: `--seed 1`) so different solvers start from the same initial vectors.

## Batch environments (Slurm/PBS)

In batch-only environments, the workflow is the same: build your executable, then run the Python scripts
as part of your job step (e.g., inside `srun` if needed).

On Perlmutter, load Python before running the harness:

```sh
module load python
python3 tests/scripts/run_suite.py --matrix SiO
```

