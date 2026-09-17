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

Shift-invert cases use the same harness and reference spectrum. For example:

```sh
python3 tests/scripts/run_case.py --solver BlockLanczos --which SAI --shift 2 --matrix SiO --blksize 5 --maxiter 100
python3 tests/scripts/run_case.py --solver BlockKrylovSchur --which SAI --shift 2 --matrix SiO --blksize 5 --maxdim 20 --maxiter 100
```

Add `--runfasteig examples/runfasteig_gpu` to check CUDA SAI on a GPU node
with cuDSS, or pass the path to a CMake-built executable. `--which SAI` selects
the reference eigenvalues nearest `--shift` (default 0), then sorts them
algebraically to match FASTEIG's final output. Checks use the reported final
residuals against the original A, not the inverse-operator progress estimates.
`run_suite.py` continues to run SR/LR cases; use `run_case.py` for SAI.

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
- `SAI` tests the `k` eigenvalues nearest the specified shift (standard problems only).
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

### Generalized Krylov solvers

`run_case.py` accepts `SR`, `LR`, `SM`, `LM`, `BE`, and `SAI`, and an optional
`--bmatrix` for generalized Lanczos, BlockLanczos, KrylovSchur, and
BlockKrylovSchur. All four support SR/LR/SM/LM/BE; generalized SAI is restricted
to KrylovSchur. Supply an explicit `--ref-file` with
reference eigenvalues for **the pair (A,B)**, in the same `ev(i) = value` format
as the existing ScaLAPACK references. An A-only reference such as SiO's must not
be reused for a nonidentity B.

```bash
python3 tests/scripts/run_case.py --solver KrylovSchur \
  --matrix /path/to/A.mtx --bmatrix /path/to/B.mtx \
  --ref-file /path/to/AB.out --which SAI --shift 2 \
  --runfasteig examples/runfasteig_gpu --fasteig-tol 1e-10
```

Use `runfasteig_cpu` for comparison with the same inputs and seed. CUDA generalized
runs require cuDSS for all six spectrum selections. A must be symmetric and B
symmetric positive definite. `SM` and `LM` compare eigenvalues in magnitude order;
`BE` alternates largest/smallest, starting at the largest; `SAI` compares the
nearest eigenvalues in ascending algebraic order. For direct interior `SM`,
consider increasing `--maxdim` or using `SAI` with a nearby nonsingular shift.

For the other generalized solvers, for example:

```bash
python3 tests/scripts/run_case.py --solver BlockLanczos \
  --matrix /path/to/A.mtx --bmatrix /path/to/B.mtx \
  --ref-file /path/to/AB.out --which SR --blksize 5 --maxiter 200 \
  --runfasteig examples/runfasteig_gpu --fasteig-tol 1e-10
```

Use `--solver BlockKrylovSchur --blksize 5 --maxdim 30` for the restarted block method,
or `--solver Lanczos` for scalar Lanczos (its `maxiter` must not exceed n).
For small matrices the generalized block methods handle the final partial block
without introducing padded zero Ritz values. The harness checks both the
original-problem residuals and the selected eigenvalues against the reference.

Block Krylov-Schur uses `--blksize` independently of `--k`. Its iteration count
is the number of restart cycles, not individual matrix-vector products. For
example, compare SiO with the same seed and tolerance but different subspaces:

```bash
python3 tests/scripts/run_case.py --solver BlockKrylovSchur --which SR --matrix SiO --k 5 --blksize 5 --maxdim 40 --maxiter 3000 --fasteig-tol 1e-8 --check-tol 1e-8
```

The residual check tolerance here matches the requested solver tolerance; the
harness default of `1e-10` would be stricter. The solver now reports a nonzero
status when its original-problem residuals fail its requested tolerance,
including when `maxiter` is reached. Floating-point accuracy can limit very
strict tolerances even after projected residual estimates become tiny.

