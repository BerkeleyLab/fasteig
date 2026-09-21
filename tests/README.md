# FASTEIG reference tests

The manual harness and CTest use the same problem registry, defaults, reference
selection, launch profiles, and numerical checks. Python 3.10+ is required;
NumPy, SciPy, MATLAB, and ScaLAPACK are **not** needed to run the tests.

## Default coverage

By default, `run_suite.py` and CTest check **SR only** for the standard **SiO**
problem with these seven solvers: Lanczos, BlockLanczos, KrylovSchur,
BlockKrylovSchur, LOBPCG, Davidson, and Chebyshev.

CTest registers **7 CPU numerical tests**, plus **7 GPU numerical tests** when
`runfasteig_gpu` is built. The manual suite runs the same seven cases using the
executable selected with `--runfasteig`. Generalized problems and other spectrum
selections are **opt-in manual tests**, not part of default CTest coverage.

## Optional problems and selections

| `--matrix` | A | B | Full reference spectrum | Default SAI shift |
| --- | --- | --- | --- | --- |
| `SiO` | `examples/Sparse_Matrices/SiO.mtx` | Identity | `tests/reference/SiO.out.121mpi` (ScaLAPACK) | 2 |
| `KuuMuu` | `examples/Sparse_Matrices/Kuu.mtx` | `examples/Sparse_Matrices/Muu.mtx` | `tests/reference/KuuMuu_eig.txt` (MATLAB `eig`) | 100 |

Generalized problems are `A*x = lambda*B*x`, with symmetric A and SPD B.

| Solver | Standard (SiO) | Generalized (KuuMuu) |
| --- | --- | --- |
| Lanczos | SR, LR, SM, LM, BE, SAI | SR, LR, SM, LM, BE, SAI |
| BlockLanczos | SR, LR, SM, LM, BE, SAI | SR, LR, SM, LM, BE, SAI |
| KrylovSchur | SR, LR, SM, LM, BE, SAI | SR, LR, SM, LM, BE, SAI |
| BlockKrylovSchur | SR, LR, SM, LM, BE, SAI | SR, LR, SM, LM, BE, SAI |
| LOBPCG | SR, LR | Not supported |
| Davidson | SR, LR | Not supported |
| Chebyshev | SR, LR | Not supported |

The table lists the available manual test combinations, not the default suite.
Davidson's additional selections are outside this harness's test policy.
Unsupported combinations appear as explicit skips in manual runs.

CUDA generalized cases and all CUDA SAI cases require cuDSS. Missing runtime
resources are not numerical passes; see the skip policy below.

## Defaults and checks

| Setting | Default |
| --- | --- |
| `--matrix`, `--which` | SiO, SR only |
| `--k` | 5 |
| `--fasteig-tol` | `1e-8` |
| `--check-tol` | `1e-8` |
| `--eig-atol`, `--eig-rtol` | `1e-10` each (unchanged) |
| `--maxiter` | **2000** |
| `--blksize` | 10 for BlockLanczos and BlockKrylovSchur |
| `--maxdim` | 40 for KrylovSchur and BlockKrylovSchur |
| `--seed` | 1 |
| `--print-every`, `--verbosity` | 0, 1 |
| `--timeout` | 3600 seconds per solver run |

The block width is independent of `k`. For BlockLanczos, `maxiter` counts block
expansions; for BlockKrylovSchur it counts restart cycles. Scalar Lanczos's
iteration limit is capped at the matrix dimension. BlockLanczos allocates its
basis and projected workspace for the full block-iteration budget at startup
(capped at the available dimension for generalized problems). For SiO, the
default 2000 blocks of width 10 require approximately 8.6 GB for the basis and
projected matrix alone, before additional CPU/CUDA workspace. Reduce `--maxiter`
or `--blksize` if memory is limited. Unrestarted methods can also consume
substantial time if convergence is slow.

Each case must return exactly `k` finite eigenvalues and final residuals.
The harness reads the **full-precision** result table in the solver's log.
It checks:

- Each reported final relative residual is nonnegative and `<= check_tol`.
- `abs(computed - reference) <= eig_atol + eig_rtol * abs(reference)`.
- The executable exits successfully. Crashes, nonzero exits, timeouts, missing
  rows, malformed logs, and nonfinite results fail the case.

`RelΔ` in the table means `abs(computed-reference)/(1+abs(reference))`.
Residuals are reported by FASTEIG; the harness does not independently recompute
them from exported eigenvectors. For SAI, checks use original-problem final
eigenpairs/residuals, not inverse-operator progress estimates. A case contributes
**at most one failure**, even if both checks fail. A failing case does not abort
the remaining suite.

Reference selection preserves multiplicities:

- SR: smallest algebraic values, ascending.
- LR: largest algebraic values, descending.
- SM / LM: smallest / largest magnitude, respectively.
- BE: alternate largest, smallest, starting at the largest (3 high + 2 low for `k=5`).
- SAI: take the `k` nearest the shift, then order algebraically ascending.

Selection support does not guarantee convergence within the chosen budget.
Unshifted interior SM can be especially slow. Such failures remain visible;
the harness does not silently switch to SAI, relax tolerances, or extend limits.

## CMake / CTest

From the repository root:

```sh
cmake -S . -B build-test -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_EXAMPLES=ON -DBUILD_TESTS=ON
cmake --build build-test --parallel 4
ctest --test-dir build-test --output-on-failure
```

On a **new** test build, `ENABLE_CUDA` defaults to `AUTO`: CMake checks for a
CUDA compiler. It builds `runfasteig_cpu` always, and `runfasteig_gpu` when CUDA
is enabled. Both execute the same solver inputs. Explicit `-DENABLE_CUDA=OFF`
forces CPU-only; `-DENABLE_CUDA=ON` requires CUDA. An existing cached choice is
preserved; use `-DENABLE_CUDA=AUTO` to re-enable detection. A detected toolkit
must still compile successfully and have its dependencies installed/configured.

Configuration does **not** require a visible GPU. Perlmutter defaults to
architecture 80; elsewhere specify `-DCMAKE_CUDA_ARCHITECTURES=...` for the
target GPU when needed. CUDA runtime availability is checked at test time,
in the same launch context as the solver. Compilation and test registration do
not run the numerical tests; invoke `ctest` to run them.

```sh
ctest --test-dir build-test -N                         # inventory
ctest --test-dir build-test -L cpu --output-on-failure
ctest --test-dir build-test -L gpu --output-on-failure
ctest --test-dir build-test -R 'reference_cpu_SiO_.*_SR' --output-on-failure
```

Numerical tests have `reference`, `cpu`/`gpu`, `standard`, `SiO`, solver, and `SR`
labels. They share a CTest resource lock so that `ctest -j`
does not launch competing numerical jobs on the same node/GPU. Existing version
API/CLI tests remain. A fixture builds the drivers before selected tests run.

Optional CMake settings (shared by all registered reference tests):

```sh
cmake -S . -B build-test -DFASTEIG_TEST_PROFILE=perlmutter \
  -DFASTEIG_TEST_LAUNCHER=auto -DFASTEIG_TEST_TIMEOUT=3600
```

`FASTEIG_TEST_PROFILE` accepts `auto`, `local`, `perlmutter`;
`FASTEIG_TEST_LAUNCHER` accepts `auto`, `direct`, `srun`.
CTest allows 120 additional seconds beyond the solver timeout for probes and
harness overhead. Long cases may require larger limits and an adequate allocation.

## Manual suites

For Make builds, run from the repository root:

```sh
python3 tests/scripts/run_suite.py --matrix SiO --runfasteig examples/runfasteig_cpu
python3 tests/scripts/run_suite.py --matrix SiO --runfasteig examples/runfasteig_gpu
```

For CMake, substitute `build-test/examples/runfasteig_cpu` or
`build-test/examples/runfasteig_gpu`. Without `--runfasteig`, the harness prefers
`examples/runfasteig_cpu`, then `examples/runfasteig` for compatibility. The
driver must be rebuilt to support the new `--backend` query; backend selection
is not inferred from its directory name or the library's CUDA build flag.

Omitting `--matrix` selects SiO. Omitting `--which` selects SR only, including
for `run_case.py`. Limit the suite or run one case:

```sh
python3 tests/scripts/run_suite.py --matrix SiO --solver Davidson --solver Chebyshev
python3 tests/scripts/run_suite.py --matrix SiO --solver Lanczos --which SR --which LR
python3 tests/scripts/run_case.py --matrix SiO --solver Lanczos
python3 tests/scripts/run_case.py --matrix SiO --solver BlockLanczos --which SAI --shift 2
```

### Optional generalized tests

Selecting `--matrix KuuMuu` uses Kuu as A, Muu as B, and the MATLAB reference.
The suite then defaults to SR for the four supported Krylov solvers. Choose
other selections explicitly; the same commands work for either executable:

```sh
python3 tests/scripts/run_suite.py --matrix KuuMuu --runfasteig examples/runfasteig_cpu
python3 tests/scripts/run_suite.py --matrix KuuMuu --runfasteig examples/runfasteig_gpu
python3 tests/scripts/run_suite.py --matrix KuuMuu --which SAI --runfasteig examples/runfasteig_cpu
python3 tests/scripts/run_case.py --matrix KuuMuu --solver KrylovSchur --which SAI \
  --runfasteig examples/runfasteig_gpu
```

SAI uses shift 100 for KuuMuu and 2 for SiO unless `--shift` is supplied.
Repeat `--which` to select several modes, e.g. `--which SR --which LR --which SAI`.

### Inspecting runs

Add `--dry-run` to inspect generated decks and launch commands without running
solvers; the executable's metadata is still queried. `--keep-tmp` retains the
generated inputs, captured stdout, and logs in the printed artifact directory.
Use `--traceback` for Python diagnostics. Temporary paths are unique per run and
case, so CPU/GPU logs do not overwrite each other.

## Perlmutter: automatic launch settings

Load Python and obtain a suitable **existing** Slurm allocation first:

```sh
module load python
```

Run the Python suite or `ctest` from the allocation shell (or an `sbatch` script).
When `NERSC_HOST=perlmutter` or the hostname starts with `perlmutter`, the auto
profile uses these settings **only for child solver/probe processes**:

| Setting | CPU | GPU |
| --- | --- | --- |
| `OMP_NUM_THREADS`, `MKL_NUM_THREADS` | 64 | 16 |
| `OMP_PLACES` | cores | cores |
| `OMP_PROC_BIND` | spread | close |
| `OMP_DYNAMIC`, `MKL_DYNAMIC` | false | false |
| Launch | `srun -N 1 -n 1 -c 64 --cpu-bind=cores` | `srun -n 1 -c 16 --gpus-per-task=1 --gpu-bind=single:1 --cpu-bind=cores` |

The launcher also supplies `--immediate=5` so unavailable step resources do not
leave tests waiting indefinitely. It does **not** request an allocation, choose
an account, or change your shell environment. The allocation must have enough
CPUs and, for GPU tests, a GPU. CPU and GPU suites can be run in separate
allocations using CTest labels. A GPU allocation can run both if it also provides
the requested CPU resources.

Inside an already-running numeric Slurm job step, auto mode runs directly to
avoid nested `srun`; that step must already have the desired CPU/GPU binding.
Use overrides when necessary:

```sh
python3 tests/scripts/run_suite.py --matrix SiO --profile perlmutter \
  --runfasteig examples/runfasteig_gpu
python3 tests/scripts/run_suite.py --matrix SiO --profile local --launcher direct --threads 8
```

Other machines use direct execution and inherit their environment unless
`--threads` is supplied. `--profile local --launcher direct` explicitly disables
Perlmutter handling.

### Unavailable resources versus failures

CTest supplies `--allow-unavailable`: missing required allocation, no allocated
GPU, a failed CUDA initialization probe, or a build without required cuDSS yields
exit code **77**, displayed as **Skipped**. A CPU-only build registers no GPU
tests. Real solver failures and numerical mismatches are never converted to
skips. Unexpected launcher errors or broken dynamic-library loading also fail.

Manual runs treat unavailable runtime resources as failures by default; add
`--allow-unavailable` to request the CTest behavior. Unsupported algorithm/problem
combinations always skip explicitly. An all-skipped suite returns 77; any failed
case returns 2; otherwise the suite returns 0. Review skip counts: an all-skipped
GPU run does not validate CUDA correctness.

## Custom references / MATLAB export

Both supplied spectra use `ev(i) = value` lines. Plain text with one finite
eigenvalue per line is also accepted. Indexed files must use contiguous unique
indices `1..n`; the spectrum length must equal the matrix dimension. Reference
files for generalized problems must be computed for the **pair**, not A alone.

From MATLAB, with the repository root as the current directory:

```matlab
fid = fopen(fullfile('tests', 'reference', 'KuuMuu_eig.txt'), 'w');
assert(fid ~= -1, 'Cannot open reference file');
fprintf(fid, '%.17e\n', KuuMuu_eig(:));
fclose(fid);
```

Custom problem example:

```sh
python3 tests/scripts/run_case.py --solver KrylovSchur --which SAI \
  --matrix /path/to/A.mtx --bmatrix /path/to/B.mtx \
  --ref-file /path/to/AB_eig.txt --shift 100
```

Relative matrix/reference paths are resolved from the repository root. A custom
generalized pair requires an explicit `--ref-file`.

## Script organization

- `scripts/run_case.py`: single-case CLI, shared defaults and checks, reference
  selection, input generation, and CPU/GPU/Perlmutter launch handling.
- `scripts/run_suite.py`: suite CLI and explicit solver/selection loop; imports
  shared case-running functions from `run_case.py`.
- `scripts/matrix_registry.py`: registered matrix/reference paths and shifts.
- `scripts/ref_eigs.py`: full reference-spectrum reader.
- `scripts/fasteig_log.py`: FASTEIG result-log reader.

CMake reads `run_suite.py --list-solvers` at configuration time to share the
seven-solver list, then registers one SiO/SR case per solver and built backend.

