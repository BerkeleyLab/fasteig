from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import traceback

from fasteig_log import parse_final_results
from matrix_registry import matrix_dimension, resolve_problem
from ref_eigs import parse_reference_spectrum


SOLVERS = ("Lanczos", "BlockLanczos", "KrylovSchur", "BlockKrylovSchur",
           "LOBPCG", "Davidson", "Chebyshev")
KRYLOV_SOLVERS = SOLVERS[:4]
BLOCK_SOLVERS = ("BlockLanczos", "BlockKrylovSchur")
RESTARTED_SOLVERS = ("KrylovSchur", "BlockKrylovSchur")
SELECTIONS = ("SR", "LR", "SM", "LM", "BE", "SAI")
DEFAULT_K = 5
DEFAULT_TOL = 1e-8
DEFAULT_EIG_TOL = 1e-10
DEFAULT_MAXITER = 2000
DEFAULT_BLOCK_SIZE = 10
DEFAULT_MAXDIM = 40
SKIP_CODE = 77


def selections(solver: str) -> tuple[str, ...]:
    return SELECTIONS if solver in KRYLOV_SOLVERS else ("SR", "LR")


def unsupported_reason(solver: str, which: str, generalized: bool) -> str | None:
    if solver not in SOLVERS:
        return f"Unknown solver: {solver}"
    if which not in selections(solver):
        return f"{solver}/{which} is not in this harness's test policy."
    if generalized and solver not in KRYLOV_SOLVERS:
        return f"{solver} does not support generalized problems."
    return None


def argument_parser(description: str):
    result = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    result.add_argument("--matrix", default="SiO", help="SiO, KuuMuu, or an A matrix path")
    result.add_argument("--bmatrix", default="", help="B matrix for a custom generalized problem")
    result.add_argument("--ref-file", default="", help="Full reference spectrum; inferred for registered problems")
    result.add_argument("--shift", type=float, help="SAI shift (SiO: 2; KuuMuu: 100; custom: 0)")
    for flag, default, help_text in (
        ("k", DEFAULT_K, "Requested eigenpair count"),
        ("maxiter", DEFAULT_MAXITER, "Iteration limit"),
        ("blksize", DEFAULT_BLOCK_SIZE, "Block-method width"),
        ("maxdim", DEFAULT_MAXDIM, "Krylov-Schur subspace dimension"),
        ("seed", 1, "Initial-vector seed"),
        ("print-every", 0, "Solver progress cadence"),
    ):
        result.add_argument("--" + flag, type=int, default=default, help=help_text)
    for flag, default, help_text in (
        ("fasteig-tol", DEFAULT_TOL, "Solver tolerance"),
        ("check-tol", DEFAULT_TOL, "Final residual tolerance"),
        ("eig-atol", DEFAULT_EIG_TOL, "Eigenvalue absolute tolerance"),
        ("eig-rtol", DEFAULT_EIG_TOL, "Eigenvalue relative tolerance"),
        ("timeout", 3600, "Per-solver-run timeout in seconds"),
    ):
        result.add_argument("--" + flag, type=float, default=default, help=help_text)
    result.add_argument("--verbosity", type=int, choices=(1, 2), default=1, help="Solver verbosity")
    result.add_argument("--runfasteig", default="", help="Executable; defaults to examples/runfasteig_cpu")
    result.add_argument("--profile", choices=("auto", "local", "perlmutter"), default="auto", help="Launch/environment profile")
    result.add_argument("--launcher", choices=("auto", "direct", "srun"), default="auto", help="Launcher override")
    result.add_argument("--threads", type=int, help="Override thread count and srun CPUs")
    result.add_argument("--allow-unavailable", action="store_true", help="Skip unavailable GPU/allocation/cuDSS (CTest mode)")
    result.add_argument("--keep-tmp", action="store_true", help="Keep generated inputs, stdout, and logs")
    result.add_argument("--dry-run", action="store_true", help="Print cases, decks, and launch settings without solving")
    result.add_argument("--traceback", action="store_true", help="Print Python tracebacks on failures")
    return result


class Unavailable(RuntimeError):
    pass


def resolve_executable(requested: str) -> Path:
    root = Path(__file__).resolve().parents[2]
    if requested:
        path = Path(requested).expanduser()
        candidates = [path, root / path, root / "examples" / path]
    else:
        candidates = [root / "examples" / name for name in ("runfasteig_cpu", "runfasteig")]
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()
    raise ValueError(f"Executable not found or not executable: {requested or candidates[0]}")


def launch_settings(backend: str, profile: str, launcher: str, threads: int | None,
                    environ=None, hostname=None):
    env = dict(os.environ if environ is None else environ)
    host = socket.gethostname() if hostname is None else hostname
    if profile == "auto":
        profile = "perlmutter" if (env.get("NERSC_HOST", "").lower() == "perlmutter"
                                   or host.startswith("perlmutter")) else "local"
    gpu = backend == "CUDA"
    count = threads or (16 if gpu else 64)
    if profile == "perlmutter":
        env.update(OMP_NUM_THREADS=str(count), MKL_NUM_THREADS=str(count),
                   OMP_PLACES="cores", OMP_PROC_BIND="close" if gpu else "spread",
                   OMP_DYNAMIC="false", MKL_DYNAMIC="false")
    elif threads:
        env.update(OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads))
    inside_step = env.get("SLURM_STEP_ID", "").isdigit()
    if launcher == "auto":
        launcher = "srun" if profile == "perlmutter" and not inside_step else "direct"
    prefix = []
    if launcher == "srun":
        prefix = ["srun", "--immediate=5"]
        if not gpu:
            prefix += ["-N", "1"]
        prefix += ["-n", "1", "-c", str(count)]
        if gpu:
            prefix += ["--gpus-per-task=1", "--gpu-bind=single:1"]
        prefix += ["--cpu-bind=cores"]
    return profile, prefix, env


class Runner:
    def __init__(self, args):
        self.args = args
        self.exe = resolve_executable(args.runfasteig)
        result = subprocess.run([str(self.exe), "--backend"], capture_output=True,
                                text=True, timeout=30)
        self.backend = result.stdout.strip()
        if result.returncode != 0 or self.backend not in ("CPU", "CUDA"):
            raise RuntimeError("The test driver must support --backend; rebuild the examples.\n"
                               + result.stdout + result.stderr)
        result = subprocess.run([str(self.exe), "--build-info"], capture_output=True,
                                text=True, timeout=30)
        if result.returncode != 0:
            raise RuntimeError("Cannot read executable build information: " + result.stderr)
        self.cudss = any(line.strip().startswith("cuDSS") and line.split(":")[-1].strip() == "Yes"
                         for line in result.stdout.splitlines())
        self.profile, self.prefix, self.env = launch_settings(
            self.backend, args.profile, args.launcher, args.threads)
        self.checked = False
        self.unavailable = None

    def command(self, argument):
        return self.prefix + [str(self.exe), str(argument)]

    def describe(self, argument):
        keys = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OMP_PLACES", "OMP_PROC_BIND",
                "OMP_DYNAMIC", "MKL_DYNAMIC")
        settings = " ".join(f"{key}={self.env[key]}" for key in keys if key in self.env)
        return f"{settings} {shlex.join(self.command(argument))}".strip()

    def check_runtime(self):
        if self.checked:
            if self.unavailable:
                raise Unavailable(self.unavailable)
            return
        if (self.profile == "perlmutter" or self.prefix) and not self.env.get("SLURM_JOB_ID"):
            self.unavailable = "An existing Slurm allocation is required; run inside salloc/sbatch."
        if self.profile == "perlmutter" and not self.prefix and self.env.get("SLURM_STEP_ID", "").isdigit():
            allocated = self.env.get("SLURM_CPUS_PER_TASK", "")
            required = int(self.env["OMP_NUM_THREADS"])
            if allocated.isdigit() and int(allocated) < required:
                self.unavailable = (f"The current Slurm step has {allocated} CPUs per task; "
                                    f"this profile needs {required}. Use a larger step or --threads.")
        if not self.unavailable and self.backend == "CUDA" and self.prefix:
            allocation = subprocess.run(
                ["scontrol", "show", "job", self.env["SLURM_JOB_ID"], "-o"],
                env=self.env, capture_output=True, text=True, timeout=30)
            resources = re.search(r"\bAllocTRES=(\S+)", allocation.stdout)
            if allocation.returncode == 0 and resources:
                gpu_counts = re.findall(r"gres/gpu(?::[^=,]+)?=(\d+)", resources.group(1))
                if not any(int(count) > 0 for count in gpu_counts):
                    self.unavailable = "The current Slurm allocation has no allocated GPUs."
        if self.unavailable:
            self.checked = True
            raise Unavailable(self.unavailable)
        if self.backend == "CUDA":
            result = subprocess.run(self.command("--check-runtime"), env=self.env,
                                    capture_output=True, text=True, timeout=90)
            if result.returncode == SKIP_CODE:
                self.unavailable = result.stdout.strip() or result.stderr.strip() or "CUDA runtime is unavailable."
                self.checked = True
                raise Unavailable(self.unavailable)
            if result.returncode:
                raise RuntimeError("CUDA runtime probe failed in the selected launch context:\n"
                                   + result.stdout + result.stderr)
        self.checked = True

    def run(self, input_path, directory):
        self.check_runtime()
        return subprocess.run(self.command(input_path), cwd=directory, env=self.env,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, timeout=self.args.timeout)


def select_reference(values, which, count, shift):
    ordered = sorted(values)
    if which == "SR":
        return ordered[:count]
    if which == "LR":
        return ordered[::-1][:count]
    if which == "SM":
        return sorted(ordered, key=abs)[:count]
    if which == "LM":
        return sorted(ordered, key=abs, reverse=True)[:count]
    if which == "BE":
        return [ordered[-1 - index // 2] if index % 2 == 0 else ordered[index // 2]
                for index in range(count)]
    if which == "SAI":
        return sorted(sorted(ordered, key=lambda value: abs(value - shift))[:count])
    raise ValueError(f"Unknown selection {which}")


def input_deck(args, problem, solver, which, dimension):
    iterations = min(args.maxiter, dimension) if solver == "Lanczos" else args.maxiter
    lines = [f"amatrix {problem.matrix}", "logfile results", "output_format text",
             f"solver {solver}", f"which {which}", f"nev {args.k}",
             f"tol {args.fasteig_tol:.17g}", f"maxiter {iterations}", f"seed {args.seed}",
             f"verbosity {args.verbosity}", f"print_every {args.print_every}"]
    if iterations != args.maxiter:
        print(f"Lanczos maxiter capped at matrix dimension: {iterations}")
    if problem.bmatrix is not None:
        lines.append(f"bmatrix {problem.bmatrix}")
    if solver in BLOCK_SOLVERS:
        if args.blksize > dimension:
            raise ValueError("--blksize exceeds the matrix dimension")
        lines.append(f"blksize {args.blksize}")
    if solver in RESTARTED_SOLVERS:
        lines.append(f"maxdim {min(args.maxdim, dimension)}")
    if which == "SAI":
        lines.append(f"shift {problem.shift:.17g}")
    return "\n".join(lines) + "\n"


def compare(results, reference, args):
    if len(results.eigenvalues) != args.k or len(results.residual_norms) != args.k:
        raise ValueError(f"Expected {args.k} eigenvalues AND residuals; got "
                         f"{len(results.eigenvalues)} and {len(results.residual_norms)}")
    failures = []
    print(f"{'Index':>5}  {'FASTEIG Eigenvalue':>24}  {'Ref Eigenvalue':>24}  {'RelΔ':>10}  {'FASTEIG Residual':>16}")
    for index, (got, expected, residual) in enumerate(zip(results.eigenvalues, reference,
                                                         results.residual_norms), start=1):
        relative = abs(got - expected) / (1.0 + abs(expected))
        print(f"{index:5d}  {got:24.16e}  {expected:24.16e}  {relative:10.3e}  {residual:16.3e}")
        if not math.isfinite(residual) or residual < 0 or residual > args.check_tol:
            failures.append(f"index {index}: residual {residual:.3e} exceeds check_tol {args.check_tol:.3e} or is invalid")
        if not math.isfinite(got) or abs(got - expected) > args.eig_atol + args.eig_rtol * abs(expected):
            failures.append(f"index {index}: eigenvalue RelΔ={relative:.3e} fails atol + rtol*|ref|")
    return failures


def run_case(args, problem, dimension, spectrum, solver, which, runner, directory):
    print(f"\n{'=' * 18} {problem.name} / {solver} / {which} {'=' * 18}", flush=True)
    reason = unsupported_reason(solver, which, problem.bmatrix is not None)
    if reason:
        print(f"[SKIP: unsupported] {reason}")
        return "SKIP"
    try:
        directory.mkdir(parents=True)
        deck = input_deck(args, problem, solver, which, dimension)
        input_path = directory / "case.input"
        input_path.write_text(deck)
        print("Command:", runner.describe(input_path), flush=True)
        if args.dry_run:
            print(deck.rstrip())
            print("[PLANNED]")
            return "PLANNED"
        if runner.backend == "CUDA" and not runner.cudss and (which == "SAI" or problem.bmatrix):
            raise Unavailable("This GPU case requires a build with cuDSS enabled.")
        reference = select_reference(spectrum, which, args.k, problem.shift)
        output = runner.run(input_path, directory)
        (directory / "stdout.txt").write_text(output.stdout)
        failures = []
        if output.returncode:
            failures.append(f"Executable returned {output.returncode}")
        try:
            results = parse_final_results(directory / f"results_{solver}.log")
            print(f"Iterations      : {results.iterations}\nRuntime (sec)   : {results.runtime_sec}")
            failures.extend(compare(results, reference, args))
        except (OSError, ValueError) as error:
            failures.append(str(error))
        if failures:
            print(output.stdout)
            print("[FAIL]\n  " + "\n  ".join(failures))
            return "FAIL"
        print("[PASS]")
        return "PASS"
    except subprocess.TimeoutExpired as error:
        output = error.stdout or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        (directory / "stdout.txt").write_text(output)
        print(f"[FAIL] Solver/probe timed out after {error.timeout} seconds")
        return "FAIL"
    except Unavailable as error:
        status = "SKIP" if args.allow_unavailable else "FAIL"
        print(f"[{status}: unavailable] {error}")
        return status
    except Exception as error:
        print(f"[FAIL] {error}")
        if args.traceback:
            traceback.print_exc()
        return "FAIL"


def load_problem(args, argument_parser):
    for name in ("k", "maxiter", "blksize", "maxdim", "timeout", "fasteig_tol", "check_tol"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            argument_parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if any(not math.isfinite(value) or value < 0 for value in (args.eig_atol, args.eig_rtol)):
        argument_parser.error("Eigenvalue tolerances must be finite and nonnegative")
    if args.threads is not None and args.threads <= 0:
        argument_parser.error("--threads must be positive")
    if args.seed < 0 or args.seed > 2**31 - 1 or args.print_every < 0:
        argument_parser.error("--seed must fit a nonnegative int32 and --print-every must be nonnegative")
    problem = resolve_problem(args.matrix, args.bmatrix, args.ref_file, args.shift)
    dimension = matrix_dimension(problem.matrix)
    if problem.bmatrix and matrix_dimension(problem.bmatrix) != dimension:
        raise ValueError("A and B dimensions differ")
    if args.k > dimension or not math.isfinite(problem.shift):
        raise ValueError("Invalid eigenpair count or shift")
    spectrum = parse_reference_spectrum(problem.reference).eigenvalues
    if len(spectrum) != dimension:
        raise ValueError(f"Expected a FULL spectrum of {dimension} eigenvalues; found {len(spectrum)}")
    return problem, dimension, spectrum


def print_configuration(args, problem, runner):
    print(f"Problem         : {problem.name} ({'generalized' if problem.bmatrix else 'standard'})")
    print(f"A matrix        : {problem.matrix}\nB matrix        : {problem.bmatrix or 'Identity'}")
    print(f"Reference       : {problem.reference}\nBackend         : {runner.backend}")
    print(f"FASTEIG tol     : {args.fasteig_tol}\nCheck tol       : {args.check_tol}")
    print(f"Eigenvalue atol : {args.eig_atol}\nEigenvalue rtol : {args.eig_rtol}")
    print(f"Maxiter         : {args.maxiter}\nk / seed        : {args.k} / {args.seed}")


def main():
    parser = argument_parser("Check one FASTEIG case against a full reference spectrum.")
    parser.add_argument("--solver", choices=SOLVERS, required=True, help="Solver to test")
    parser.add_argument("--which", choices=SELECTIONS, type=str.upper, default="SR",
                        help="Spectrum selection")
    args = parser.parse_args()
    problem, dimension, spectrum = load_problem(args, parser)
    runner = Runner(args)
    print_configuration(args, problem, runner)
    root = Path(tempfile.mkdtemp(prefix="fasteig-tests-"))
    try:
        status = run_case(args, problem, dimension, spectrum, args.solver, args.which,
                          runner, root / f"{args.solver}_{args.which}")
    finally:
        if args.keep_tmp:
            print(f"Artifacts       : {root}")
        else:
            shutil.rmtree(root)
    if status == "FAIL":
        return 2
    return SKIP_CODE if status == "SKIP" else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"[FAIL] {error}", file=sys.stderr)
        if "--traceback" in sys.argv:
            traceback.print_exc()
        sys.exit(2)

