from __future__ import annotations

import argparse
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import List

from fasteig_log import parse_final_results
from matrix_registry import resolve_matrix_arg
from ref_eigs import default_ref_path_for_matrix, parse_scalapack_output


def _run_fasteig(fasteig_exe: Path, input_path: Path, cwd: Path) -> None:
    proc = subprocess.run(
        [str(fasteig_exe), str(input_path)],
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"FASTEIG executable failed (code {proc.returncode}). Output:\n{proc.stdout}")


def _resolve_fasteig_exe(repo_root: Path, requested: str) -> Path:
    if requested:
        p = Path(requested).expanduser()
        candidates = [p if p.is_absolute() else (repo_root / p)]
        if not p.is_absolute() and len(p.parts) == 1:
            candidates.append(repo_root / "examples" / p)
    else:
        candidates = [
            repo_root / "examples" / "runfasteig",
            repo_root / "examples" / "runfasteig_cpu",
        ]

    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved

    tried = ", ".join(str(c.resolve()) for c in candidates)
    raise RuntimeError(f"FASTEIG executable not found. Tried: {tried}. Build it first (see tests/README.md).")


def _max_abs_diff(a, b) -> float:
    return max(abs(x - y) for x, y in zip(a, b))

def _eig_error_ok(*, got: float, ref: float, atol: float, rtol: float) -> bool:
    return abs(got - ref) <= (atol + rtol * abs(ref))

def _eig_error_rows(*, got_vals: List[float], ref_vals: List[float], atol: float, rtol: float):
    n = min(len(got_vals), len(ref_vals))
    bad = []
    max_abs = 0.0
    max_rel = 0.0
    max_allow = 0.0
    for i in range(n):
        dabs = abs(got_vals[i] - ref_vals[i])
        drel = dabs / (1.0 + abs(ref_vals[i]))
        allow = atol + rtol * abs(ref_vals[i])
        allow_rel = allow / (1.0 + abs(ref_vals[i]))
        max_abs = max(max_abs, dabs)
        max_rel = max(max_rel, drel)
        max_allow = max(max_allow, allow)
        if dabs > allow:
            bad.append((i + 1, drel, allow_rel))
    return bad, max_abs, max_rel, max_allow


def _print_comparison_table(
    *,
    got_vals: List[float],
    got_res: List[float],
    ref_vals: List[float],
) -> None:
    n = min(len(got_vals), len(ref_vals), len(got_res))
    print("")
    print("RESULTS (FASTEIG vs Reference)")
    print("------------------------------")
    w_idx = 5
    w_ev = 24
    w_drel = 10
    w_res = 14
    print(
        f"{'Index':>{w_idx}}  "
        f"{'FASTEIG Eigenvalue':>{w_ev}}  "
        f"{'Ref Eigenvalue':>{w_ev}}  "
        f"{'RelΔ':>{w_drel}}  "
        f"{'FASTEIG Residual':>{w_res}}"
    )
    for i in range(n):
        dabs = abs(got_vals[i] - ref_vals[i])
        drel = dabs / (1.0 + abs(ref_vals[i]))
        print(
            f"{i+1:>{w_idx}d}  "
            f"{got_vals[i]:>{w_ev}.16e}  "
            f"{ref_vals[i]:>{w_ev}.16e}  "
            f"{drel:>{w_drel}.3e}  "
            f"{got_res[i]:>{w_res}.3e}  "
        )
    print("")

def _banner(title: str) -> None:
    w = 78
    t = f" {title.strip()} "
    if len(t) >= w:
        print(t)
        return
    left = (w - len(t)) // 2
    right = w - len(t) - left
    print("=" * left + t + "=" * right)

def _print_kv(key: str, value: str) -> None:
    key_w = 16
    print(f"{key:<{key_w}}: {value}")

def _which_pretty(which: str) -> str:
    w = which.upper()
    if w == "SR":
        return "Smallest"
    if w == "LR":
        return "Largest"
    if w == "SAI":
        return "Nearest shift (SAI)"
    return which


def _select_ref_values(*, ref_path: Path, which: str, k: int, shift: float = 0.0) -> List[float]:
    spec = parse_scalapack_output(ref_path)
    w = which.upper()
    if w == "SR":
        return spec.smallest(k)
    if w == "LR":
        return spec.largest(k)
    if w == "SM":
        return sorted(spec.eigenvalues, key=abs)[:k]
    if w == "LM":
        return sorted(spec.eigenvalues, key=abs, reverse=True)[:k]
    if w == "BE":
        ordered = sorted(spec.eigenvalues)
        both = []
        lo, hi = 0, len(ordered) - 1
        while lo <= hi:
            both.append(ordered[hi])
            hi -= 1
            if lo <= hi:
                both.append(ordered[lo])
                lo += 1
        return both[:k]
    if w == "SAI":
        # SAI returns the k nearest eigenvalues in ascending algebraic order.
        return sorted(sorted(spec.eigenvalues, key=lambda value: abs(value - shift))[:k])
    raise ValueError(f"Unsupported which for reference test: {which} (use SR, LR, SM, LM, BE, or SAI)")


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare FASTEIG with ScaLAPACK reference eigenvalues on a MatrixMarket matrix.")
    ap.add_argument("--solver", required=True, help="Solver name (e.g., Lanczos, KrylovSchur)")
    ap.add_argument("--which", required=True, choices=["SR", "LR", "SM", "LM", "BE", "SAI"], help="Spectrum selection; SAI targets eigenvalues nearest --shift")
    ap.add_argument("--shift", type=float, default=0.0, help="Shift for SAI (default: 0)")
    ap.add_argument("--matrix", default="SiO", help="Matrix alias (e.g., SiO) or path to .mtx")
    ap.add_argument("--bmatrix", default="", help="SPD B matrix for a generalized Krylov solver; requires an explicit --ref-file for (A,B)")
    ap.add_argument(
        "--ref-file",
        default="",
        help="Reference eigenvalue file (ScaLAPACK output). Default: tests/reference/<matrix>.out.121mpi",
    )

    ap.add_argument("--k", type=int, default=5, help="Number of eigenvalues to compare (default: 5)")
    ap.add_argument("--fasteig-tol", type=float, default=1e-14, help="FASTEIG solver tolerance (default: 1e-14)")
    ap.add_argument("--check-tol", type=float, default=1e-10, help="Assertion tolerance for residual norms (default: 1e-10)")
    ap.add_argument("--maxiter", type=int, default=2000, help="FASTEIG maxiter (default: 2000)")
    ap.add_argument("--seed", type=int, default=1, help="Random seed passed to FASTEIG (default: 1)")
    ap.add_argument("--blksize", type=int, help="BlockLanczos/BlockKrylovSchur block width (otherwise use the input parser default)")
    ap.add_argument("--maxdim", type=int, help="Maximum restarted subspace dimension (otherwise use the input parser default)")

    ap.add_argument("--print-every", type=int, default=0, help="print_every in FASTEIG input (0 disables)")
    ap.add_argument("--verbosity", type=int, default=1, choices=[1, 2], help="Verbosity for FASTEIG stdout")

    ap.add_argument("--eig-atol", type=float, default=1e-10, help="Absolute eigenvalue tolerance (default: 1e-10)")
    ap.add_argument("--eig-rtol", type=float, default=1e-10, help="Relative eigenvalue tolerance (default: 1e-10)")

    ap.add_argument(
        "--runfasteig",
        default="",
        help="Path/name of FASTEIG executable. Default: auto-detect examples/runfasteig, then examples/runfasteig_cpu.",
    )
    ap.add_argument("--keep-tmp", action="store_true", help="Keep temporary test directory for debugging.")
    ap.add_argument("--traceback", action="store_true", help="On failure, print a Python traceback (default: off).")

    args = ap.parse_args()
    if args.bmatrix:
        if args.solver not in ("Lanczos", "BlockLanczos", "KrylovSchur", "BlockKrylovSchur") or not args.ref_file:
            ap.error("--bmatrix requires a Krylov solver and an explicit generalized --ref-file")
        if args.which == "SAI" and args.solver != "KrylovSchur":
            ap.error("generalized SAI is currently supported only by KrylovSchur")
    if args.k <= 0:
        ap.error("--k must be positive")
    if not math.isfinite(args.shift):
        ap.error("--shift must be finite")
    if args.blksize is not None and args.blksize <= 0:
        ap.error("--blksize must be positive")
    if args.maxdim is not None and args.maxdim <= 0:
        ap.error("--maxdim must be positive")

    repo_root = Path(__file__).resolve().parents[2]
    matrix_path = resolve_matrix_arg(args.matrix)
    bmatrix_path = resolve_matrix_arg(args.bmatrix) if args.bmatrix else None
    if bmatrix_path is not None and not bmatrix_path.exists():
        raise RuntimeError(f"B matrix not found: {bmatrix_path}")
    fasteig_exe = _resolve_fasteig_exe(repo_root, args.runfasteig)
    ref_path = (Path(args.ref_file).expanduser().resolve() if args.ref_file else default_ref_path_for_matrix(matrix_path))

    if not matrix_path.exists():
        raise RuntimeError(f"Matrix not found: {matrix_path}")
    if not ref_path.exists():
        raise RuntimeError(f"Reference file not found: {ref_path}")

    _banner("TEST CASE")
    _print_kv("Solver", args.solver)
    _print_kv("Which", _which_pretty(args.which))
    if args.which == "SAI":
        _print_kv("Shift", f"{args.shift:g}")
    if args.blksize is not None:
        _print_kv("Block size", str(args.blksize))
    if args.maxdim is not None:
        _print_kv("Max subspace dim", str(args.maxdim))
    _print_kv("Matrix", f"{args.matrix}  ({matrix_path})")
    if bmatrix_path is not None:
        _print_kv("B matrix", str(bmatrix_path))
    _print_kv("Ref file", str(ref_path))
    _print_kv("k", str(args.k))
    _print_kv("FASTEIG tol", f"{args.fasteig_tol:g}")
    _print_kv("Check tol", f"{args.check_tol:g}")
    if args.eig_atol == args.eig_rtol and args.eig_atol == args.check_tol:
        _print_kv("Eigenvalue tol", f"{args.check_tol:g}")
    else:
        _print_kv("Eigenvalue atol", f"{args.eig_atol:g}")
        _print_kv("Eigenvalue rtol", f"{args.eig_rtol:g}")
    _print_kv("FASTEIG maxiter", str(args.maxiter))
    _print_kv("FASTEIG exe", str(fasteig_exe))
    print("")

    _banner("REFERENCE (ScaLAPACK)")
    print("Loading reference eigenvalues...")
    ref_vals = _select_ref_values(ref_path=ref_path, which=args.which, k=args.k, shift=args.shift)
    print(f"Reference loaded  : {len(ref_vals)} values used (of full spectrum in {ref_path.name})")
    print("")

    # Run fasteig in a temp dir (keeps logs isolated).
    if args.keep_tmp:
        td_path = Path(tempfile.mkdtemp(prefix="fasteig_test_"))
        print(f"[INFO] Keeping temp dir: {td_path}")
        cleanup = None
    else:
        tmp_ctx = tempfile.TemporaryDirectory(prefix="fasteig_test_")
        td_path = Path(tmp_ctx.name)
        cleanup = tmp_ctx.cleanup

    try:
        stem = matrix_path.stem if matrix_path.suffix.lower() == ".mtx" else "matrix"
        logprefix = td_path / stem

        extra_options = []
        if bmatrix_path is not None:
            extra_options.append(f"bmatrix {bmatrix_path}")
        if args.which == "SAI":
            extra_options.append(f"shift {args.shift}")
        if args.blksize is not None:
            extra_options.append(f"blksize {args.blksize}")
        if args.maxdim is not None:
            extra_options.append(f"maxdim {args.maxdim}")
        input_txt = "\n".join(
            [
                f"amatrix {matrix_path}",
                f"logfile {logprefix}",
                "output_format text",
                f"seed {args.seed}",
                f"solver {args.solver}",
                f"which {args.which}",
                f"nev {args.k}",
                f"maxiter {args.maxiter}",
                f"tol {args.fasteig_tol}",
                f"verbosity {args.verbosity}",
                f"print_every {args.print_every}",
                *extra_options,
                "",
            ]
        )
        input_path = td_path / f"{stem}_test.input"
        input_path.write_text(input_txt, encoding="utf-8")

        _banner("FASTEIG")
        print("Running...")
        _run_fasteig(fasteig_exe, input_path, cwd=repo_root)
        print("Finished.")

        log_path = Path(f"{logprefix}_{args.solver}.log")
        if not log_path.exists():
            raise RuntimeError(f"Expected log file not found: {log_path}")

        print("Reading log...")
        results = parse_final_results(log_path)
        got_vals = results.eigenvalues[: args.k]
        got_res = results.residual_norms[: args.k]

        print("")
        _banner("FASTEIG DIAGNOSTICS")
        _print_kv("Iterations", str(results.iterations) if results.iterations is not None else "n/a")
        _print_kv(
            "Runtime (sec)",
            f"{results.runtime_sec:.6f}" if results.runtime_sec is not None else "n/a",
        )

        if len(got_vals) != args.k:
            raise RuntimeError(f"Expected {args.k} eigenvalues, got {len(got_vals)}")
        if len(ref_vals) != args.k:
            raise RuntimeError(f"Reference returned {len(ref_vals)} eigenvalues, expected {args.k}.")

        _banner("COMPARISON")
        _print_comparison_table(got_vals=got_vals, got_res=got_res, ref_vals=ref_vals)

        _banner("CHECKS")
        print("Checking residuals...")
        bad = [(i, r) for i, r in enumerate(got_res, start=1) if not (r <= args.check_tol)]
        if bad:
            msg = "\n".join([f"  index {i}: residual {r:.3e} > check_tol {args.check_tol:.3e}" for i, r in bad])
            raise RuntimeError(f"Residual norms exceed tolerance:\n{msg}")

        print("Checking eigenvalues...")
        eig_bad, max_abs, max_rel, max_allow = _eig_error_rows(
            got_vals=got_vals,
            ref_vals=ref_vals,
            atol=args.eig_atol,
            rtol=args.eig_rtol,
        )
        if eig_bad:
            msg = "\n".join([f"  index {i}: RelΔ={drel:.3e} > allowed RelΔ {allow_rel:.3e}" for i, drel, allow_rel in eig_bad])
            raise RuntimeError(
                "Eigenvalues differ from reference.\n"
                f"{msg}\n"
                f"  max RelΔ = {max_rel:.3e}\n"
            )
    finally:
        if cleanup is not None:
            cleanup()

    _banner("PASS")
    print(f"solver={args.solver} which={args.which} matrix={args.matrix} k={args.k}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as e:
        print(f"[FAIL] {e}", file=sys.stderr)
        if "--traceback" in sys.argv:
            raise
        raise SystemExit(2)

