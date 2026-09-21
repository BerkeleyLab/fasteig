from __future__ import annotations

from collections import Counter
from pathlib import Path
import shutil
import sys
import tempfile
import traceback

import run_case as case_runner


def main():
    parser = case_runner.argument_parser("Run FASTEIG reference cases (default: all seven solvers, SiO/SR).")
    parser.add_argument("--solver", choices=case_runner.SOLVERS, action="append",
                        help="Solver to test (repeatable; defaults to all supported solvers)")
    parser.add_argument("--which", choices=case_runner.SELECTIONS, type=str.upper, action="append",
                        help="Spectrum selection (repeatable; defaults to SR only)")
    parser.add_argument("--list-solvers", action="store_true",
                        help="List the default standard-problem solvers without running tests")
    args = parser.parse_args()
    if args.list_solvers:
        print("\n".join(case_runner.SOLVERS))
        return 0

    problem, dimension, spectrum = case_runner.load_problem(args, parser)
    default_solvers = case_runner.KRYLOV_SOLVERS if problem.bmatrix else case_runner.SOLVERS
    solvers = list(dict.fromkeys(args.solver or default_solvers))
    selections = list(dict.fromkeys(args.which or ["SR"]))
    runner = case_runner.Runner(args)
    case_runner.print_configuration(args, problem, runner)

    counts = Counter()
    root = Path(tempfile.mkdtemp(prefix="fasteig-tests-"))
    try:
        for solver in solvers:
            for which in selections:
                status = case_runner.run_case(args, problem, dimension, spectrum, solver,
                                              which, runner, root / f"{solver}_{which}")
                counts[status] += 1
        print(f"\n[SUITE] {counts['PASS']} passed, {counts['FAIL']} failed, "
              f"{counts['SKIP']} skipped" + (f", {counts['PLANNED']} planned" if args.dry_run else ""))
    finally:
        if args.keep_tmp:
            print(f"Artifacts       : {root}")
        else:
            shutil.rmtree(root)
    if counts["FAIL"]:
        return 2
    return 0 if counts["PASS"] or counts["PLANNED"] else case_runner.SKIP_CODE


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"[FAIL] {error}", file=sys.stderr)
        if "--traceback" in sys.argv:
            traceback.print_exc()
        sys.exit(2)

