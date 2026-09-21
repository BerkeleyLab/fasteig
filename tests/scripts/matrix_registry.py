from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import Dict


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


MATRIX_ALIASES: Dict[str, Path] = {
    "SIO": _repo_root() / "examples" / "Sparse_Matrices" / "SiO.mtx",
    "KUUMUU": _repo_root() / "examples" / "Sparse_Matrices" / "Kuu.mtx",
}


@dataclass(frozen=True)
class TestProblem:
    name: str
    matrix: Path
    bmatrix: Path | None
    reference: Path
    shift: float


PROBLEMS = {
    "SiO": TestProblem("SiO", MATRIX_ALIASES["SIO"], None,
                       _repo_root() / "tests/reference/SiO.out.121mpi", 2.0),
    "KuuMuu": TestProblem("KuuMuu", MATRIX_ALIASES["KUUMUU"],
                          _repo_root() / "examples/Sparse_Matrices/Muu.mtx",
                          _repo_root() / "tests/reference/KuuMuu_eig.txt", 100.0),
}


def resolve_problem(matrix: str, bmatrix: str = "", reference: str = "",
                    shift: float | None = None) -> TestProblem:
    from ref_eigs import default_ref_path_for_matrix

    registered = next((value for key, value in PROBLEMS.items()
                       if key.upper() == matrix.strip().upper()), None)
    matrix_path = resolve_matrix_arg(matrix)
    mass_path = resolve_matrix_arg(bmatrix) if bmatrix else (
        registered.bmatrix if registered else None)
    if reference:
        reference_path = resolve_matrix_arg(reference)
    elif registered and (not bmatrix or mass_path == registered.bmatrix):
        reference_path = registered.reference
    elif mass_path is not None:
        raise ValueError("A custom generalized problem requires --ref-file for that (A,B) pair.")
    else:
        reference_path = default_ref_path_for_matrix(matrix_path)
    return TestProblem(registered.name if registered else matrix_path.stem,
                       matrix_path, mass_path, reference_path,
                       shift if shift is not None else registered.shift if registered else 0.0)


def matrix_dimension(path: Path) -> int:
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip() and not line.startswith("%"):
                rows, columns, _ = map(int, line.split())
                if rows <= 0 or rows != columns:
                    raise ValueError(f"Expected a nonempty square matrix: {path}")
                return rows
    raise ValueError(f"No MatrixMarket dimensions found: {path}")


def resolve_matrix_arg(matrix: str) -> Path:
    """
    Resolve a `--matrix` argument.

    Accepted forms:
    - Alias name (case-insensitive), e.g. "SiO"
    - A filesystem path, e.g. "examples/Sparse_Matrices/SiO.mtx"
    """
    key = matrix.strip()
    if not key:
        raise ValueError("--matrix must be non-empty")

    alias = MATRIX_ALIASES.get(key.upper())
    if alias is not None:
        return alias

    # Treat as path (relative to repo root unless absolute).
    p = Path(key).expanduser()
    if not p.is_absolute():
        p = _repo_root() / p
    return p.resolve()

