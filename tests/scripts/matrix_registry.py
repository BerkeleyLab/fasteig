from __future__ import annotations

from pathlib import Path
from typing import Dict


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


MATRIX_ALIASES: Dict[str, Path] = {
    # Canonical smoke-test matrix used so far.
    "SIO": _repo_root() / "examples" / "Sparse_Matrices" / "SiO.mtx",
}


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
    p = Path(key)
    if not p.is_absolute():
        p = _repo_root() / p
    return p.resolve()


