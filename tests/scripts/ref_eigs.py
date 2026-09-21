from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import re
from typing import List


_EV_RE = re.compile(
    r"""ev\(\s*(\d+)\s*\)\s*=\s*([+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eEdD][+-]?\d+)?)""",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ReferenceSpectrum:
    eigenvalues: List[float]

    def smallest(self, k: int) -> List[float]:
        if k <= 0:
            raise ValueError("k must be > 0")
        return sorted(self.eigenvalues)[:k]

    def largest(self, k: int) -> List[float]:
        if k <= 0:
            raise ValueError("k must be > 0")
        return sorted(self.eigenvalues, reverse=True)[:k]


def parse_reference_spectrum(path: str | Path) -> ReferenceSpectrum:
    """
    Parse a full spectrum in ScaLAPACK/MATLAB indexed form:
      ev( i ) = <value>

    A plain text file with one floating-point value per line is also accepted.
    Consumers sort the full spectrum according to the requested selection.
    """
    p = Path(path)
    txt = p.read_text(encoding="utf-8", errors="replace")

    by_i: dict[int, float] = {}
    for line in txt.splitlines():
        m = _EV_RE.search(line)
        if not m:
            continue
        i = int(m.group(1))
        v_txt = m.group(2).replace("D", "E").replace("d", "e")
        if i in by_i:
            raise ValueError(f"Duplicate eigenvalue index {i} in {p}")
        by_i[i] = float(v_txt)

    if not by_i:
        vals = [float(line.strip().replace("D", "E").replace("d", "e"))
                for line in txt.splitlines() if line.strip()]
        if not vals or not all(math.isfinite(value) for value in vals):
            raise ValueError(f"Empty or nonfinite reference spectrum: {p}")
        return ReferenceSpectrum(eigenvalues=vals)

    # If indices look 1..N contiguous, preserve that; otherwise just return values.
    keys = sorted(by_i.keys())
    if keys and keys[0] == 1 and keys[-1] == len(keys):
        vals = [by_i[i] for i in range(1, len(keys) + 1)]
    else:
        raise ValueError(f"Reference indices must be contiguous, starting at 1: {p}")
    if not all(math.isfinite(value) for value in vals):
        raise ValueError(f"Nonfinite reference eigenvalues: {p}")
    return ReferenceSpectrum(eigenvalues=vals)


parse_scalapack_output = parse_reference_spectrum


def default_ref_path_for_matrix(matrix_path: Path) -> Path:
    """
    Best-effort default reference path:
      <repo_root>/tests/reference/<stem>.out.121mpi
    """
    repo_root = Path(__file__).resolve().parents[2]
    return (repo_root / "tests" / "reference" / f"{matrix_path.stem}.out.121mpi").resolve()

