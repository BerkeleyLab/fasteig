from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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


def parse_scalapack_output(path: str | Path) -> ReferenceSpectrum:
    """
    Parse a ScaLAPACK output file containing lines like:
      ev( i ) = <value>

    Returns the full spectrum as a list of floats. Ordering in the file may
    vary; consumers should sort as needed.
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
        by_i[i] = float(v_txt)

    if not by_i:
        raise ValueError(f"No 'ev(i) =' lines found in reference file: {p}")

    # If indices look 1..N contiguous, preserve that; otherwise just return values.
    keys = sorted(by_i.keys())
    if keys and keys[0] == 1 and keys[-1] == len(keys):
        vals = [by_i[i] for i in range(1, len(keys) + 1)]
    else:
        vals = [by_i[i] for i in keys]
    return ReferenceSpectrum(eigenvalues=vals)


def default_ref_path_for_matrix(matrix_path: Path) -> Path:
    """
    Best-effort default reference path:
      <repo_root>/tests/reference/<stem>.out.121mpi
    """
    repo_root = Path(__file__).resolve().parents[2]
    return (repo_root / "tests" / "reference" / f"{matrix_path.stem}.out.121mpi").resolve()

