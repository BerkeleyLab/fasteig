from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import List, Optional


@dataclass(frozen=True)
class FinalResults:
    eigenvalues: List[float]
    residual_norms: List[float]
    runtime_sec: Optional[float] = None
    iterations: Optional[int] = None


_SECTION_RE = re.compile(r"^\s*FINAL RESULTS\s*$", re.IGNORECASE)
_RAW_SECTION_RE = re.compile(r"^\s*RAW RESULTS\b", re.IGNORECASE)
_TABLE_HEADER_RE = re.compile(r"^\s*(Index|Idx|Eigenpair)\b", re.IGNORECASE)
_ROW_RE = re.compile(
    r"^\s*(\d+)\s+([+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?)\s+([+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?)\s*$"
)
_TOL_RE = re.compile(r"^\s*Tolerance\s*:\s*(\S+)\s*$", re.IGNORECASE)
_RUNTIME_RE = re.compile(
    r"^\s*Runtime\s*\(sec\)\s*:\s*([+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?)\s*$",
    re.IGNORECASE,
)
_ITER_RE = re.compile(r"^\s*Iteration\s+(\d+)\s*\|")
_DIAG_ITERS_RE = re.compile(r"^\s*Iterations\s*:\s*(\d+)\s*$", re.IGNORECASE)


def parse_tolerance(text: str) -> Optional[float]:
    for line in text.splitlines():
        m = _TOL_RE.match(line)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                return None
    return None


def parse_final_results(path: str | Path) -> FinalResults:
    p = Path(path)
    txt = p.read_text(encoding="utf-8", errors="replace")

    lines = txt.splitlines()
    runtime_sec: Optional[float] = None
    iterations: Optional[int] = None
    for line in lines:
        m = _DIAG_ITERS_RE.match(line)
        if m:
            try:
                iterations = int(m.group(1))
            except ValueError:
                iterations = None
        m = _RUNTIME_RE.match(line)
        if m:
            try:
                runtime_sec = float(m.group(1))
            except ValueError:
                runtime_sec = None
        m = _ITER_RE.match(line)
        if m:
            try:
                it = int(m.group(1))
            except ValueError:
                continue
            # Keep this as a fallback only; explicit DIAGNOSTICS value wins.
            if iterations is None:
                iterations = it
            else:
                iterations = max(iterations, it)

    # Prefer RAW RESULTS (FULL PRECISION) if present.
    raw_start = None
    for i, line in enumerate(lines):
        if _RAW_SECTION_RE.match(line):
            raw_start = i
            break
    if raw_start is not None:
        header_i = None
        for i in range(raw_start, len(lines)):
            if lines[i].strip().lower().startswith("index "):
                header_i = i
                break
        if header_i is not None:
            eigenvalues: List[float] = []
            residuals: List[float] = []
            for i in range(header_i + 1, len(lines)):
                line = lines[i].strip()
                if not line:
                    break
                if line.upper().startswith("DIAGNOSTICS"):
                    break
                m = _ROW_RE.match(lines[i])
                if not m:
                    if eigenvalues:
                        break
                    continue
                eigenvalues.append(float(m.group(2)))
                residuals.append(float(m.group(3)))
            if eigenvalues and len(eigenvalues) == len(residuals):
                return FinalResults(
                    eigenvalues=eigenvalues,
                    residual_norms=residuals,
                    runtime_sec=runtime_sec,
                    iterations=iterations,
                )

    start = None
    for i, line in enumerate(lines):
        if _SECTION_RE.match(line):
            start = i
            break
    if start is None:
        raise ValueError(f"FINAL RESULTS section not found in log: {p}")

    # Find table header after the section.
    header_i = None
    for i in range(start, len(lines)):
        if _TABLE_HEADER_RE.match(lines[i]):
            header_i = i
            break
    if header_i is None:
        raise ValueError(f"FINAL RESULTS table header not found in log: {p}")

    eigenvalues: List[float] = []
    residuals: List[float] = []

    for i in range(header_i + 1, len(lines)):
        line = lines[i].strip()
        if not line:
            # blank line ends the table in our logs
            break
        if line.upper().startswith("DIAGNOSTICS"):
            break
        m = _ROW_RE.match(lines[i])
        if not m:
            # tolerate non-data lines, but stop if we already started reading rows and hit junk
            if eigenvalues:
                break
            continue
        eigenvalues.append(float(m.group(2)))
        residuals.append(float(m.group(3)))

    if not eigenvalues:
        raise ValueError(f"No FINAL RESULTS rows parsed from log: {p}")
    if len(eigenvalues) != len(residuals):
        raise ValueError(f"Mismatched eigenvalues/residual rows in log: {p}")

    return FinalResults(
        eigenvalues=eigenvalues,
        residual_norms=residuals,
        runtime_sec=runtime_sec,
        iterations=iterations,
    )

