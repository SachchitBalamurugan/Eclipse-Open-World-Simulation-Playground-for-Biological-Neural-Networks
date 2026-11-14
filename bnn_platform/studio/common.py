"""Shared paths and JSON output helpers for the Studio backend (see docs/STUDIO_CONTRACT.md)."""
from __future__ import annotations

import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
USER_ROBOTS = ROOT / "user_robots"
USER_RUNS = ROOT / "user_runs"
NAME_RE = re.compile(r"^[a-z0-9_]{1,40}$")

# The real stdout. Studio commands redirect sys.stdout to stderr so that stray prints from
# libraries can never corrupt the JSON stream; everything machine-readable goes through emit().
_OUT = sys.stdout


def set_stdout(f) -> None:
    global _OUT
    _OUT = f


def emit(obj: dict, log_file=None) -> None:
    line = json.dumps(obj, separators=(",", ":"), allow_nan=False, default=_default)
    _OUT.write(line + "\n")
    _OUT.flush()
    if log_file is not None:
        log_file.write(line + "\n")
        log_file.flush()


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _default(o):
    try:
        import numpy as np
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:  # pragma: no cover
        pass
    if isinstance(o, pathlib.Path):
        return o.as_posix()
    raise TypeError(f"not JSON serialisable: {type(o)}")


def rel(p) -> str:
    p = pathlib.Path(p).resolve()
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def r6(x, nd=5):
    """Round a float / sequence for compact JSON."""
    if hasattr(x, "__len__"):
        return [round(float(v), nd) for v in x]
    return round(float(x), nd)


class StudioError(Exception):
    """Human-readable failure; reported as {"ok":false,"error":...} or {"type":"error",...}."""
