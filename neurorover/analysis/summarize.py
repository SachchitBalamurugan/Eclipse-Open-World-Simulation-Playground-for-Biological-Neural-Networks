"""Summarize benchmark results and paired seed-level comparisons.

Run from the repository root with:
    .venv/Scripts/python -m neurorover.analysis.summarize

Every invocation scans neurorover/results anew. JSON files that are temporarily
incomplete while a sweep is writing them are ignored and picked up next run.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import re
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "neurorover" / "results"
OUTPUT = Path(__file__).resolve().parent / "summary.md"
BUDGETS = (25, 50, 100, 200)
CONDITIONS = (
    "culture_frozen",
    "culture_fep",
    "culture_rstdp",
    "culture_teach",
    "culture_reservoir",
    "random",
    "heuristic_tmin",
    "logistic",
    "mlp",
    "logistic_rawfeat",
)
N_BOOT = 20_000


def _auc_values(record: dict) -> dict[int, float]:
    """Accept the current nested format and older flattened auc25 formats."""
    values: dict[int, float] = {}
    nested = record.get("auc", {})
    if isinstance(nested, dict):
        for key, value in nested.items():
            match = re.search(r"\d+", str(key))
            if match:
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(number):
                    values[int(match.group())] = number
    for key, value in record.items():
        match = re.fullmatch(r"auc[_-]?(\d+)", str(key), flags=re.IGNORECASE)
        if match:
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                values[int(match.group(1))] = number
    return values


def load_results() -> tuple[dict[str, dict[int, dict[int, float]]], int]:
    """Read a fresh snapshot of all valid result files from the results folder."""
    data: dict[str, dict[int, dict[int, float]]] = {}
    skipped = 0
    if not RESULTS.exists():
        return data, skipped
    for path in sorted(RESULTS.glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            condition = record.get("condition")
            seed = int(record["seed"])
            aucs = _auc_values(record)
            if not condition or not aucs or record.get("tag"):  # tagged = hyperparameter variants, not headline
                skipped += 1
                continue
            # A file is the unit of work. If duplicates exist, the later
            # filename in sorted order deterministically supplies that seed.
            data.setdefault(str(condition), {})[seed] = aucs
        except (OSError, ValueError, TypeError, KeyError, AttributeError, json.JSONDecodeError):
            skipped += 1
    return data, skipped


def _rng(tag: str) -> np.random.Generator:
    # Stable seed without Python's process-randomized hash().
    seed = 0x5EED
    for byte in tag.encode("utf-8"):
        seed = (seed * 131 + byte) % (2**32)
    return np.random.default_rng(seed)


def bootstrap_mean_ci(values, rng: np.random.Generator, n_boot: int = N_BOOT):
    x = np.asarray(values, dtype=float)
    if len(x) < 2:
        return None
    draws = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[draws].mean(axis=1)
    return tuple(float(v) for v in np.quantile(means, [0.025, 0.975]))


def summarize_cell(values: list[float], tag: str) -> str:
    if not values:
        return "—"
    mean = float(np.mean(values))
    ci = bootstrap_mean_ci(values, _rng(tag))
    if ci is None:
        return f"{mean:.3f} [CI unavailable]; n={len(values)}"
    return f"{mean:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]; n={len(values)}"


def _paired_values(data, left: str, right: str, budget: int):
    left_seeds = data.get(left, {})
    right_seeds = data.get(right, {})
    seeds = sorted(
        seed for seed in left_seeds.keys() & right_seeds.keys()
        if budget in left_seeds[seed] and budget in right_seeds[seed]
    )
    differences = np.array(
        [left_seeds[seed][budget] - right_seeds[seed][budget] for seed in seeds],
        dtype=float,
    )
    return seeds, differences


def paired_test_line(data, left: str, right: str, budget: int) -> str:
    seeds, differences = _paired_values(data, left, right, budget)
    heading = f"- **{left} − {right}, {budget} samples:** "
    if len(differences) == 0:
        return heading + "no matched seed results; no comparison is possible."
    delta = float(differences.mean())
    ci = bootstrap_mean_ci(
        differences, _rng(f"paired:{left}:{right}:{budget}")
    )
    if ci is None:
        return (
            heading
            + f"mean paired Δ={delta:+.3f}, n=1; insufficient pairs for inference."
        )
    significant = ci[0] > 0.0 or ci[1] < 0.0
    status = "significant" if significant else "not significant"
    return (
        heading
            + f"mean paired Δ={delta:+.4f} (95% bootstrap CI "
            + f"[{ci[0]:+.4f}, {ci[1]:+.4f}], n={len(differences)}): "
        + f"**{status}** at the 5% level "
        + ("(CI excludes zero)." if significant else "(CI includes zero).")
    )


def build_report(data, skipped: int) -> str:
    found = set(data)
    conditions = [c for c in CONDITIONS if c in found]
    conditions += sorted(found - set(conditions))
    total = sum(len(seeds) for seeds in data.values())
    lines = [
        "# Benchmark summary",
        "",
        f"Loaded {total} condition/seed result records from neurorover/results/ "
        f"({sum(len(row) for row in data.values())} files with valid AUC data).",
        "This script scans the folder again each time it runs. Confidence intervals "
        "are percentile bootstrap intervals over seeds (20,000 resamples). "
        "Intervals are unavailable for a single seed.",
    ]
    if skipped:
        lines.append(
            f"Skipped {skipped} unreadable, incomplete, or unsupported result file(s); "
            "they will be reconsidered on the next run."
        )
    lines.extend(["", "## AUC by condition and sample budget", ""])
    lines.append("| Condition | " + " | ".join(map(str, BUDGETS)) + " |")
    lines.append("|---|" + "|".join("---:" for _ in BUDGETS) + "|")
    for condition in conditions:
        cells = []
        for budget in BUDGETS:
            vals = [
                per_seed[budget]
                for per_seed in data.get(condition, {}).values()
                if budget in per_seed
            ]
            cells.append(summarize_cell(vals, f"cell:{condition}:{budget}"))
        lines.append("| " + condition + " | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "Each cell gives mean AUC [95% bootstrap CI]; n=number of seeds.",
            "",
            "## H1: plastic culture conditions vs frozen culture",
            "",
            "Paired comparisons use the intersection of seeds at each budget. "
            "The reported interval bootstraps the per-seed AUC difference "
            "(left condition minus right condition); a difference is called "
            "significant when its two-sided 95% interval excludes zero.",
            "",
        ]
    )
    for condition in ("culture_fep", "culture_rstdp", "culture_teach"):
        for budget in BUDGETS:
            lines.append(
                paired_test_line(data, condition, "culture_frozen", budget)
            )
    lines.extend(
        [
            "",
            "## H2: reservoir culture vs logistic and MLP",
            "",
            "The same paired bootstrap rule is used at each sample budget.",
            "Significance calls are unadjusted across the tested budgets and contrasts.",
            "",
        ]
    )
    for baseline in ("logistic", "mlp"):
        for budget in BUDGETS:
            lines.append(
                paired_test_line(data, "culture_reservoir", baseline, budget)
            )
    lines.extend(
        [
            "",
            "These seed counts are small and the intervals describe uncertainty "
            "across the observed seeds; treat the significance calls as "
            "seed-level, exploratory comparisons.",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    # PowerShell may expose a legacy code page that cannot print Markdown's
    # Unicode symbols; the saved report remains UTF-8.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    data, skipped = load_results()
    report = build_report(data, skipped)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nWrote {OUTPUT}")


if __name__ == "__main__":
    main()
