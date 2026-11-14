"""Learning-rate sensitivity table for the plastic culture conditions.

Reads tagged result files (e.g. culture_rstdp_lr1_s3.json) plus the untagged
default-lr runs (lr=0.02), and writes analysis/sensitivity.md. Every setting is
reported; nothing is selected on the evaluation set.
"""
from __future__ import annotations

import collections
import json
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = pathlib.Path(__file__).resolve().parent / "sensitivity.md"
BUDGETS = ["25", "50", "100", "200"]


def main():
    cells = collections.defaultdict(lambda: collections.defaultdict(list))
    for f in RES.glob("culture_*.json"):
        r = json.loads(f.read_text())
        if r["condition"] not in ("culture_rstdp", "culture_fep", "culture_frozen"):
            continue
        lr = (r.get("culture_kw") or {}).get("lr", 0.02)
        if r["condition"] == "culture_frozen":
            if r.get("tag"):
                continue
            lr = "-"
        elif r.get("tag") and "lr" not in (r.get("culture_kw") or {}):
            continue
        if not r.get("tag") and r["condition"] != "culture_frozen" and r["seed"] >= 10:
            # default-lr runs: compare on the same 10 seeds as the tagged runs
            continue
        if r["condition"] == "culture_frozen" and r["seed"] >= 10:
            continue
        for b in BUDGETS:
            if b in r["auc"]:
                cells[(r["condition"], str(lr))][b].append(r["auc"][b])
    lines = ["# Learning-rate sensitivity (plastic cultures, seeds 0-9)", "",
             "Validation AUC, mean ± s.e.m. over seeds. All settings shown; none was picked using the evaluation set.", "",
             "| condition | lr | " + " | ".join(BUDGETS) + " |", "|---|---|" + "---:|" * len(BUDGETS)]
    for (cond, lr) in sorted(cells, key=lambda k: (k[0], float(k[1]) if k[1] != "-" else -1)):
        row = []
        for b in BUDGETS:
            v = np.array(cells[(cond, lr)][b])
            row.append(f"{v.mean():.3f} ± {v.std(ddof=1) / np.sqrt(len(v)):.3f} (n={len(v)})" if len(v) > 1 else "-")
        lines.append(f"| {cond} | {lr} | " + " | ".join(row) + " |")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
