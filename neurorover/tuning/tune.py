"""Multiprocessing inner-split hyperparameter grid for plastic cultures.

Defaults use the requested R-STDP grid (36 settings, 3 seeds). FEP has 9
gain/noise settings because its plain-STDP branch ignores Culture.lr. The
worker calls bench.run_job with a training-only split; results stay local.
"""
from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import pathlib
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from neurorover import bench
from neurorover.tuning.inner_split import InnerFrostTask

OUT = pathlib.Path(__file__).resolve().parent / "results"


def _worker(args):
    condition, seed, kw, eval_n = args
    bench.FrostTask = lambda s: InnerFrostTask(s, n_eval_per_class=eval_n)
    bench.BUDGETS = [200]
    t0 = time.perf_counter()
    result = bench.run_job(condition, seed, kw, tag="inner200")
    result["eval_split"] = "stratified train-only rows excluding full 500-row stream"
    result["eval_per_class"] = eval_n
    result["wall_s"] = round(time.perf_counter() - t0, 1)
    key = f"{condition}_lr{kw['lr']:g}_gain{kw['stim_gain']:g}_noise{kw['noise']:g}_s{seed}"
    path = OUT / f"{key}.json"
    path.write_text(json.dumps(result, indent=2))
    print(f"{key}: auc200={result['auc'][200]:.3f} ({result['wall_s']}s)", flush=True)
    return result


def _result_path(condition, seed, kw):
    key = f"{condition}_lr{kw['lr']:g}_gain{kw['stim_gain']:g}_noise{kw['noise']:g}_s{seed}"
    return OUT / f"{key}.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--procs", type=int, default=3)
    ap.add_argument("--eval-per-class", type=int, default=75)
    ap.add_argument("--lrs", default="0.02,0.2,1,5")
    ap.add_argument("--stim-gains", default="15,30,60")
    ap.add_argument("--noises", default="3.5,4,4.5")
    ap.add_argument("--conditions", default="culture_rstdp,culture_fep")
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--max-jobs", type=int, default=0,
                    help="run at most this many missing jobs, then exit (0 = all)")
    args = ap.parse_args()
    if not 1 <= args.procs <= 3:
        ap.error("--procs must be between 1 and 3")
    OUT.mkdir(parents=True, exist_ok=True)
    lrs = [float(x) for x in args.lrs.split(",")]
    gains = [float(x) for x in args.stim_gains.split(",")]
    noises = [float(x) for x in args.noises.split(",")]
    conditions = args.conditions.split(",")
    seeds = range(args.seed_start, args.seed_start + args.seeds)
    jobs = [(condition, seed,
             {"lr": lr, "stim_gain": gain, "noise": noise},
             args.eval_per_class)
            for condition in conditions
            for lr, gain, noise in itertools.product(
                (lrs if condition == "culture_rstdp" else lrs[:1]), gains, noises)
            for seed in seeds]
    if "culture_fep" in conditions:
        print("FEP uses plain STDP; Culture.lr does not affect it, so FEP runs once per gain/noise pair.", flush=True)
    missing = [job for job in jobs if not _result_path(job[0], job[1], job[2]).exists()]
    if args.max_jobs:
        missing = missing[:args.max_jobs]
    print(f"Running {len(missing)} missing jobs with {args.procs} processes", flush=True)
    if args.procs == 1:
        for job in missing:
            _worker(job)
    elif missing:
        with mp.get_context("spawn").Pool(args.procs) as pool:
            for _ in pool.imap_unordered(_worker, missing, chunksize=1):
                pass

    import numpy as np
    aggregates = {}
    saved_results = []
    for result_path in OUT.glob("culture_*.json"):
        try:
            r = json.loads(result_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if r.get("condition") in conditions and r.get("tag") == "inner200":
            saved_results.append(r)
    for r in saved_results:
        k = (r["condition"], r["culture_kw"]["lr"],
             r["culture_kw"]["stim_gain"], r["culture_kw"]["noise"])
        auc200 = r["auc"].get("200", r["auc"].get(200))
        aggregates.setdefault(k, []).append(float(auc200))
    summary = []
    for (condition, lr, gain, noise), scores in aggregates.items():
        summary.append({"condition": condition, "lr": lr, "stim_gain": gain,
                        "noise": noise, "auc200_mean": float(np.mean(scores)),
                        "auc200_sd": float(np.std(scores)), "n_seeds": len(scores)})
    summary.sort(key=lambda row: row["auc200_mean"], reverse=True)
    (OUT / "tune_summary.json").write_text(json.dumps(summary, indent=2))
    for condition in conditions:
        candidates = [row for row in summary if row["condition"] == condition]
        if candidates:
            print(f"BEST {condition}: {candidates[0]}", flush=True)


if __name__ == "__main__":
    main()
