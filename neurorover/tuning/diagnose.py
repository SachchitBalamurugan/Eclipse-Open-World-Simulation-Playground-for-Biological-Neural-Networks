"""Instrument 200 training trials for the plastic culture conditions."""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from neurorover import bench
from neurorover.culture.core import Culture
from neurorover.tuning.inner_split import InnerFrostTask

OUT = pathlib.Path(__file__).resolve().parent / "results"


def diagnose(condition: str, seed: int = 0, culture_kw: dict | None = None):
    rule = bench.RULE[condition]
    if rule not in ("fep", "rstdp"):
        raise ValueError("diagnose supports culture_fep and culture_rstdp")
    task = InnerFrostTask(seed)
    X, y = task.train_stream(200)
    levels = task.levels(X)
    rng = np.random.default_rng(seed)
    c = Culture(seed=seed, rule="stdp" if rule == "fep" else rule,
                **(culture_kw or {}))
    c.run(1000)
    initial_w = c.W.copy()
    plastic = c.plastic
    if not plastic.any():
        raise RuntimeError("culture has no plastic synapses")

    records = []
    current = {"requested_dw": 0.0, "applied_dw": 0.0,
               "elig_mean_abs": None, "elig_mean": None,
               "requested_dw_signed": None, "reward": None}
    original_reward = c.reward

    def measured_reward(r):
        if c.elig is not None:
            current["elig_mean_abs"] = float(np.mean(np.abs(c.elig[plastic])))
            current["elig_mean"] = float(np.mean(c.elig[plastic]))
            current["requested_dw"] = float(np.mean(np.abs(c.lr * r * c.elig[plastic])))
            current["requested_dw_signed"] = float(np.mean(c.lr * r * c.elig[plastic]))
            before = c.W[plastic].copy()
            current["reward"] = float(r)
            original_reward(r)
            current["applied_dw"] = float(np.mean(np.abs(c.W[plastic] - before)))
        else:
            original_reward(r)
    c.reward = measured_reward

    for i, (lv, label) in enumerate(zip(levels, y)):
        current.update(requested_dw=0.0, applied_dw=0.0,
                       elig_mean_abs=None, elig_mean=None,
                       requested_dw_signed=None, reward=None)
        spike_count = [0]
        original_step = c.step

        def counted_step():
            fired = original_step()
            spike_count[0] += fired.size
            return fired
        c.step = counted_step

        w_before = c.W.copy()
        _, score, action, correct = bench.run_trial(
            c, lv, rng, label=int(label), rule=rule)
        c.step = original_step
        local_delta = c.W[plastic] - w_before[plastic]
        firing_hz = spike_count[0] / c.N / (400 / 1000)
        records.append({
            "trial": i + 1,
            "label": int(label),
            "deploy_minus_hold": float(score),
            "action": int(action),
            "correct": bool(correct),
            "mean_abs_dw_plastic_this_trial": float(np.mean(np.abs(local_delta))),
            "sum_abs_dw_plastic_this_trial": float(np.sum(np.abs(local_delta))),
            "reward_time_mean_abs_eligibility": current["elig_mean_abs"],
            "reward_time_mean_eligibility_signed": current["elig_mean"],
            "reward_time_requested_mean_abs_dw": current["requested_dw"],
            "reward_time_requested_mean_dw_signed": current["requested_dw_signed"],
            "reward_time_applied_mean_abs_dw": current["applied_dw"],
            "reward": current["reward"],
            "firing_rate_hz_per_neuron": float(firing_hz),
            "cumulative_mean_abs_drift_plastic": float(
                np.mean(np.abs(c.W[plastic] - initial_w[plastic]))),
            "cumulative_mean_abs_drift_all": float(np.mean(np.abs(c.W - initial_w))),
            "cumulative_abs_drift_sum_plastic": float(
                np.sum(np.abs(c.W[plastic] - initial_w[plastic]))),
            "cumulative_abs_drift_sum_all": float(np.sum(np.abs(c.W - initial_w))),
        })

    summary = {"condition": condition, "seed": seed,
               "culture_kw": culture_kw or {}, "trials": len(records)}
    for label in (0, 1):
        subset = [r for r in records if r["label"] == label]
        summary[f"label_{label}"] = {
            "n": len(subset),
            "mean_deploy_minus_hold": float(np.mean([r["deploy_minus_hold"] for r in subset])),
            "median_deploy_minus_hold": float(np.median([r["deploy_minus_hold"] for r in subset])),
            "accuracy": float(np.mean([r["correct"] for r in subset])),
        }
    summary.update({
        "mean_abs_dw_plastic_per_trial": float(np.mean([r["mean_abs_dw_plastic_this_trial"] for r in records])),
        "mean_reward_time_abs_eligibility": (float(np.mean([
            r["reward_time_mean_abs_eligibility"] for r in records
            if r["reward_time_mean_abs_eligibility"] is not None]))
            if rule == "rstdp" else None),
        "mean_reward_time_signed_eligibility": (float(np.mean([
            r["reward_time_mean_eligibility_signed"] for r in records
            if r["reward_time_mean_eligibility_signed"] is not None]))
            if rule == "rstdp" else None),
        "mean_reward_requested_abs_dw": (float(np.mean([r["reward_time_requested_mean_abs_dw"] for r in records]))
                                          if rule == "rstdp" else None),
        "mean_reward_applied_abs_dw": (float(np.mean([r["reward_time_applied_mean_abs_dw"] for r in records]))
                                        if rule == "rstdp" else None),
        "final_mean_abs_drift_plastic": records[-1]["cumulative_mean_abs_drift_plastic"],
        "final_mean_abs_drift_all": records[-1]["cumulative_mean_abs_drift_all"],
        "final_sum_abs_drift_plastic": records[-1]["cumulative_abs_drift_sum_plastic"],
        "final_sum_abs_drift_all": records[-1]["cumulative_abs_drift_sum_all"],
        "plastic_weight_stats": {
            "initial_mean": float(np.mean(initial_w[plastic])),
            "final_mean": float(np.mean(c.W[plastic])),
            "final_min": float(np.min(c.W[plastic])),
            "final_max": float(np.max(c.W[plastic])),
            "fraction_at_zero": float(np.mean(c.W[plastic] <= 1e-8)),
            "fraction_at_w_max": float(np.mean(c.W[plastic] >= c.w_max - 1e-8)),
        },
        "firing_rate_hz_per_neuron": {
            "mean": float(np.mean([r["firing_rate_hz_per_neuron"] for r in records])),
            "p05": float(np.quantile([r["firing_rate_hz_per_neuron"] for r in records], .05)),
            "p95": float(np.quantile([r["firing_rate_hz_per_neuron"] for r in records], .95)),
            "silent_trials": sum(r["firing_rate_hz_per_neuron"] == 0 for r in records),
        },
        "per_trial": records,
    })
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--culture-kw", default="{}")
    args = ap.parse_args()
    kw = json.loads(args.culture_kw)
    OUT.mkdir(parents=True, exist_ok=True)
    for condition in ("culture_rstdp", "culture_fep"):
        result = diagnose(condition, args.seed, kw)
        path = OUT / f"diagnose_{condition}_s{args.seed}.json"
        path.write_text(json.dumps(result, indent=2))
        brief = {k: v for k, v in result.items() if k != "per_trial"}
        print(json.dumps(brief, indent=2))


if __name__ == "__main__":
    main()

