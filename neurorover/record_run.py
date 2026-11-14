"""Run one simulated culture benchmark job and record trial-level signals."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from neurorover import bench
from neurorover.culture.core import Culture, MOTOR_DEPLOY, MOTOR_HOLD, N_CH, REWARD, SENSORY
from neurorover.env.frost import auc
from neurorover.interface import codec


class SignalWriter:
    def __init__(self, out: pathlib.Path):
        out.mkdir(parents=True, exist_ok=True)
        self._spikes_f = (out / "spikes.csv").open("w", newline="", encoding="utf-8")
        self._stims_f = (out / "stims.csv").open("w", newline="", encoding="utf-8")
        self._rates_f = (out / "electrode_rates.csv").open("w", newline="", encoding="utf-8")
        self.spikes = csv.writer(self._spikes_f)
        self.stims = csv.writer(self._stims_f)
        self.rates = csv.writer(self._rates_f)
        self.spikes.writerow(["trial", "phase", "t_ms", "channel"])
        self.stims.writerow(["trial", "phase", "t_ms", "channel", "kind"])
        self.rates.writerow(["trial", "phase", "window"] + [f"ch{i}" for i in range(N_CH)])
        self.active = False
        self.n_spikes = 0
        self.n_stims = 0

    def attach(self, culture: Culture):
        self.culture = culture
        self.original_step = culture.step
        self.original_stim = culture.stim

        def logged_step():
            if not self.active:
                return self.original_step()
            t = int(culture.t_ms - self.start_ms)
            fired = self.original_step()
            if fired.size:
                channels = culture.ch[fired].astype(int)
                for ch in channels:
                    self.spikes.writerow([self.trial, self.phase, t, int(ch)])
                self.n_spikes += int(fired.size)
                self.trial_spikes += int(fired.size)
                counts = np.bincount(channels, minlength=N_CH)
                if t < codec.STIM_MS:
                    self.window_counts[0] += counts
                if t < codec.RESP_MS:
                    self.window_counts[1] += counts
                if codec.FB_START <= t < codec.FB_START + codec.FB_MS:
                    self.window_counts[2] += counts
                if t >= codec.FB_START + codec.FB_MS:
                    self.window_counts[3] += counts
            return fired

        def logged_stim(channel: int, current_uA: float = 1.0):
            if self.active:
                t = int(culture.t_ms - self.start_ms)
                ch = int(channel)
                kind = "reward" if ch in REWARD else ("sensory" if t < codec.STIM_MS else "feedback")
                self.stims.writerow([self.trial, self.phase, t, ch, kind])
                self.n_stims += 1
            return self.original_stim(channel, current_uA)

        culture.step = logged_step
        culture.stim = logged_stim

    def detach(self):
        self.culture.step = self.original_step
        self.culture.stim = self.original_stim

    def begin(self, trial: int, phase: str):
        self.trial, self.phase = int(trial), phase
        self.start_ms = self.culture.t_ms
        self.window_counts = np.zeros((4, N_CH), dtype=np.int64)
        self.trial_spikes = 0
        self.active = True

    def end(self):
        self.active = False
        for name, counts in zip(("stim", "response", "feedback", "rest"), self.window_counts):
            self.rates.writerow([self.trial, self.phase, name] + counts.tolist())
        return self.trial_spikes

    def close(self):
        self._spikes_f.close()
        self._stims_f.close()
        self._rates_f.close()


def run(condition: str, seed: int, budget: int, eval_n: int, out_root: pathlib.Path):
    t0 = time.perf_counter()
    if not 1 <= budget <= max(bench.BUDGETS):
        raise ValueError(f"budget must be between 1 and {max(bench.BUDGETS)}")
    if not 2 <= eval_n <= 300:
        raise ValueError("eval must be between 2 and 300")

    out = out_root / f"{condition}_s{seed}"
    out.mkdir(parents=True, exist_ok=True)
    task = bench.FrostTask(seed)
    rng = np.random.default_rng(seed)
    Xs, ys = task.train_stream(max(bench.BUDGETS))
    train_levels = task.levels(Xs[:budget])
    rule = bench.RULE[condition]
    c = Culture(seed=seed, rule="stdp" if rule == "fep" else rule)
    start_W = c.W.copy() if rule in ("fep", "rstdp", "teach") else None
    c.run(1000)  # Same initial settling period as bench.run_job.

    signal = SignalWriter(out)
    trials_path = out / "trials.csv"
    plastic = rule in ("fep", "rstdp", "teach")
    trials_f = trials_path.open("w", newline="", encoding="utf-8")
    trials = csv.writer(trials_f)
    trials.writerow(["trial", "phase", "station_label", "action", "correct", "score",
                     "deploy_spikes", "hold_spikes", "total_spikes"])

    weights_f = weights = None
    if plastic:
        weights_f = (out / "weights.csv").open("w", newline="", encoding="utf-8")
        weights = csv.writer(weights_f)
        weights.writerow(["trial", "mean_W_sens_to_deploy", "mean_W_sens_to_hold",
                          "mean_abs_dW_since_start"])

    tr_counts = np.zeros((budget, N_CH))
    n_pos = eval_n // 2
    n_neg = eval_n - n_pos
    eval_idx = np.concatenate((np.arange(n_pos), 150 + np.arange(n_neg)))
    eval_levels = task.levels(task.Xev[eval_idx])
    eval_labels = task.yev[eval_idx]
    checkpoints = sorted({n for n in bench.BUDGETS if n < budget} | {budget})
    # Measure weight change from initialization, including the benchmark's settle period.
    dmask = c.plastic & np.isin(c.ch[None, :], MOTOR_DEPLOY)
    hmask = c.plastic & np.isin(c.ch[None, :], MOTOR_HOLD)
    pmask = c.plastic
    signal.attach(c)
    try:
        for i in range(budget):
            signal.begin(i, "train")
            counts, score, action, correct = bench.run_trial(
                c, train_levels[i], rng, label=int(ys[i]), rule=rule)
            total_spikes = signal.end()
            tr_counts[i] = counts
            deploy = int(counts[MOTOR_DEPLOY].sum())
            hold = int(counts[MOTOR_HOLD].sum())
            trials.writerow([i, "train", int(ys[i]), action, int(bool(correct)), score,
                             deploy, hold, total_spikes])
            if plastic:
                weights.writerow([i, float(c.W[dmask].mean()), float(c.W[hmask].mean()),
                                  float(np.abs(c.W[pmask] - start_W[pmask]).mean())])
            if i + 1 in checkpoints:
                # Match bench.run_job's checkpoint evaluations on a frozen copy.
                signal.detach()
                c2 = copy.deepcopy(c)
                c2.rule = "frozen"
                c2.elig = None
                signal.attach(c2)
                eval_counts = np.zeros((eval_n, N_CH))
                raw_scores = np.zeros(eval_n)
                for j, lv in enumerate(eval_levels):
                    signal.begin(j, "eval")
                    counts, score, action, _ = bench.run_trial(c2, lv, rng)
                    total_spikes = signal.end()
                    eval_counts[j] = counts
                    raw_scores[j] = score
                    deploy = int(counts[MOTOR_DEPLOY].sum())
                    hold = int(counts[MOTOR_HOLD].sum())
                    trials.writerow([j, "eval", int(eval_labels[j]), action,
                                     int(action == int(eval_labels[j])), score,
                                     deploy, hold, total_spikes])
                signal.detach()
                signal.attach(c)
                if condition == "culture_reservoir":
                    mu = tr_counts[:i + 1].mean(0)
                    sd = tr_counts[:i + 1].std(0) + 1e-6
                    fitted = bench.fit_logistic((tr_counts[:i + 1] - mu) / sd,
                                                ys[:i + 1], lam=10.0)
                    auc_scores = fitted((eval_counts - mu) / sd)
                else:
                    auc_scores = raw_scores
                final_auc = auc(auc_scores, eval_labels)
    finally:
        signal.detach()
        signal.close()
        trials_f.close()
        if weights_f:
            weights_f.close()

    wall_s = round(time.perf_counter() - t0, 1)
    meta = {"condition": condition, "seed": seed, "budget": budget, "eval": eval_n,
            "final_auc": float(final_auc), "n_spikes": signal.n_spikes,
            "n_stims": signal.n_stims, "wall_s": wall_s}
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return out, meta


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--condition", required=True, choices=bench.CULTURE_CONDS)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--budget", type=int, default=50)
    ap.add_argument("--eval", type=int, default=60)
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("runs"))
    args = ap.parse_args()
    out, meta = run(args.condition, args.seed, args.budget, args.eval, args.out)
    print(f"{out}: AUC={meta['final_auc']:.6f}, spikes={meta['n_spikes']}, stims={meta['n_stims']}, wall={meta['wall_s']}s")


if __name__ == "__main__":
    main()
