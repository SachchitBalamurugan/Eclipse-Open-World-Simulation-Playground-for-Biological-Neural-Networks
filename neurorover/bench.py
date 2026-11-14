"""Sample-efficiency benchmark: simulated culture conditions vs ANN baselines.

One job = (condition, seed). It trains on a class-balanced stream of up to
max(BUDGETS) station-nights and evaluates validation AUC at every budget
checkpoint on a fixed stratified subset (150 freeze + 150 no-freeze).

Usage:
  python -m neurorover.bench --condition rstdp --seed 0
  python -m neurorover.bench --all --seeds 10 --procs 8            # local grid
  python -m neurorover.bench --all --seeds 30 --shard 3 --nshards 30  # one Freestyle VM
"""
from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import pathlib
import time

import numpy as np

from .culture.core import N_CH, Culture
from .env.frost import FrostTask, auc
from .interface import codec

BUDGETS = [25, 50, 100, 200]
CULTURE_CONDS = ["culture_frozen", "culture_fep", "culture_rstdp", "culture_reservoir"]
BASELINE_CONDS = ["random", "heuristic_tmin", "logistic", "mlp", "logistic_rawfeat"]
ALL = CULTURE_CONDS + BASELINE_CONDS
RULE = {"culture_frozen": "frozen", "culture_fep": "fep", "culture_rstdp": "rstdp",
        "culture_reservoir": "frozen"}
OUT = pathlib.Path(__file__).resolve().parent / "results"


# ---------------------------------------------------------------- culture trials
def run_trial(c: Culture, levels, rng, label=None, rule="frozen", total_ms=None):
    """Present one observation. Returns (counts[64], score, action, correct)."""
    pulses = codec.encode(levels)
    total_ms = total_ms or (codec.TRIAL_MS if label is not None else codec.EVAL_TRIAL_MS)
    counts = np.zeros(N_CH)
    pi = 0
    fb = []
    fi = 0
    score = action = correct = None
    for t in range(total_ms):
        while pi < len(pulses) and pulses[pi][0] == t:
            c.stim(pulses[pi][1])
            pi += 1
        if fb:
            while fi < len(fb) and fb[fi][0] + codec.FB_START == t:
                c.stim(fb[fi][1], fb[fi][2])
                fi += 1
        fired = c.step()
        if t < codec.RESP_MS and fired.size:
            counts += np.bincount(c.ch[fired], minlength=N_CH)
        if t == codec.RESP_MS - 1:
            score = codec.decode(counts)
            action = int(score > 0) if score != 0 else int(rng.random() < 0.5)
            if label is not None:
                correct = action == label
                fb = codec.feedback(rule, correct, rng)
    return counts, score, action, correct


def eval_culture(c: Culture, task: FrostTask, rng):
    c2 = copy.deepcopy(c)
    c2.rule = "frozen"
    c2.elig = None
    L = task.levels(task.Xev)
    counts = np.zeros((len(L), N_CH))
    scores = np.zeros(len(L))
    for i, lv in enumerate(L):
        counts[i], scores[i], _, _ = run_trial(c2, lv, rng)
    return counts, scores


# ---------------------------------------------------------------- learners
def fit_logistic(X, y, lam=1.0, iters=50):
    X = np.c_[X, np.ones(len(X))]
    w = np.zeros(X.shape[1])
    R = lam * np.eye(X.shape[1])
    R[-1, -1] = 0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X @ w, -30, 30)))
        g = X.T @ (p - y) + R @ w
        H = X.T @ (X * (p * (1 - p))[:, None]) + R + 1e-6 * np.eye(len(w))
        w -= np.linalg.solve(H, g)
    return lambda Z: np.c_[Z, np.ones(len(Z))] @ w


def fit_mlp(X, y, seed, hidden=16, epochs=400, lr=0.02, l2=1e-3):
    rng = np.random.default_rng(seed)
    W1 = rng.normal(0, 1 / np.sqrt(X.shape[1]), (X.shape[1], hidden)); b1 = np.zeros(hidden)
    W2 = rng.normal(0, 1 / np.sqrt(hidden), hidden); b2 = 0.0
    params = [W1, b1, W2, np.array([b2])]
    m = [np.zeros_like(p) for p in params]; v = [np.zeros_like(p) for p in params]
    for t in range(1, epochs + 1):
        W1, b1, W2, b2a = params
        h = np.tanh(X @ W1 + b1)
        p = 1 / (1 + np.exp(-np.clip(h @ W2 + b2a[0], -30, 30)))
        d = (p - y) / len(y)
        gW2 = h.T @ d + l2 * W2; gb2 = np.array([d.sum()])
        dh = np.outer(d, W2) * (1 - h ** 2)
        gW1 = X.T @ dh + l2 * W1; gb1 = dh.sum(0)
        for i, g in enumerate([gW1, gb1, gW2, gb2]):
            m[i] = 0.9 * m[i] + 0.1 * g; v[i] = 0.999 * v[i] + 0.001 * g * g
            params[i] -= lr * (m[i] / (1 - 0.9 ** t)) / (np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
    W1, b1, W2, b2a = params
    return lambda Z: np.tanh(Z @ W1 + b1) @ W2 + b2a[0]


# ---------------------------------------------------------------- jobs
def run_job(condition: str, seed: int) -> dict:
    t0 = time.perf_counter()
    task = FrostTask(seed)
    rng = np.random.default_rng(seed)
    Xs, ys = task.train_stream(max(BUDGETS))
    res = {"condition": condition, "seed": seed, "auc": {}, "train_acc": {}}
    if condition in BASELINE_CONDS:
        for n in BUDGETS:
            X, y = Xs[:n], ys[:n]
            if condition == "random":
                s = rng.random(len(task.yev))
            elif condition == "heuristic_tmin":
                s = -task.Xev[:, 0]
            elif condition == "logistic":
                s = fit_logistic(task.levels(X), y)(task.levels(task.Xev))
            elif condition == "logistic_rawfeat":
                s = fit_logistic(task.z(X), y)(task.z(task.Xev))
            elif condition == "mlp":
                s = fit_mlp(task.levels(X) / 4.0, y, seed)(task.levels(task.Xev) / 4.0)
            res["auc"][n] = auc(s, task.yev)
    else:
        rule = RULE[condition]
        c = Culture(seed=seed, rule="stdp" if rule == "fep" else rule)
        c.run(1000)  # settle
        L = task.levels(Xs)
        tr_counts = np.zeros((len(L), N_CH))
        correct = []
        done = 0
        for n in BUDGETS:
            for i in range(done, n):
                tr_counts[i], _, _, ok = run_trial(c, L[i], rng, label=int(ys[i]), rule=rule)
                correct.append(bool(ok))
            done = n
            ev_counts, ev_scores = eval_culture(c, task, rng)
            if condition == "culture_reservoir":
                mu, sd = tr_counts[:n].mean(0), tr_counts[:n].std(0) + 1e-6
                f = fit_logistic((tr_counts[:n] - mu) / sd, ys[:n], lam=10.0)
                ev_scores = f((ev_counts - mu) / sd)
            res["auc"][n] = auc(ev_scores, task.yev)
            res["train_acc"][n] = float(np.mean(correct[-25:]))
        res["mean_rate_hz"] = float(c.t_ms and 0)  # placeholder overwritten below
        spk = c.run(1000)
        res["mean_rate_hz"] = len(spk) / c.N
    res["wall_s"] = round(time.perf_counter() - t0, 1)
    return res


def _worker(args):
    cond, seed = args
    out = OUT / f"{cond}_s{seed}.json"
    if out.exists():
        return json.loads(out.read_text())
    r = run_job(cond, seed)
    out.write_text(json.dumps(r))
    print(f"{cond} s{seed}: " + " ".join(f"{k}:{v:.3f}" for k, v in r["auc"].items()) + f"  ({r['wall_s']}s)", flush=True)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=ALL)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    conds = ALL if a.all else [a.condition]
    seeds = range(a.seeds) if a.all or a.seeds > 1 else [a.seed]
    jobs = [(c, s) for s in seeds for c in conds]
    jobs = jobs[a.shard::a.nshards]
    if a.procs > 1:
        with mp.Pool(a.procs) as p:
            p.map(_worker, jobs, chunksize=1)
    else:
        for j in jobs:
            _worker(j)


if __name__ == "__main__":
    main()
