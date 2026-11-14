"""Frost-guard robot task built from the DNAForge freeze-preparation data.

Each trial is one station-night. The robot sees 3-7-day-lagged weather and
location features and decides DEPLOY (frost cover / heater) or HOLD. The true
outcome is whether the daily minimum fell below 0 C (`target`).

Data: challenges/freeze-preparation/09-freeze/data.json.gz (NOAA GHCN-derived,
see that folder's NOTICE.md). Train rows are used for learning, a fixed
stratified validation subset for evaluation.
"""
from __future__ import annotations

import gzip
import json
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = ROOT / "resources" / "challenges" / "freeze-preparation" / "09-freeze" / "data.json.gz"

# Feature order (DATA-DICTIONARY): tmin lag3, lag4, lag7, tmin 7d mean,
# tmax lag3, lag4, lag7, tmax 7d mean, 4 seasonal terms, lat, lon, elevation.
ENCODED = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14]  # 14 features -> 14 sensory electrodes
FEATURE_NAMES = ["tmin_l3", "tmin_l4", "tmin_l7", "tmin_7d", "tmax_l3", "tmax_l4", "tmax_l7",
                 "tmax_7d", "season1", "season2", "season3", "season4", "lat", "elev"]
LEVELS = 5  # 0..4 pulses


_cache: dict = {}


def load():
    if "d" not in _cache:
        d = json.loads(gzip.decompress(DATA.read_bytes()))
        _cache["d"] = d
    return _cache["d"]


class FrostTask:
    def __init__(self, seed: int = 0, n_eval_per_class: int = 150):
        d = load()
        rng = np.random.default_rng(10_000 + seed)
        tr = d["train"]
        self.Xtr = np.array([r["features"] for r in tr], float)[:, ENCODED]
        self.ytr = np.array([r["target"] for r in tr], int)
        va = d["validation"]
        Xva = np.array([r["features"] for r in va], float)[:, ENCODED]
        yva = np.array([r["target"] for r in va], int)
        # fixed stratified eval subset (same for every seed/condition)
        erng = np.random.default_rng(12345)
        pos = erng.choice(np.flatnonzero(yva == 1), n_eval_per_class, replace=False)
        neg = erng.choice(np.flatnonzero(yva == 0), n_eval_per_class, replace=False)
        idx = np.concatenate([pos, neg])
        self.Xev, self.yev = Xva[idx], yva[idx]
        # per-feature quantile edges from *train* for the pulse-count code
        qs = np.linspace(0, 1, LEVELS + 1)[1:-1]
        self.edges = np.quantile(self.Xtr, qs, axis=0).T  # (F, LEVELS-1)
        self.mu, self.sd = self.Xtr.mean(0), self.Xtr.std(0) + 1e-9
        # class-balanced training stream for this seed
        p = np.flatnonzero(self.ytr == 1)
        n = np.flatnonzero(self.ytr == 0)
        self._stream = []
        for _ in range(500):
            self._stream.append(rng.choice(p) if rng.random() < 0.5 else rng.choice(n))

    def train_stream(self, n: int):
        idx = np.array(self._stream[:n])
        return self.Xtr[idx], self.ytr[idx]

    def levels(self, X: np.ndarray) -> np.ndarray:
        """Quantile level 0..4 per feature (pulse count for the culture)."""
        X = np.atleast_2d(X)
        return np.stack([np.searchsorted(self.edges[f], X[:, f]) for f in range(X.shape[1])], 1)

    def z(self, X):
        return (X - self.mu) / self.sd


def auc(score: np.ndarray, label: np.ndarray) -> float:
    score = np.asarray(score, float)
    label = np.asarray(label).astype(bool)
    # average ranks for ties
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    s = score[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    p = label.sum()
    n = len(label) - p
    return float((ranks[label].sum() - p * (p - 1) / 2) / (p * n))
