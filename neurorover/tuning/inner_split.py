"""Training-only inner evaluation split for tuning plastic cultures.

The validation split is never read here. We construct the same class-balanced
stream as FrostTask and draw evaluation rows only from train rows absent from
the stream.
"""
from __future__ import annotations

import numpy as np

from ..env import frost


class InnerFrostTask:
    """FrostTask-compatible object with a stratified, training-only eval set."""

    def __init__(self, seed: int = 0, n_eval_per_class: int = 75):
        d = frost.load()
        tr = d["train"]
        self.Xtr = np.array([r["features"] for r in tr], float)[:, frost.ENCODED]
        self.ytr = np.array([r["target"] for r in tr], int)

        # Match FrostTask's stream exactly so held-out rows cannot appear in
        # any training example in its 500-row stream.
        rng = np.random.default_rng(10_000 + seed)
        pos = np.flatnonzero(self.ytr == 1)
        neg = np.flatnonzero(self.ytr == 0)
        self._stream = [int(rng.choice(pos if rng.random() < 0.5 else neg))
                        for _ in range(500)]

        used = np.zeros(len(self.ytr), dtype=bool)
        used[np.asarray(self._stream, dtype=int)] = True
        available_pos = np.flatnonzero((self.ytr == 1) & ~used)
        available_neg = np.flatnonzero((self.ytr == 0) & ~used)
        if min(len(available_pos), len(available_neg)) < n_eval_per_class:
            raise ValueError("not enough unused train rows for stratified inner eval")
        erng = np.random.default_rng(12_345 + seed)
        idx = np.concatenate([
            erng.choice(available_pos, n_eval_per_class, replace=False),
            erng.choice(available_neg, n_eval_per_class, replace=False),
        ])
        self.Xev, self.yev = self.Xtr[idx], self.ytr[idx]

        fit_mask = np.ones(len(self.ytr), dtype=bool)
        fit_mask[idx] = False
        Xfit = self.Xtr[fit_mask]
        qs = np.linspace(0, 1, frost.LEVELS + 1)[1:-1]
        self.edges = np.quantile(Xfit, qs, axis=0).T
        self.mu, self.sd = Xfit.mean(0), Xfit.std(0) + 1e-9

    def train_stream(self, n: int):
        idx = np.asarray(self._stream[:n], dtype=int)
        return self.Xtr[idx], self.ytr[idx]

    def levels(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        return np.stack([np.searchsorted(self.edges[f], X[:, f])
                         for f in range(X.shape[1])], 1)

    def z(self, X):
        return (X - self.mu) / self.sd

