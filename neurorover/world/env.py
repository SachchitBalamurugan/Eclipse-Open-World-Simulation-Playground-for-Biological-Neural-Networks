"""FrostWorld: a gym-style season of consecutive station-nights.

obs    : the 14 encoded features for tonight as quantile levels 0..4 (same code the
         culture gets in the benchmark; edges come from the train split)
action : 1 = DEPLOY frost protection, 0 = HOLD
reward : -1 for deploying (energy/labour), -10 crop damage if HOLD on a freeze night, else 0
info   : truth (did it freeze), running crop health, raw features, date

source="real"     replays validation-split nights of that station in date order
                  (the file is a subsample of station-days, so "consecutive" means
                  consecutive available nights; gaps of a few days occur)
source="imagined" rolls the season out of the learned WorldModel

Smoke test:  python -m neurorover.world.env
"""
from __future__ import annotations

import datetime as dt

import numpy as np

from ..env.frost import ENCODED, LEVELS, load
from .model import get_model

COST_DEPLOY, COST_DAMAGE = -1.0, -10.0
_cache: dict = {}


def _edges():
    if "e" not in _cache:
        Xtr = np.array([r["features"] for r in load()["train"]], float)[:, ENCODED]
        qs = np.linspace(0, 1, LEVELS + 1)[1:-1]
        _cache["e"] = np.quantile(Xtr, qs, axis=0).T  # identical to FrostTask.edges
    return _cache["e"]


def _val_by_station():
    if "v" not in _cache:
        by: dict = {}
        for r in load()["validation"]:
            by.setdefault(r["station"], []).append(r)
        for s in by:
            by[s].sort(key=lambda r: r["date"])
        _cache["v"] = by
    return _cache["v"]


def encode(features) -> np.ndarray:
    x = np.asarray(features, float)[ENCODED]
    e = _edges()
    return np.array([np.searchsorted(e[f], x[f]) for f in range(len(x))])


class FrostWorld:
    def __init__(self, n_nights: int = 60):
        self.n_nights = n_nights

    def stations(self):
        return sorted(_val_by_station())

    def reset(self, seed: int = 0, station: str | None = None, source: str = "real", start_date: str | None = None):
        rng = np.random.default_rng(seed)
        station = station or self.stations()[int(rng.integers(len(self.stations())))]
        if start_date is None:  # a winter inside the validation years
            start_date = f"{2015 + int(rng.integers(4))}-12-01"
        self.station, self.source, self.start_date = station, source, start_date
        if source == "real":
            rows = [r for r in _val_by_station()[station] if r["date"] >= start_date][: self.n_nights]
            self.F = np.array([r["features"] for r in rows], float)
            self.y = np.array([r["target"] for r in rows], int)
            self.dates = [r["date"] for r in rows]
        elif source == "imagined":
            self.F, self.y, _ = get_model().season(station, start_date, self.n_nights, rng)
            o = dt.date.fromisoformat(start_date).toordinal()
            self.dates = [dt.date.fromordinal(o + i).isoformat() for i in range(self.n_nights)]
        else:
            raise ValueError(source)
        if len(self.y) == 0:
            raise ValueError(f"no nights for {station} from {start_date}")
        self.t, self.health, self.total = 0, 1.0, 0.0
        return self._obs()

    def _obs(self):
        return encode(self.F[self.t]) if self.t < len(self.y) else np.zeros(len(ENCODED), int)

    @property
    def raw(self):
        """Raw 15 features of tonight (for hand-written policies; not given to the culture)."""
        return self.F[min(self.t, len(self.y) - 1)]

    def step(self, action: int):
        truth = int(self.y[self.t])
        reward = COST_DEPLOY if action == 1 else (COST_DAMAGE if truth else 0.0)
        if action == 0 and truth:
            self.health = max(0.0, self.health - 0.1)
        self.total += reward
        info = {"truth": truth, "crop_health": self.health, "date": self.dates[self.t],
                "features": self.F[self.t].tolist(), "season_reward": self.total}
        self.t += 1
        done = self.t >= len(self.y)
        return self._obs(), reward, done, info


# ---------------------------------------------------------------- reference policies
def fit_tmin_threshold() -> float:
    """Deploy iff tmin_lag3 < thr. thr maximises mean reward on the TRAIN split."""
    if "thr" not in _cache:
        d = load()["train"]
        x = np.array([r["features"][0] for r in d])
        y = np.array([r["target"] for r in d])
        grid = np.arange(-10, 15.01, 0.5)
        rew = [np.where(x < g, COST_DEPLOY, COST_DAMAGE * y).mean() for g in grid]
        _cache["thr"] = float(grid[int(np.argmax(rew))])
    return _cache["thr"]


def run_season(env: FrostWorld, policy, **reset_kw) -> dict:
    """policy(obs, env) -> action. Returns season totals."""
    obs = env.reset(**reset_kw)
    done, n_dep, n_frz, n_dmg = False, 0, 0, 0
    while not done:
        a = int(policy(obs, env))
        obs, r, done, info = env.step(a)
        n_dep += a
        n_frz += info["truth"]
        n_dmg += int(a == 0 and info["truth"])
    return {"reward": env.total, "nights": len(env.y), "freezes": n_frz, "deploys": n_dep,
            "damaged_nights": n_dmg, "crop_health": info["crop_health"]}


if __name__ == "__main__":
    env = FrostWorld()
    thr = fit_tmin_threshold()
    prng = np.random.default_rng(0)
    pol = {"random": lambda o, e: prng.integers(2), "always_hold": lambda o, e: 0, "always_deploy": lambda o, e: 1,
           f"tmin_l3<{thr}": lambda o, e: e.raw[0] < thr}
    st = env.stations()[0]
    print("obs example:", env.reset(seed=0, station=st), "station", st)
    for src in ("real", "imagined"):
        for name, p in pol.items():
            print(f"{src:9s} {name:14s}", run_season(env, p, seed=0, station=st, source=src, start_date="2016-12-01"))
