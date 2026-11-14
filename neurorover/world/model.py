"""Learned world model of station weather, fitted ONLY on the train split.

The data file has no raw daily series, only lagged features per station-day.
We reconstruct partial per-station daily tmin/tmax series from them (lag3 of
date D is the observation of D-3, lag4 of D-4, lag7 of D-7), then fit

    x_t = clim_s(doy_t) + a_t,      a_t = A a_{t-1} + e_t,   e_t ~ N(0, Sigma_s)

clim_s : per-station seasonal mean, 2 harmonics of day-of-year (least squares)
A      : pooled 2x2 VAR(1) on (tmin, tmax) anomalies
Sigma_s: per-station residual covariance

Usage:  python -m neurorover.world.model      (fits, evaluates, writes results JSON)
"""
from __future__ import annotations

import datetime as dt
import json
import pathlib

import numpy as np

from ..env.frost import load

OUT = pathlib.Path(__file__).resolve().parent / "results"
YEAR = 365.25
BURN = 9  # days of history needed before a target night (lags 3..9)


def _ord(date) -> int:
    return (dt.date.fromisoformat(date) if isinstance(date, str) else date).toordinal()


def _doy(ordinals) -> np.ndarray:
    o = np.atleast_1d(np.asarray(ordinals, int))
    return np.array([dt.date.fromordinal(int(k)).timetuple().tm_yday for k in o], float)


def _harm(doy) -> np.ndarray:
    w = 2 * np.pi * np.asarray(doy, float) / YEAR
    return np.stack([np.ones_like(w), np.sin(w), np.cos(w), np.sin(2 * w), np.cos(2 * w)], -1)


def reconstruct(rows) -> dict:
    """rows -> {station: {ordinal_day: (tmin, tmax)}} from the lag3/4/7 features."""
    S: dict = {}
    for r in rows:
        d0, f = _ord(r["date"]), r["features"]
        s = S.setdefault(r["station"], {})
        for lag, i in ((3, 0), (4, 1), (7, 2)):
            s[d0 - lag] = (f[i], f[i + 4])
    return S


def _pairs(series: dict, k: int = 1):
    """(day_t, x_t, x_{t+k}) for all days where both are known."""
    days = np.array(sorted(series))
    X = np.array([series[d] for d in days], float)
    pos = {d: i for i, d in enumerate(days)}
    i0 = [i for i, d in enumerate(days) if d + k in pos]
    i1 = [pos[days[i] + k] for i in i0]
    return days[i0], X[i0], X[i1]


class WorldModel:
    def fit(self, train_rows):
        S = reconstruct(train_rows)
        self.stations = sorted(S)
        self.meta = {}
        cnt = {}
        for r in train_rows:
            self.meta.setdefault(r["station"], r["features"][12:15])
            cnt[r["station"]] = cnt.get(r["station"], 0) + 1
        self.p_station = np.array([cnt[s] for s in self.stations], float)
        self.p_station /= self.p_station.sum()
        # the file only covers Oct-Apr: imagined nights reuse the train (station, date) distribution
        self.keys = [(r["station"], _ord(r["date"])) for r in train_rows]
        self.beta = {}
        P, Q, per = [], [], {}
        for s in self.stations:
            days = np.array(sorted(S[s]))
            X = np.array([S[s][d] for d in days], float)
            H = _harm(_doy(days))
            self.beta[s] = np.linalg.lstsq(H, X, rcond=None)[0]  # (5, 2)
            d0, x0, x1 = _pairs(S[s])
            a0 = x0 - _harm(_doy(d0)) @ self.beta[s]
            a1 = x1 - _harm(_doy(d0 + 1)) @ self.beta[s]
            per[s] = (a0, a1)
            P.append(a0)
            Q.append(a1)
        P, Q = np.concatenate(P), np.concatenate(Q)
        self.A = np.linalg.lstsq(P, Q, rcond=None)[0].T  # a1 = A a0
        self.Sigma, self.chol, self.chol0 = {}, {}, {}
        for s in self.stations:
            a0, a1 = per[s]
            e = a1 - a0 @ self.A.T
            self.Sigma[s] = np.cov(e.T)
            self.chol[s] = np.linalg.cholesky(self.Sigma[s])
            V = self.Sigma[s].copy()  # stationary covariance: V = A V A' + Sigma
            for _ in range(200):
                V = self.A @ V @ self.A.T + self.Sigma[s]
            self.chol0[s] = np.linalg.cholesky(V)
        self.n_train_days = int(sum(len(S[s]) for s in self.stations))
        return self

    # ------------------------------------------------------------ generation
    def clim(self, station, ordinals) -> np.ndarray:
        return _harm(_doy(ordinals)) @ self.beta[station]

    def rollout(self, station, start_date, n_days, rng, a0=None) -> np.ndarray:
        """(n_days, 2) imagined daily [tmin, tmax] starting at start_date."""
        o = _ord(start_date) + np.arange(n_days)
        a = self.chol0[station] @ rng.standard_normal(2) if a0 is None else np.asarray(a0, float)
        out = np.empty((n_days, 2))
        mu = self.clim(station, o)
        for t in range(n_days):
            if t:
                a = self.A @ a + self.chol[station] @ rng.standard_normal(2)
            out[t] = mu[t] + a
        return out

    def features(self, station, target_ord: int, hist: np.ndarray) -> list:
        """15-feature row for target day D; hist = [tmin,tmax] of days D-9..D-3 (7 rows)."""
        w = 2 * np.pi * _doy(target_ord)[0] / YEAR
        f = [hist[6, 0], hist[5, 0], hist[2, 0], hist[:, 0].mean(),
             hist[6, 1], hist[5, 1], hist[2, 1], hist[:, 1].mean(),
             np.sin(w), np.cos(w), np.sin(2 * w), np.cos(2 * w)]
        return [float(v) for v in f] + list(self.meta[station])

    def season(self, station, start_date, n_nights, rng):
        """Imagined consecutive nights -> (features (n,15), target (n,), tmin (n,))."""
        o0 = _ord(start_date)
        x = self.rollout(station, dt.date.fromordinal(o0 - BURN), n_nights + BURN, rng)
        F = [self.features(station, o0 + i, x[i:i + 7]) for i in range(n_nights)]
        tmin = x[BURN:, 0]
        return np.array(F), (tmin < 0).astype(int), tmin

    def imagine_rows(self, n, rng, balanced=False):
        """n synthetic station-nights in the data-file row format.

        (station, date) is drawn from the empirical train distribution (the file
        only has Oct-Apr station-days); the weather itself is imagined."""
        want = {1: n // 2, 0: n - n // 2} if balanced else None
        rows = []
        while len(rows) < n:
            s, d = self.keys[int(rng.integers(len(self.keys)))]
            x = self.rollout(s, dt.date.fromordinal(d - BURN), BURN + 1, rng)
            y = int(x[BURN, 0] < 0)
            if want is not None:
                if want[y] == 0:
                    continue
                want[y] -= 1
            rows.append({"id": f"imag{len(rows)}", "station": s, "date": dt.date.fromordinal(d).isoformat(),
                         "features": self.features(s, d, x[:7]), "target": y,
                         "imagined_tmin_c": float(x[BURN, 0])})
        order = rng.permutation(n)
        return [rows[i] for i in order]

    # ------------------------------------------------------------ evaluation
    def evaluate(self, val_rows, steps=(1, 3)) -> dict:
        S = reconstruct(val_rows)
        res = {}

        def rmse(e):
            return [round(float(v), 3) for v in np.sqrt((np.concatenate(e) ** 2).mean(0))]

        for k in steps:
            Ak = np.linalg.matrix_power(self.A, k)
            em, ep, ec = [], [], []
            for s in S:
                if s not in self.beta:
                    continue
                d0, x0, x1 = _pairs(S[s], k)
                c0, c1 = self.clim(s, d0), self.clim(s, d0 + k)
                em.append(c1 + (x0 - c0) @ Ak.T - x1)
                ep.append(x0 - x1)
                ec.append(c1 - x1)
            res[f"{k}_step"] = {"n_pairs": int(sum(len(e) for e in em)),
                                "rmse_model_tmin_tmax": rmse(em), "rmse_persistence_tmin_tmax": rmse(ep),
                                "rmse_climatology_tmin_tmax": rmse(ec)}
        return res


_wm = {}


def get_model() -> WorldModel:
    if "m" not in _wm:
        _wm["m"] = WorldModel().fit(load()["train"])
    return _wm["m"]


def main():
    d = load()
    wm = get_model()
    res = {"fit": {"split": "train only", "n_stations": len(wm.stations),
                   "n_reconstructed_station_days": wm.n_train_days,
                   "A": np.round(wm.A, 4).tolist(),
                   "mean_resid_sd_tmin_tmax": np.round(
                       np.mean([np.sqrt(np.diag(v)) for v in wm.Sigma.values()], 0), 3).tolist()},
           "validation": wm.evaluate(d["validation"])}
    rng = np.random.default_rng(0)
    im = wm.imagine_rows(20000, rng)
    res["freeze_base_rate"] = {
        "real_train": round(float(np.mean([r["target"] for r in d["train"]])), 4),
        "real_validation": round(float(np.mean([r["target"] for r in d["validation"]])), 4),
        "imagined_20000_nights": round(float(np.mean([r["target"] for r in im])), 4)}
    Fi = np.array([r["features"] for r in im])
    Ft = np.array([r["features"] for r in d["train"]])
    res["feature_marginals_first8"] = {
        "imagined_mean": np.round(Fi[:, :8].mean(0), 2).tolist(), "train_mean": np.round(Ft[:, :8].mean(0), 2).tolist(),
        "imagined_sd": np.round(Fi[:, :8].std(0), 2).tolist(), "train_sd": np.round(Ft[:, :8].std(0), 2).tolist()}
    OUT.mkdir(exist_ok=True)
    (OUT / "world_model_eval.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
