"""EnvGenerator: realistic, jointly-sampled night conditions for the frost-guard rover.

Fitted ONLY on train years (2000-2014) of
  * Open-Meteo archive nightly aggregates (cache/nightly.parquet, built by fetch.py)
  * the NOAA station tmin series reconstructed from the freeze-data lag features
    (neurorover.world.model.reconstruct) and the train-only WorldModel (VAR(1) weather model)
  * the KU Leuven DRIP soil-moisture sensor calibration data (train split) for the probe error model

Sampling = analog (nearest-neighbour) resampling of whole real nights:
  1. the night's station minimum temperature comes from WorldModel.rollout (or is given);
  2. a real train-year night of the same station within +-WINDOW days of the day-of-year is drawn
     with a Gaussian kernel on |station_tmin(analog) - tmin| (so clear/calm/dry analogs come with
     cold nights, windy/cloudy/wet analogs with mild ones: the real joint dependence is kept);
  3. temperature-like variables of the analog are shifted by the tmin difference (delta method).
sample_season adds day-to-day persistence: with prob. P_CONTINUE the analog of night t+1 is the
real night following analog t (if it is still a good match), so wet spells, cloudy spells and soil
moisture evolve like real sequences.

Physical effects (see README for sources / assumptions): ground friction, wind force on the cover,
radiative-cooling cue, and noisy sensor readings (SHT45, DS18B20, soil-moisture probe).
"""
from __future__ import annotations

import datetime as dt
import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
NIGHTLY = HERE / "cache" / "nightly.parquet"
SOIL = HERE.parents[1] / "resources" / "challenges" / "soil-moisture-calibration" / "08-soil" / "data.json"
TRAIN_YEARS = (2000, 2014)
WINDOW = 21          # +- days of day-of-year for analog search
BANDWIDTH = 1.5      # C, kernel width on station tmin
P_CONTINUE = 0.8     # analog persistence in sample_season
VARS = ["tmin_night_c", "tdmin_night_c", "rh_mean_night_pct", "wind_max_kmh", "gust_max_kmh", "wind_mean_kmh",
        "cloud_mean_pct", "precip_24h_mm", "soil_moist_06_m3m3", "soil_temp_06_c", "tmax_calday_c"]
SM, CL, PR = (VARS.index(v) for v in ("soil_moist_06_m3m3", "cloud_mean_pct", "precip_24h_mm"))
SHIFT = {"tmin_night_c": 1.0, "tdmin_night_c": 1.0, "soil_temp_06_c": 0.5, "tmax_calday_c": 1.0}

# --------------------------------------------------------------------- physics constants / assumptions
SIGMA = 5.670e-8
SAT_VWC = {"Z": 0.43, "S": 0.44, "P": 0.45, "L": 0.45, "A": 0.46}  # Rawls et al. 1982 total porosity by texture
# Belgian texture codes in DRIP: Z sand, S loamy sand, P light sandy loam, L sandy loam, A loam/silt loam
MU_DRY = {"Z": 0.55, "S": 0.65, "P": 0.70, "L": 0.72, "A": 0.75}     # ASSUMPTION (see README)
MU_WET_DROP = {"Z": 0.05, "S": 0.15, "P": 0.25, "L": 0.30, "A": 0.40}  # ASSUMPTION: loss at saturation
MU_FROZEN_WET = 0.30  # ASSUMPTION: frosted/iced surface
SOIL_TYPE_P = {"Z": 0.15, "S": 0.5, "P": 0.1, "L": 0.15, "A": 0.1}  # DRIP sample frequencies (rounded)


def _doy(date) -> int:
    return date.timetuple().tm_yday


def _as_date(d):
    if isinstance(d, (int, np.integer)):        # day of year -> a representative date (2001 non-leap)
        return dt.date(2001 if d >= 182 else 2002, 1, 1) + dt.timedelta(days=int(d) - 1)
    return dt.date.fromisoformat(d) if isinstance(d, str) else d


class SoilProbeModel:
    """reading = a_soil * true + b_soil + device_offset + noise, fitted on DRIP train split."""

    def fit(self, path=SOIL):
        d = json.loads(pathlib.Path(path).read_text())
        tr = d["train"]
        self.coef, self.sd_dev, self.sd_noise = {}, {}, {}
        allres = []
        for s in sorted({r["soil_type"] for r in tr}):
            R = [r for r in tr if r["soil_type"] == s]
            y = np.array([r["target"] for r in R]); x = np.array([r["sensor_mean"] for r in R])
            a, b = np.polyfit(y, x, 1)
            res = x - (a * y + b)
            dev = np.array([r["device"] for r in R])
            means = {k: res[dev == k].mean() for k in set(dev)}
            within = res - np.array([means[k] for k in dev])
            sd_w = float(within.std(ddof=1))
            sd_b = float(np.sqrt(max(np.var([means[k] for k in means], ddof=1) - sd_w ** 2 / 3, 0)))
            self.coef[s] = (float(a), float(b)); self.sd_dev[s] = sd_b; self.sd_noise[s] = sd_w
            allres.append(res)
        # validation residual sd for honesty
        va = d["validation"]
        r = [r["sensor_mean"] - (self.coef[r["soil_type"]][0] * r["target"] + self.coef[r["soil_type"]][1])
             for r in va if r["soil_type"] in self.coef]
        self.val_resid_sd = float(np.std(r))
        self.train_resid_sd = float(np.concatenate(allres).std())
        return self

    def read(self, true_vwc, soil_type, rng, device_offset=None):
        a, b = self.coef[soil_type]
        off = rng.normal(0, self.sd_dev[soil_type]) if device_offset is None else device_offset
        return float(np.clip(a * true_vwc + b + off + rng.normal(0, self.sd_noise[soil_type]), 0.0, 0.6))

    def summary(self):
        return {s: {"sensor = a*true + b": [round(v, 3) for v in self.coef[s]],
                    "device_offset_sd": round(self.sd_dev[s], 4), "noise_sd": round(self.sd_noise[s], 4)}
                for s in self.coef} | {"train_resid_sd": round(self.train_resid_sd, 4),
                                        "validation_resid_sd": round(self.val_resid_sd, 4)}


class EnvGenerator:
    """EnvGenerator() is lazy: it fits itself (train years only, ~10 s) on first use."""
    _fitted = False

    def _ensure(self):
        if not self._fitted:
            self.fit()

    def fit(self, nightly=None, years=TRAIN_YEARS, use_world_model=True):
        self._fitted = True
        import pandas as pd
        from ..env.frost import load
        from ..world.model import reconstruct
        df = pd.read_parquet(NIGHTLY) if nightly is None else nightly
        df = df.copy()
        dd = pd.to_datetime(df.date)
        df = df[(dd.dt.year >= years[0]) & (dd.dt.year <= years[1])]
        rows = load()["train"]
        S = reconstruct(rows)       # station tmin/tmax series (NOAA scale), train split only
        o = pd.to_datetime(df.date).map(lambda t: t.toordinal()).values
        df["station_tmin_c"] = [S.get(s, {}).get(k, (np.nan,))[0] for s, k in zip(df.station, o)]
        df["ord"] = o
        df = df.dropna(subset=VARS + ["station_tmin_c"])
        self.df = df.reset_index(drop=True)
        self.X = self.df[VARS].to_numpy(float)
        self.st_tmin = self.df.station_tmin_c.to_numpy(float)
        self.ordv = self.df.ord.to_numpy(int)
        self.doy = np.array([dt.date.fromordinal(int(k)).timetuple().tm_yday for k in self.ordv])
        self.idx = {s: np.flatnonzero(self.df.station.values == s) for s in self.df.station.unique()}
        self.by_key = {(s, k): i for i, (s, k) in enumerate(zip(self.df.station.values, self.ordv))}
        meta = {}
        for split in ("train", "validation", "cases"):
            for r in load()[split]:
                meta.setdefault(r["station"], tuple(r["features"][12:15]))
        self.meta = meta
        self.stations = sorted(self.idx)
        self.wm = None
        if use_world_model:
            from ..world.model import get_model
            self.wm = get_model()
        self.soil = SoilProbeModel().fit()
        return self

    # ------------------------------------------------------------------ helpers
    def resolve_station(self, station):
        """Stations without train data (4 cases-only stations) borrow the nearest fitted station."""
        if station in self.idx:
            return station
        lat, lon, el = self.meta[station]
        dist = [(np.hypot(lat - self.meta[s][0], (lon - self.meta[s][1]) * np.cos(np.radians(lat)))
                 + abs(el - self.meta[s][2]) / 300.0, s) for s in self.stations]
        return min(dist)[1]

    def _candidates(self, station, doy):
        ii = self.idx[station]
        dd = np.abs(self.doy[ii] - doy)
        dd = np.minimum(dd, 365 - dd)
        for w in (WINDOW, 2 * WINDOW, 4 * WINDOW, 183):   # widen only if the pool is thin
            c = ii[dd <= w]
            if len(c) >= 30:
                return c
        return ii

    def _draw(self, cand, tmin, rng, prev=None):
        z = ((self.st_tmin[cand] - tmin) / BANDWIDTH) ** 2
        if prev is not None:   # season mode: keep slow state (soil moisture, cloud regime) close to last night
            dsm = self.X[cand, SM] - self.X[prev, SM]
            dsm = np.where((dsm > 0) & (self.X[cand, PR] > 1.0), 0.25 * dsm, dsm)  # rain may wet the soil
            z = z + (dsm / 0.015) ** 2 + ((self.X[cand, CL] - self.X[prev, CL]) / 25.0) ** 2
        w = np.exp(-0.5 * (z - z.min())) + 1e-12
        return int(rng.choice(cand, p=w / w.sum()))

    def _world_tmin(self, station, date, n, rng):
        if self.wm is not None and station in self.wm.beta:
            return self.wm.rollout(station, date, n, rng)[:, 0]
        # fallback: AR(1) on analog-pool climatology
        c = self.st_tmin[self._candidates(station, _doy(date))]
        x, out = rng.normal(), []
        for _ in range(n):
            out.append(c.mean() + c.std() * x); x = 0.6 * x + 0.8 * rng.normal()
        return np.array(out)

    def _night(self, station, date, tmin, i, rng, soil_type, cover_area_m2):
        v = dict(zip(VARS, self.X[i]))
        delta = tmin - self.st_tmin[i]
        for k, f in SHIFT.items():
            v[k] = v[k] + f * delta
        v["tdmin_night_c"] = min(v["tdmin_night_c"], v["tmin_night_c"])
        n = {"station": station, "date": date.isoformat(), "doy": _doy(date),
             "air_tmin_c": float(tmin),                 # station-scale (NOAA) minimum, the frost ground truth
             "frost": bool(tmin < 0.0),
             "analog": {"station": self.df.station.values[i],
                        "date": dt.date.fromordinal(int(self.ordv[i])).isoformat()}}
        n.update({("reanalysis_" + k if k in ("tmin_night_c", "tmax_calday_c") else k): float(val)
                  for k, val in v.items()})
        n["soil_type"] = soil_type
        n.update(physics(n, soil_type, cover_area_m2, self.meta.get(station, (0, 0, 0))[2]))
        return n

    # ------------------------------------------------------------------ public API
    def sample_night(self, station, date, rng, tmin=None, soil_type="S", cover_area_m2=2.0):
        """One night. date: ISO str / datetime.date / int day-of-year. tmin: optional station tmin (C)."""
        self._ensure()
        date = _as_date(date)
        if station is None:
            station = self.stations[int(rng.integers(len(self.stations)))]
        st = self.resolve_station(station)
        if tmin is None:
            tmin = float(self._world_tmin(st, date, 1, rng)[0])
        i = self._draw(self._candidates(st, _doy(date)), tmin, rng)
        n = self._night(station, date, tmin, i, rng, soil_type, cover_area_m2)
        n["sensors"] = sensors(n, rng, self.soil)
        return n

    def sample_season(self, station, start_date, n_nights, rng, soil_type="S", cover_area_m2=2.0, tmins=None):
        """n consecutive nights with persistence (world-model tmin + analog continuation)."""
        self._ensure()
        d0 = _as_date(start_date)
        if station is None:
            station = self.stations[int(rng.integers(len(self.stations)))]
        st = self.resolve_station(station)
        tm = self._world_tmin(st, d0, n_nights, rng) if tmins is None else np.asarray(tmins, float)
        out, prev = [], None
        dev = {"sht45_bias": rng.normal(0, 0.05), "ds18b20_bias": rng.uniform(-0.4, 0.4),
               "soil_offset": rng.normal(0, self.soil.sd_dev.get(soil_type, 0.02))}
        for t in range(n_nights):
            date = d0 + dt.timedelta(days=t)
            i = None
            if prev is not None and rng.random() < P_CONTINUE:
                j = self.by_key.get((self.df.station.values[prev], int(self.ordv[prev]) + 1))
                if j is not None and abs(self.st_tmin[j] - tm[t]) < 2 * BANDWIDTH:
                    i = j
            if i is None:
                i = self._draw(self._candidates(st, _doy(date)), tm[t], rng, prev)
            n = self._night(station, date, float(tm[t]), i, rng, soil_type, cover_area_m2)
            n["sensors"] = sensors(n, rng, self.soil, dev)
            out.append(n)
            prev = i
        return out


    def night(self, station, date, rng=None, **kw):
        """sim/night_loop.py hook: sample_night + the keys FrostRoverEnv.dr overrides expect.
        wind_ms  = max gust at cover height (m/s); friction = ground_friction_mu;
        noise_level = sensor-noise multiplier (ASSUMPTION: 1.0 dry/calm, up to ~1.6 in rain + wind)."""
        if rng is None:
            self._rng = getattr(self, "_rng", None) or np.random.default_rng(0)
            rng = self._rng
        n = self.sample_night(station, date, rng, **kw)
        n["wind_ms"] = n["gust_at_cover_ms"]
        n["friction"] = n["ground_friction_mu"]
        n["noise_level"] = round(1.0 + 0.3 * min(n["precip_24h_mm"] / 5.0, 1.0)
                                 + 0.3 * min(n["gust_at_cover_ms"] / 10.0, 1.0), 3)
        return n


# --------------------------------------------------------------------- physics + sensors
def physics(n, soil_type="S", cover_area_m2=2.0, elev_m=0.0, panel_area_m2=0.12, z_cover_m=0.5):
    """Physical effects for the robot sim. All formulas documented in README."""
    tk = n["air_tmin_c"] + 273.15
    # ground friction (ASSUMPTION, see README): falls with relative saturation, lower when frozen-wet
    sat = float(np.clip(n["soil_moist_06_m3m3"] / SAT_VWC.get(soil_type, 0.45), 0, 1))
    mu = MU_DRY.get(soil_type, 0.7) - MU_WET_DROP.get(soil_type, 0.25) * sat ** 2
    if n["soil_temp_06_c"] < 0 and sat > 0.5:
        mu = min(mu, MU_FROZEN_WET)
    # wind at cover height: log profile from 10 m, z0 = 0.03 m (open flat farmland)
    z0 = 0.03
    f = np.log(z_cover_m / z0) / np.log(10 / z0)
    rho = 101325 * np.exp(-elev_m / 8434) / (287.05 * tk)
    cd = 1.2  # flat sheet normal to flow (Hoerner 1965), conservative for a cloth on hoops
    u_mean = n["wind_mean_kmh"] / 3.6 * f
    u_gust = n["gust_max_kmh"] / 3.6 * f
    q = lambda u: 0.5 * rho * cd * u ** 2
    # radiative cooling cue: net longwave loss of a black surface at air temperature
    tdk = n["tdmin_night_c"] + 273.15
    eps_clear = 0.787 + 0.764 * np.log(tdk / 273.0)                 # Berdahl & Martin 1984
    N = n["cloud_mean_pct"] / 10.0                                 # tenths
    eps = min(eps_clear * (1 + 0.0224 * N - 0.0035 * N ** 2 + 0.00028 * N ** 3), 1.0)
    lnet = SIGMA * tk ** 4 * (1 - eps)
    # leaf energy balance (Campbell & Norman 1998): one side sees the sky, two sides exchange heat;
    # boundary-layer conductance 0.135*sqrt(u/d) mol/m2/s, d = 5 cm leaf, cp = 29.3 J/mol/K
    hc = 2 * 29.3 * 0.135 * np.sqrt(max(u_mean, 0.1) / 0.05)
    dleaf = -lnet / (2 * 4 * SIGMA * tk ** 3 + hc)
    return {"ground_friction_mu": round(float(mu), 3),
            "soil_relative_saturation": round(sat, 3),
            "wind_mean_at_cover_ms": round(float(u_mean), 2), "gust_at_cover_ms": round(float(u_gust), 2),
            "air_density_kgm3": round(float(rho), 4),
            "cover_force_mean_N": round(float(q(u_mean) * cover_area_m2), 2),
            "cover_force_gust_N": round(float(q(u_gust) * cover_area_m2), 2),
            "panel_force_gust_N": round(float(q(u_gust) * panel_area_m2), 3),
            "sky_emissivity": round(float(eps), 3),
            "net_longwave_loss_wm2": round(float(lnet), 1),
            "leaf_minus_air_c": round(float(dleaf), 2),
            "leaf_tmin_c": round(float(n["air_tmin_c"] + dleaf), 2)}


def sensors(n, rng, soil_model, dev=None):
    """Noisy readings at the coldest hour. dev: per-device persistent biases (from sample_season)."""
    dev = dev or {"sht45_bias": rng.normal(0, 0.05), "ds18b20_bias": rng.uniform(-0.4, 0.4), "soil_offset": None}
    air = n["air_tmin_c"] + dev["sht45_bias"] + rng.normal(0, 0.1)            # SHT45 +-0.1 C typ
    # RH at the coldest hour from dew point; SHT45 +-1.0 %RH typ
    a, b = 17.625, 243.04
    # (computed within the reanalysis: its tmin and dew point are mutually consistent; station tmin is not)
    t_r = n["reanalysis_tmin_night_c"]
    rh_true = 100 * np.exp(a * n["tdmin_night_c"] / (b + n["tdmin_night_c"]) - a * t_r / (b + t_r))
    rh = np.clip(min(rh_true, 100) + rng.normal(0, 1.0), 0, 100)
    leaf = n["leaf_tmin_c"] + dev["ds18b20_bias"] + rng.normal(0, 0.1)        # DS18B20 +-0.5 C max
    leaf = np.round(leaf / 0.0625) * 0.0625                                   # 12-bit resolution
    soil = soil_model.read(n["soil_moist_06_m3m3"], n.get("soil_type", "S"), rng, dev.get("soil_offset"))
    return {"sht45_air_c": round(float(air), 3), "sht45_rh_pct": round(float(rh), 2),
            "ds18b20_leaf_c": float(leaf), "soil_probe_vwc": round(soil, 4)}


_g = {}


def get_generator() -> EnvGenerator:
    if "g" not in _g:
        _g["g"] = EnvGenerator().fit()
    return _g["g"]


if __name__ == "__main__":   # python -m neurorover.envdata.generator  -> one demo night + a 3-night season
    g = get_generator()
    r = np.random.default_rng(0)
    print(json.dumps(g.sample_night("USW00093193", "2016-01-10", r), indent=1))
    for n in g.sample_season("USW00093193", "2016-12-01", 3, r):
        print(n["date"], n["air_tmin_c"], n["cloud_mean_pct"], n["soil_moist_06_m3m3"], n["ground_friction_mu"])
