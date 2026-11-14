"""Fetch + cache + nightly aggregation of Open-Meteo historical weather for the
35 NOAA stations of the freeze-preparation data, joined to the freeze rows.

    python -m neurorover.envdata.fetch fetch [--max-minutes 40]   # resumable download
    python -m neurorover.envdata.fetch aggregate                  # cache -> nightly.parquet + join report

Requests: one per station x frost season (Sep 22 .. Apr 30), 8 hourly variables
(relative humidity is derived from T and dew point with the Magnus formula, which
saves 1/9 of the API cost). Raw responses are cached gzipped in cache/raw/
(gitignored); the compact nightly table cache/nightly.parquet is small enough to commit.

Open-Meteo cost = n_vars/10 * n_days/14 "calls" per request; free tier is
600/min, 5000/hour, 10000/day. A full 26-season x 35-station pull costs ~11.6k,
so seasons are fetched in priority order (validation years, then train years
newest-first, then the 2020-2025 cases years) and the script stops cleanly on
the hourly/daily limit; just re-run it later to fill the rest.

Weather data by Open-Meteo.com (CC BY 4.0), ERA5/ERA5-Land reanalysis via the archive API.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import pathlib
import sys
import time
import http.client
import urllib.error
import urllib.request

import numpy as np

from ..env.frost import load

HERE = pathlib.Path(__file__).resolve().parent
RAW = HERE / "cache" / "raw"
NIGHTLY = HERE / "cache" / "nightly.parquet"
RESULTS = HERE / "results"
API = "https://archive-api.open-meteo.com/v1/archive"
HOURLY = ["temperature_2m", "dew_point_2m", "wind_speed_10m", "wind_gusts_10m", "cloud_cover",
          "precipitation", "soil_moisture_0_to_7cm", "soil_temperature_0_to_7cm"]
SEASON_START = (9, 22)   # covers lag-9 inputs of Oct 1 targets
SEASON_END = (4, 30)


def stations() -> dict:
    """station -> (lat, lon, elev_m) from the freeze rows (public NOAA coordinates)."""
    d = load()
    out = {}
    for split in ("train", "validation", "cases"):
        for r in d[split]:
            if r["station"] not in out:
                f = r["features"]
                out[r["station"]] = (float(f[12]), float(f[13]), float(f[14]))
    return dict(sorted(out.items()))


def season_of(date: dt.date) -> int:
    """Season label = year in which the season starts (Oct 2014 .. Apr 2015 -> 2014)."""
    return date.year if date.month >= 9 else date.year - 1


def needed_jobs() -> list:
    """(priority, station, season) for every station-season that has freeze rows."""
    d = load()
    have = set()
    for split in ("train", "validation", "cases"):
        for r in d[split]:
            have.add((r["station"], season_of(dt.date.fromisoformat(r["date"]))))

    def prio(season):
        if 2014 <= season <= 2019:        # validation years 2015-2019 (season 2014 holds Jan-Apr 2015)
            return (0, -season)
        if season <= 2013:                 # train years, newest first
            return (1, -season)
        return (2, season)                 # cases years 2020-2025
    return sorted((prio(se), st, se) for st, se in have)


def raw_path(station, season) -> pathlib.Path:
    return RAW / f"{station}_{season}.json.gz"


def fetch_one(lat, lon, start, end, tries=6):
    q = (f"{API}?latitude={lat:.4f}&longitude={lon:.4f}&start_date={start}&end_date={end}"
         f"&hourly={','.join(HOURLY)}&timezone=auto")
    wait = 5.0
    for k in range(tries):
        try:
            req = urllib.request.Request(q, headers={"User-Agent": "envdata-research-client/0.1"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = resp.read()
                cost = resp.headers.get("X-RateLimit-Cost")
                return body, cost
        except urllib.error.HTTPError as e:
            msg = e.read().decode("utf8", "replace")
            if e.code == 429:
                if "Daily" in msg or "daily" in msg:
                    raise RuntimeError("daily limit: " + msg)
                if "Hourly" in msg or "hourly" in msg:
                    raise RuntimeError("hourly limit: " + msg)
                time.sleep(65)            # minutely limit
                continue
            if e.code >= 500:
                time.sleep(wait); wait *= 2
                continue
            raise
        except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException, OSError):
            time.sleep(wait); wait *= 2
    raise RuntimeError("giving up on " + q)


def cmd_fetch(max_minutes: float, train_oldest_first: bool = False):
    RAW.mkdir(parents=True, exist_ok=True)
    S = stations()
    jobs = needed_jobs()
    if train_oldest_first:  # 2nd worker: train seasons oldest-first, meets worker 1 in the middle
        jobs = sorted((j for j in jobs if j[0][0] == 1), key=lambda j: (j[2], j[1]))
    t0 = time.time()
    done = sum(raw_path(st, se).exists() for _, st, se in jobs)
    print(f"{len(jobs)} station-seasons needed, {done} cached", flush=True)
    spent = 0.0
    for i, (_, st, se) in enumerate(jobs):
        p = raw_path(st, se)
        if p.exists() or p.with_suffix(".part").exists():
            continue
        if (time.time() - t0) / 60 > max_minutes:
            print("time budget reached", flush=True)
            break
        lat, lon, _ = S[st]
        start = dt.date(se, *SEASON_START)
        end = min(dt.date(se + 1, *SEASON_END), dt.date.today() - dt.timedelta(days=7))
        lock = p.with_suffix(".part")
        lock.write_bytes(b"")
        try:
            body, cost = fetch_one(lat, lon, start, end)
        except RuntimeError as e:
            lock.unlink(missing_ok=True)
            print("STOP:", e, flush=True)
            break
        p.write_bytes(gzip.compress(body))
        lock.unlink(missing_ok=True)
        spent += float(cost or 0)
        done += 1
        print(f"[{done}/{len(jobs)}] {st} {se} cost={cost} total={spent:.0f}", flush=True)
        time.sleep(1.0)
    print(f"cached {done}/{len(jobs)} station-seasons; spent ~{spent:.0f} calls this run", flush=True)


# --------------------------------------------------------------------- aggregation
def rh_from_dew(t, td):
    """Magnus formula (Alduchov & Eskridge 1996 coefficients), %."""
    a, b = 17.625, 243.04
    return 100 * np.exp(a * td / (b + td) - a * t / (b + t))


def nightly_from_raw(station, js) -> "pd.DataFrame":
    import pandas as pd
    h = js["hourly"]
    df = pd.DataFrame({k: np.asarray(h[k], dtype=float) for k in HOURLY},
                      index=pd.to_datetime(h["time"]))
    df["rh"] = rh_from_dew(df["temperature_2m"].values, df["dew_point_2m"].values)
    # night of target date D = 18:00 local on D-1 .. 09:00 local on D (inclusive)
    key = (df.index + pd.Timedelta(hours=6)).normalize()     # 18:00 D-1 -> D
    hr = df.index.hour
    night = (hr >= 18) | (hr <= 9)
    g = df[night].groupby(key[night])
    out = pd.DataFrame({
        "tmin_night_c": g["temperature_2m"].min(),
        "tdmin_night_c": g["dew_point_2m"].min(),
        "rh_mean_night_pct": g["rh"].mean(),
        "wind_max_kmh": g["wind_speed_10m"].max(),
        "gust_max_kmh": g["wind_gusts_10m"].max(),
        "wind_mean_kmh": g["wind_speed_10m"].mean(),
        "cloud_mean_pct": g["cloud_cover"].mean(),
        "n_hours": g["temperature_2m"].count(),
    })
    # precipitation over the 24 h ending 09:00 on D (09:00 D-1 < t <= 09:00 D)
    pkey = (df.index + pd.Timedelta(hours=14)).normalize()
    out["precip_24h_mm"] = df["precipitation"].groupby(pkey).sum()
    cal = df.groupby(df.index.normalize())
    out["tmin_calday_c"] = cal["temperature_2m"].min()
    out["tmax_calday_c"] = cal["temperature_2m"].max()
    six = df[hr == 6]
    six.index = six.index.normalize()
    out["soil_moist_06_m3m3"] = six["soil_moisture_0_to_7cm"]
    out["soil_temp_06_c"] = six["soil_temperature_0_to_7cm"]
    out = out[out["n_hours"] >= 14].drop(columns="n_hours")
    out.index.name = "date"
    out = out.reset_index()
    out.insert(0, "station", station)
    return out


def cmd_aggregate():
    import pandas as pd
    frames = []
    for p in sorted(RAW.glob("*.json.gz")):
        st = p.name.split("_")[0]
        frames.append(nightly_from_raw(st, json.loads(gzip.decompress(p.read_bytes()))))
    df = pd.concat(frames).drop_duplicates(["station", "date"]).sort_values(["station", "date"])
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    for c in df.columns[2:]:
        df[c] = df[c].astype("float32")
    df.to_parquet(NIGHTLY, index=False)
    print(f"nightly table: {len(df)} station-nights, {df.station.nunique()} stations -> {NIGHTLY} "
          f"({NIGHTLY.stat().st_size / 1e6:.2f} MB)")
    join_report(df)


def joined(df=None):
    """Freeze rows (all splits) joined with the archive night of the target date and lag days."""
    import pandas as pd
    if df is None:
        df = pd.read_parquet(NIGHTLY)
    d = load()
    rows = []
    for split in ("train", "validation", "cases"):
        for r in d[split]:
            f = r["features"]
            rows.append({"id": r["id"], "split": split, "station": r["station"], "date": r["date"],
                         "target": r.get("target", -1), **{f"f{i}": v for i, v in enumerate(f)}})
    fr = pd.DataFrame(rows)
    return fr.merge(df, on=["station", "date"], how="left"), df


def join_report(df):
    import pandas as pd
    fr, _ = joined(df)
    lab = fr[(fr.split != "cases") & fr.tmin_night_c.notna()]
    res = {"n_freeze_rows_labelled": int((fr.split != "cases").sum()),
           "n_joined_labelled": int(len(lab)),
           "n_joined_cases": int(((fr.split == "cases") & fr.tmin_night_c.notna()).sum()),
           "n_stations_joined": int(lab.station.nunique())}
    for col in ("tmin_night_c", "tmin_calday_c"):
        pred = (lab[col] < 0).astype(int)
        y = lab.target.values
        tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        best = max(((float((((lab[col] < t).astype(int)) == y).mean()), t) for t in np.arange(-3, 3.01, 0.25)))
        res[col] = {"agreement_threshold_0C": round(float((pred == y).mean()), 4),
                    "freeze_recall": round(tp / max(tp + fn, 1), 4),
                    "freeze_precision": round(tp / max(tp + fp, 1), 4),
                    "best_threshold_c": round(best[1], 2), "agreement_best_threshold": round(best[0], 4),
                    "base_rate_target": round(float(y.mean()), 4),
                    "archive_freeze_rate": round(float(pred.mean()), 4)}
    # station tmin lags vs archive tmin on the same (lagged) day
    a = df.assign(o=pd.to_datetime(df.date)).set_index(["station", "o"])["tmin_calday_c"]
    corr = {}
    for lag, fi in ((3, 0), (4, 1), (7, 2)):
        o = pd.to_datetime(lab.date) - pd.Timedelta(days=lag)
        v = a.reindex(list(zip(lab.station, o))).values
        m = ~np.isnan(v)
        corr[f"tmin_lag{lag}"] = {"pearson_r": round(float(np.corrcoef(v[m], lab[f"f{fi}"].values[m])[0, 1]), 4),
                                  "mean_bias_archive_minus_station_c": round(float((v[m] - lab[f"f{fi}"].values[m]).mean()), 3),
                                  "n": int(m.sum())}
    res["archive_calday_tmin_vs_station_lag_features"] = corr
    per = lab.groupby("station").apply(lambda g: float(((g.tmin_night_c < 0).astype(int) == g.target).mean()))
    res["per_station_agreement_night_0C"] = {"min": round(float(per.min()), 3), "median": round(float(per.median()), 3),
                                             "max": round(float(per.max()), 3),
                                             "worst3": {k: round(v, 3) for k, v in per.nsmallest(3).items()}}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "join.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fetch", "aggregate"])
    ap.add_argument("--max-minutes", type=float, default=45)
    ap.add_argument("--train-oldest-first", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "fetch":
        cmd_fetch(a.max_minutes, a.train_oldest_first)
    else:
        cmd_aggregate()


if __name__ == "__main__":
    sys.exit(main())
