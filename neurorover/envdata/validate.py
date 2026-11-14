"""Validate EnvGenerator (fitted on 2000-2014) against held-out real nights (2015-2019).

    python -m neurorover.envdata.validate   -> results/validation.json + results/*.png
"""
from __future__ import annotations

import datetime as dt
import json

import numpy as np

from .generator import HERE, NIGHTLY, VARS, EnvGenerator

RES = HERE / "results"
HOLD = (2015, 2019)
COMPARE = ["air_tmin_c", "tdmin_night_c", "rh_mean_night_pct", "wind_max_kmh", "gust_max_kmh", "wind_mean_kmh",
           "cloud_mean_pct", "precip_24h_mm", "soil_moist_06_m3m3", "soil_temp_06_c"]
PAIRS = [("air_tmin_c", "cloud_mean_pct"), ("air_tmin_c", "wind_mean_kmh"), ("air_tmin_c", "tdmin_night_c"),
         ("soil_moist_06_m3m3", "precip_24h_mm"), ("cloud_mean_pct", "precip_24h_mm"),
         ("air_tmin_c", "soil_temp_06_c")]


def heldout_real():
    import pandas as pd
    from ..env.frost import load
    from ..world.model import reconstruct
    df = pd.read_parquet(NIGHTLY)
    t = pd.to_datetime(df.date)
    df = df[(t.dt.year >= HOLD[0]) & (t.dt.year <= HOLD[1]) & (t.dt.month.isin([10, 11, 12, 1, 2, 3, 4]))].copy()
    S = reconstruct(load()["validation"])
    o = pd.to_datetime(df.date).map(lambda x: x.toordinal()).values
    df["air_tmin_c"] = [S.get(s, {}).get(k, (np.nan,))[0] for s, k in zip(df.station, o)]
    return df.dropna(subset=VARS + ["air_tmin_c"]).reset_index(drop=True)


def spearman(a, b):
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).statistic)


def within_station(df, a, b):
    v = [spearman(g[a], g[b]) for _, g in df.groupby("station") if len(g) > 30]
    return float(np.nanmean(v))


def main(max_nights=12000, seed=0):
    import pandas as pd
    from scipy.stats import ks_2samp
    rng = np.random.default_rng(seed)
    gen = EnvGenerator().fit()
    real = heldout_real()
    real = real[real.station.isin(gen.stations)]
    if len(real) > max_nights:
        real = real.sample(max_nights, random_state=seed).reset_index(drop=True)
    G = [gen.sample_night(s, d, rng) for s, d in zip(real.station, real.date)]
    gdf = pd.DataFrame([{k: g[k] for k in ("station", "date")} |
                        {c: g[c] for c in COMPARE if c in g} |
                        {"ground_friction_mu": g["ground_friction_mu"], "cover_force_gust_N": g["cover_force_gust_N"],
                         "leaf_minus_air_c": g["leaf_minus_air_c"]} for g in G])
    res = {"fit": {"train_years": [2000, 2014], "n_analog_nights": int(len(gen.df)),
                   "n_stations": len(gen.stations),
                   "analog_seasons_per_station_median": float(gen.df.groupby("station").ord.apply(
                       lambda o: len({dt.date.fromordinal(int(k)).year for k in o})).median())},
           "heldout": {"years": list(HOLD), "n_real_nights": int(len(real)), "n_generated": int(len(gdf)),
                       "tmin_source_generated": "WorldModel.rollout (train-only VAR), not the real tmin"},
           "ks": {}, "correlations_spearman": {}, "freeze_frequency": {}}
    for c in COMPARE:
        k = ks_2samp(real[c], gdf[c])
        res["ks"][c] = {"ks_stat": round(float(k.statistic), 4),
                        "real_mean": round(float(real[c].mean()), 3), "gen_mean": round(float(gdf[c].mean()), 3),
                        "real_sd": round(float(real[c].std()), 3), "gen_sd": round(float(gdf[c].std()), 3)}
    for a, b in PAIRS:
        res["correlations_spearman"][f"{a}~{b}"] = {
            "real_pooled": round(spearman(real[a], real[b]), 3), "gen_pooled": round(spearman(gdf[a], gdf[b]), 3),
            "real_within_station": round(within_station(real, a, b), 3),
            "gen_within_station": round(within_station(gdf, a, b), 3)}
    fr = real.groupby("station").air_tmin_c.apply(lambda x: float((x < 0).mean()))
    fg = gdf.groupby("station").air_tmin_c.apply(lambda x: float((x < 0).mean()))
    j = pd.concat([fr.rename("real"), fg.rename("gen")], axis=1).dropna()
    res["freeze_frequency"] = {"overall_real": round(float((real.air_tmin_c < 0).mean()), 4),
                               "overall_gen": round(float((gdf.air_tmin_c < 0).mean()), 4),
                               "per_station_pearson_r": round(float(np.corrcoef(j.real, j.gen)[0, 1]), 4),
                               "per_station_mae": round(float((j.real - j.gen).abs().mean()), 4),
                               "per_station": {s: [round(r, 3), round(g, 3)] for s, (r, g) in j.iterrows()}}
    # day-to-day persistence: lag-1 autocorrelation in 30-night seasons (Nov 1 start)
    def ac(x):
        x = np.asarray(x, float); return float(np.corrcoef(x[:-1], x[1:])[0, 1]) if x.std() > 0 else np.nan
    pers = {"soil_moist_06_m3m3": ([], []), "cloud_mean_pct": ([], []), "air_tmin_c": ([], [])}
    for s in gen.stations:
        g = real[real.station == s]
        for y in range(HOLD[0], HOLD[1]):
            seq = heldout_seq(s, y)
            if seq is not None:
                for v in pers:
                    pers[v][0].append(ac(seq[v]))
            sim = gen.sample_season(s, f"{y}-11-01", 30, rng)
            for v in pers:
                pers[v][1].append(ac([n[v] for n in sim]))
    res["persistence_lag1_autocorr"] = {v: {"real": round(float(np.nanmean(a)), 3), "gen": round(float(np.nanmean(b)), 3)}
                                        for v, (a, b) in pers.items()}
    res["physics_generated_summary"] = {c: {"p5": round(float(gdf[c].quantile(.05)), 3),
                                            "median": round(float(gdf[c].median()), 3),
                                            "p95": round(float(gdf[c].quantile(.95)), 3)}
                                        for c in ("ground_friction_mu", "cover_force_gust_N", "leaf_minus_air_c")}
    res["soil_probe_model"] = gen.soil.summary()
    RES.mkdir(exist_ok=True)
    (RES / "validation.json").write_text(json.dumps(res, indent=1))
    plots(real, gdf, j)
    print(json.dumps({k: res[k] for k in ("fit", "heldout", "ks", "correlations_spearman",
                                          "persistence_lag1_autocorr")}, indent=1))
    print("freeze freq", {k: v for k, v in res["freeze_frequency"].items() if k != "per_station"})


_SEQ = {}


def heldout_seq(station, year):
    """Real 30-night sequence from Nov 1 of `year` (held-out years), or None if incomplete."""
    import pandas as pd
    if "r" not in _SEQ:
        _SEQ["r"] = heldout_real().set_index(["station", "date"])
    r = _SEQ["r"]
    dates = [(dt.date(year, 11, 1) + dt.timedelta(days=i)).isoformat() for i in range(30)]
    keys = [(station, d) for d in dates]
    sub = r.reindex(keys)
    if sub.air_tmin_c.isna().sum() > 0:
        return None
    return {c: sub[c].values for c in ("soil_moist_06_m3m3", "cloud_mean_pct", "air_tmin_c")}


def plots(real, gdf, j):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 5, figsize=(18, 7))
    for ax, c in zip(axs.flat, COMPARE):
        lo, hi = np.nanpercentile(np.r_[real[c], gdf[c]], [0.5, 99.5])
        bins = np.linspace(lo, hi, 40)
        ax.hist(real[c], bins, density=True, alpha=.55, label="real 2015-19", color="#3b6ea5")
        ax.hist(gdf[c], bins, density=True, alpha=.55, label="generated", color="#d9822b")
        ax.set_title(c, fontsize=9)
    axs.flat[0].legend(fontsize=8)
    fig.suptitle("EnvGenerator (fit 2000-2014) vs held-out real nights 2015-2019: marginals")
    fig.tight_layout(); fig.savefig(RES / "marginals.png", dpi=90); plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (a, b) in zip(axs, [("cloud_mean_pct", "air_tmin_c"), ("wind_mean_kmh", "air_tmin_c"),
                                ("precip_24h_mm", "soil_moist_06_m3m3")]):
        ax.scatter(real[a], real[b], s=2, alpha=.25, label="real", color="#3b6ea5")
        ax.scatter(gdf[a], gdf[b], s=2, alpha=.25, label="generated", color="#d9822b")
        ax.set_xlabel(a); ax.set_ylabel(b)
        if a == "precip_24h_mm":
            ax.set_xscale("symlog")
    axs[0].legend(markerscale=5)
    fig.tight_layout(); fig.savefig(RES / "joint.png", dpi=90); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(j.real, j.gen, color="#3b6ea5")
    m = max(j.real.max(), j.gen.max()) * 1.05
    ax.plot([0, m], [0, m], "k--", lw=1)
    ax.set_xlabel("real freeze frequency (2015-19)"); ax.set_ylabel("generated freeze frequency")
    ax.set_title("per-station freeze frequency")
    fig.tight_layout(); fig.savefig(RES / "freeze_frequency.png", dpi=90); plt.close(fig)


if __name__ == "__main__":
    main()
