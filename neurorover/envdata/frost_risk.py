"""Does real weather context help freeze prediction?

Logistic regression (neurorover.bench.fit_logistic) on train rows, AUC on validation rows:
  base     : the 14 encoded freeze features (neurorover.env.frost.ENCODED)
  +lagged  : + Open-Meteo covariates from the INPUT window only (nights D-3, D-4 and the D-9..D-3 mean),
             i.e. information available at decision time (no leakage)
  oracle   : + covariates of the target night D itself  -> LEAKAGE, upper bound only

    python -m neurorover.envdata.frost_risk   -> results/frost_risk.json
"""
from __future__ import annotations

import json

import numpy as np

from .fetch import NIGHTLY, RESULTS, joined

COV = ["tmin_night_c", "tdmin_night_c", "rh_mean_night_pct", "wind_mean_kmh", "gust_max_kmh", "cloud_mean_pct",
       "precip_24h_mm", "soil_moist_06_m3m3", "soil_temp_06_c"]
MEAN7 = ["cloud_mean_pct", "precip_24h_mm", "wind_mean_kmh", "tdmin_night_c", "soil_moist_06_m3m3"]


def auc(s, y):
    from scipy.stats import rankdata
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def build():
    import pandas as pd
    fr, nd = joined()
    fr = fr[fr.split.isin(["train", "validation"])].copy()
    a = nd.assign(o=pd.to_datetime(nd.date)).set_index(["station", "o"])
    t = pd.to_datetime(fr.date)

    def at(lag, cols):
        idx = list(zip(fr.station, t - pd.Timedelta(days=lag)))
        return a.reindex(idx)[cols].to_numpy(float)
    lag3, lag4 = at(3, COV), at(4, COV)
    with np.errstate(all="ignore"):
        import warnings; warnings.simplefilter("ignore", RuntimeWarning)
        m7 = np.nanmean(np.stack([at(k, MEAN7) for k in range(3, 10)]), 0)
    tgt = fr[COV].to_numpy(float)
    names = ([f"{c}@D-3" for c in COV] + [f"{c}@D-4" for c in COV] + [f"{c}@mean(D-9..D-3)" for c in MEAN7])
    return fr, np.c_[lag3, lag4, m7], tgt, names


def main():
    from ..bench import fit_logistic
    from ..env.frost import ENCODED
    fr, L, T, names = build()
    F = fr[[f"f{i}" for i in ENCODED]].to_numpy(float)
    y = fr.target.to_numpy(int)
    ok = ~np.isnan(L).any(1) & ~np.isnan(T).any(1)
    tr = (fr.split.values == "train") & ok
    va = (fr.split.values == "validation") & ok
    res = {"n_train_rows_with_covariates": int(tr.sum()), "n_validation_rows_with_covariates": int(va.sum()),
           "n_train_rows_total": int((fr.split.values == "train").sum()),
           "n_validation_rows_total": int((fr.split.values == "validation").sum()),
           "freeze_rate_val_subset": round(float(y[va].mean()), 4), "lagged_covariates": names,
           "auc_validation": {}}
    small = [names.index(f"{c}@D-3") for c in ("tdmin_night_c", "cloud_mean_pct", "wind_mean_kmh", "rh_mean_night_pct")]
    sets = {"base_14_features": F, "base+lagged_covariates": np.c_[F, L],
            "base+D-3_dewpoint_cloud_wind_rh": np.c_[F, L[:, small]],
            "ORACLE_LEAKAGE_base+target_night_covariates": np.c_[F, L, T],
            "archive_target_night_tmin_only_LEAKAGE": T[:, :1]}
    for k, X in sets.items():
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        Z = (X - mu) / sd
        res["auc_validation"][k] = {f"lam={lam:g}": round(auc(fit_logistic(Z[tr], y[tr], lam=lam)(Z[va]), y[va]), 4)
                                    for lam in (1.0, 10.0, 100.0)}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "frost_risk.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "lagged_covariates"}, indent=1))


if __name__ == "__main__":
    main()
