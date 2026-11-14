"""Imagination experiment: does pre-training the simulated culture on nights imagined
by the learned world model help when real labelled nights are scarce?

Regimes (per seed, real budget N in {25, 50}):
  real      : N real samples
  imag+real : 150 imagined samples first, then N real
  imag      : 150 imagined samples only
for culture_teach and, for reference, logistic regression on the same level code.
Everything is scored on the real fixed eval set (FrostTask.Xev/yev, AUC). Each trained
culture also plays 3 real validation seasons in FrostWorld.

Usage:
  python -m neurorover.world.bench_world --seeds 8 --procs 4
  python -m neurorover.world.bench_world --report        # rebuild REPORT.md from JSONs
"""
from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import pathlib
import time

import numpy as np

from ..bench import Culture, eval_culture, fit_logistic, run_trial
from ..env.frost import ENCODED, FrostTask, auc
from .env import FrostWorld, fit_tmin_threshold, run_season
from .model import get_model

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "results"
BUDGETS = [25, 50]
N_IMAG = 150
# three coldest stations by TRAIN freeze rate, one validation winter each (fixed a priori)
SEASONS = [("USW00093209", "2016-12-01"), ("USW00093227", "2017-12-01"), ("USW00053119", "2018-12-01")]
N_NIGHTS = 60


def imagined_set(seed: int, task: FrostTask):
    rows = get_model().imagine_rows(N_IMAG, np.random.default_rng(777_000 + seed), balanced=True)
    X = np.array([r["features"] for r in rows], float)[:, ENCODED]
    return X, np.array([r["target"] for r in rows], int)


def culture_seasons(c: Culture, seed: int) -> list:
    c2 = copy.deepcopy(c)
    c2.rule = "frozen"
    c2.elig = None
    rng = np.random.default_rng(555 + seed)
    env = FrostWorld(N_NIGHTS)
    pol = lambda obs, e: run_trial(c2, obs, rng)[2]
    return [run_season(env, pol, seed=0, station=s, source="real", start_date=d) for s, d in SEASONS]


def checkpoint(c, task, seed):
    _, scores = eval_culture(c, task, np.random.default_rng(999 + seed))
    seas = culture_seasons(c, seed)
    return {"auc": auc(scores, task.yev), "frac_deploy_eval": float(np.mean(scores > 0)),
            "season_reward": float(np.mean([s["reward"] for s in seas])), "seasons": seas}


def run_job(arm: str, seed: int) -> dict:
    """arm 'R': real only (checkpoints at 25, 50). arm 'I': 150 imagined (checkpoint), then real."""
    t0 = time.perf_counter()
    task = FrostTask(seed)
    rng = np.random.default_rng(seed)
    Xr, yr = task.train_stream(max(BUDGETS))
    Lr = task.levels(Xr)
    c = Culture(seed=seed, rule="teach")
    c.run(1000)
    res = {"arm": arm, "seed": seed, "culture": {}, "logistic": {}}
    if arm == "I":
        Xi, yi = imagined_set(seed, task)
        Li = task.levels(Xi)
        for i in range(N_IMAG):
            run_trial(c, Li[i], rng, label=int(yi[i]), rule="teach")
        res["culture"]["imag"] = checkpoint(c, task, seed)
        Lev = task.levels(task.Xev)
        res["logistic"]["imag"] = auc(fit_logistic(Li, yi)(Lev), task.yev)
        res["imag_freeze_frac"] = float(yi.mean())
    done = 0
    for n in BUDGETS:
        for i in range(done, n):
            run_trial(c, Lr[i], rng, label=int(yr[i]), rule="teach")
        done = n
        key = f"real{n}" if arm == "R" else f"imag+real{n}"
        res["culture"][key] = checkpoint(c, task, seed)
        Lev = task.levels(task.Xev)
        if arm == "R":
            res["logistic"][key] = auc(fit_logistic(Lr[:n], yr[:n])(Lev), task.yev)
        else:
            res["logistic"][key] = auc(fit_logistic(np.r_[Li, Lr[:n]], np.r_[yi, yr[:n]])(Lev), task.yev)
    res["wall_s"] = round(time.perf_counter() - t0, 1)
    return res


def _worker(args):
    arm, seed = args
    out = OUT / f"world_{arm}_s{seed}.json"
    if out.exists():
        return json.loads(out.read_text())
    r = run_job(arm, seed)
    out.write_text(json.dumps(r))
    print(f"{arm} s{seed}: " + " ".join(f"{k}:{v['auc']:.3f}" for k, v in r["culture"].items()) + f" ({r['wall_s']}s)", flush=True)
    return r


# ---------------------------------------------------------------- report
def boot_ci(d, n=10000):
    d = np.asarray(d, float)
    rng = np.random.default_rng(0)
    m = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return float(d.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def reference_seasons() -> dict:
    env = FrostWorld(N_NIGHTS)
    thr = fit_tmin_threshold()
    pols = {"always_hold": lambda o, e: 0, "always_deploy": lambda o, e: 1,
            f"tmin_lag3 < {thr:g} C (thr fit on train)": lambda o, e: e.raw[0] < thr}
    out = {}
    for name, p in pols.items():
        seas = [run_season(env, p, seed=0, station=s, source="real", start_date=d) for s, d in SEASONS]
        out[name] = {"reward": float(np.mean([x["reward"] for x in seas])), "deploys": float(np.mean([x["deploys"] for x in seas])),
                     "damaged": float(np.mean([x["damaged_nights"] for x in seas]))}
    out["_freezes_per_season"] = [x["freezes"] for x in seas]
    return out


def report():
    jobs = {}
    for p in sorted(OUT.glob("world_[RI]_s*.json")):
        r = json.loads(p.read_text())
        jobs[(r["arm"], r["seed"])] = r
    seeds = sorted(s for (a, s) in jobs if a == "R" and ("I", s) in jobs)
    get = lambda learner, key, field=None: np.array([
        (lambda v: v if field is None else v[field])(jobs[("R" if key.startswith("real") else "I", s)][learner][key]) for s in seeds])
    fmt = lambda x: f"{np.mean(x):.3f} +/- {np.std(x, ddof=1) / np.sqrt(len(x)):.3f}"
    ci = lambda d: "{:+.3f} [{:+.3f}, {:+.3f}]".format(*boot_ci(d))
    wm = json.loads((OUT / "world_model_eval.json").read_text())
    ref = reference_seasons()
    summary = {"seeds": seeds, "auc": {}, "paired": {}, "season": {}, "reference_policies": ref}
    L = ["# Imagination experiment: world-model pre-training for the simulated culture", "",
         "Everything here is a SIMULATED culture (Izhikevich network), not wetware. The world model is fitted on the",
         "train split only; all AUCs are on the real fixed 300-row eval set (150 freeze / 150 no-freeze) from the",
         "validation split. Nothing was tuned on the eval set; this is the first and only configuration run.", "",
         "## 1. World model (per-station 2-harmonic climatology + pooled VAR(1) on tmin/tmax anomalies, Gaussian residuals)", "",
         f"Fitted on {wm['fit']['n_reconstructed_station_days']} station-days reconstructed from the lag3/lag4/lag7 features of "
         f"{wm['fit']['n_stations']} stations (train split). A = {wm['fit']['A']}.", "",
         "RMSE on the validation split (deg C, tmin / tmax):", "",
         "| horizon | pairs | world model | persistence | climatology |", "|---|---|---|---|---|"]
    for k, v in wm["validation"].items():
        f2 = lambda a: f"{a[0]:.2f} / {a[1]:.2f}"
        L.append(f"| {k.replace('_', '-')} | {v['n_pairs']} | {f2(v['rmse_model_tmin_tmax'])} | {f2(v['rmse_persistence_tmin_tmax'])} | {f2(v['rmse_climatology_tmin_tmax'])} |")
    fb = wm["freeze_base_rate"]
    L += ["", f"Freeze base rate: real train {fb['real_train']:.3f}, real validation {fb['real_validation']:.3f}, "
          f"imagined (20000 nights, train station/date distribution) {fb['imagined_20000_nights']:.3f}.", "",
          f"## 2. AUC on the real eval set (mean +/- s.e.m. over {len(seeds)} seeds: {seeds})", "",
          "| regime | culture_teach | logistic (levels) |", "|---|---|---|"]
    keys = ["real25", "imag+real25", "real50", "imag+real50", "imag"]
    names = {"real25": "25 real", "imag+real25": "150 imagined + 25 real", "real50": "50 real",
             "imag+real50": "150 imagined + 50 real", "imag": "150 imagined only"}
    for k in keys:
        a, b = get("culture", k, "auc"), get("logistic", k)
        summary["auc"][k] = {"culture": a.tolist(), "logistic": b.tolist()}
        L.append(f"| {names[k]} | {fmt(a)} | {fmt(b)} |")
    L += ["", "Paired differences across seeds (mean [95% bootstrap CI, 10000 resamples over seeds]):", "",
          "| contrast | culture_teach dAUC | logistic dAUC |", "|---|---|---|"]
    contrasts = [("imag+real25", "real25"), ("imag+real50", "real50"), ("imag", "real25"), ("imag", "real50")]
    verdicts = []
    for x, y in contrasts:
        dc = get("culture", x, "auc") - get("culture", y, "auc")
        dl = get("logistic", x) - get("logistic", y)
        summary["paired"][f"{x} - {y}"] = {"culture": boot_ci(dc), "logistic": boot_ci(dl)}
        L.append(f"| ({names[x]}) - ({names[y]}) | {ci(dc)} | {ci(dl)} |")
        m, lo, hi = boot_ci(dc)
        verdicts.append((x, y, m, lo, hi))
    L += ["", f"## 3. Season reward in FrostWorld (3 real validation seasons x {N_NIGHTS} nights; reward -1 per deploy, -10 per unprotected freeze)", "",
          "Seasons: " + ", ".join(f"{s} from {d}" for s, d in SEASONS) + f" (the three coldest stations by train freeze rate; freeze nights per season {ref['_freezes_per_season']}).",
          "The culture acts on the sign of its decode score (DEPLOY if deploy-pool spikes > hold-pool spikes); no cost-aware threshold was fitted.", "",
          "| policy | mean season reward | deploys / season | damaged nights / season |", "|---|---|---|---|"]
    for name, v in ref.items():
        if not name.startswith("_"):
            L.append(f"| {name} | {v['reward']:.1f} | {v['deploys']:.1f} | {v['damaged']:.1f} |")
    for k in keys:
        arm = "R" if k.startswith("real") else "I"
        rw = get("culture", k, "season_reward")
        dep = np.array([np.mean([x["deploys"] for x in jobs[(arm, s)]["culture"][k]["seasons"]]) for s in seeds])
        dmg = np.array([np.mean([x["damaged_nights"] for x in jobs[(arm, s)]["culture"][k]["seasons"]]) for s in seeds])
        summary["season"][k] = rw.tolist()
        L.append(f"| culture_teach, {names[k]} | {fmt(rw)} | {dep.mean():.1f} | {dmg.mean():.1f} |")
    L += ["", "Paired season-reward differences (culture_teach):", ""]
    for x, y in contrasts[:2]:
        d = get("culture", x, "season_reward") - get("culture", y, "season_reward")
        summary["paired"][f"season {x} - {y}"] = boot_ci(d)
        L.append(f"- ({names[x]}) - ({names[y]}): {ci(d)}")
    L += ["", "## 4. Did imagination help?", ""]
    for x, y, m, lo, hi in verdicts[:2]:
        v = "HELPED (CI excludes 0)" if lo > 0 else "HURT (CI excludes 0)" if hi < 0 else "NO MEASURABLE DIFFERENCE (CI includes 0)"
        L.append(f"- culture_teach, {names[x]} vs {names[y]}: dAUC {m:+.3f} [{lo:+.3f}, {hi:+.3f}] -> {v}.")
    L += ["", "## 5. Caveats", "",
          "- 8 seeds; bootstrap CIs over 8 paired values are rough and somewhat too narrow.",
          "- The imagination arm sees 150 more teaching trials in total, so any gain confounds 'imagined data' with 'more trials';",
          "  the fair reading is 'imagined trials are free, real labelled nights are not'.",
          "- The world model is linear-Gaussian and fitted on series reconstructed from lag features (Oct-Apr station-days only);",
          "  it has no cloud/wind/humidity, so imagined freezes are just cold-tail draws of a VAR(1).",
          "- Imagined rows reuse the empirical train (station, date) distribution; only the weather is generated.",
          "- Season rewards: 3 seasons at cold stations where always-DEPLOY is near-optimal under the -1/-10 costs; the culture's",
          "  action threshold is uncalibrated, so season reward mostly reflects its deploy bias, not its ranking skill.",
          "- 'Real' seasons are consecutive available validation rows (near-daily, occasional gaps).", ""]
    (HERE / "REPORT.md").write_text("\n".join(L))
    (OUT / "world_bench_summary.json").write_text(json.dumps(summary, indent=1))
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    if not a.report:
        jobs = [(arm, s) for s in range(a.seeds) for arm in ("I", "R")]
        with mp.Pool(min(a.procs, 4)) as p:
            p.map(_worker, jobs, chunksize=1)
    report()


if __name__ == "__main__":
    main()
