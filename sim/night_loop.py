"""Close the loop: weather -> simulated culture (culture_teach) -> DEPLOY/HOLD -> MuJoCo rover episode (PPO policy).

    .venv/Scripts/python -m sim.night_loop --run ppo_s0 --nights 30

1. Train a culture_teach Culture on 100 class-balanced training samples with the unmodified neurorover.bench
   functions (run_trial with rule="teach"), seed 0. Cached in sim/runs/culture_teach_s0_n100.pkl.
2. For each night of a real validation season (FrostWorld source="real"), encode tonight's features exactly as
   the benchmark does, ask the frozen culture for DEPLOY/HOLD, step FrostWorld for the truth and crop outcome.
3. Run one randomised MuJoCo rover episode with that decision (the drive happens either way: the rover parks at
   the row's station to read the SHT45; it only extends the cover on DEPLOY).
Writes sim/runs/<run>/season.csv and season_summary.json / .md
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import pathlib
import pickle

import numpy as np

from neurorover.bench import run_trial
from neurorover.culture.core import Culture
from neurorover.env.frost import FrostTask
from neurorover.world.env import FrostWorld, fit_tmin_threshold

from . import params as P
from .rover_env import FrostRoverEnv

ROOT = pathlib.Path(__file__).resolve().parent


def trained_culture(n=100, seed=0):
    cache = ROOT / "runs" / f"culture_teach_s{seed}_n{n}.pkl"
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    task = FrostTask(seed)
    rng = np.random.default_rng(seed)
    c = Culture(seed=seed, rule="teach")
    c.run(1000)
    Xs, ys = task.train_stream(n)
    L = task.levels(Xs)
    for i in range(n):
        run_trial(c, L[i], rng, label=int(ys[i]), rule="teach")
    c2 = copy.deepcopy(c); c2.rule = "frozen"; c2.elig = None   # same freeze as bench.eval_culture
    cache.write_bytes(pickle.dumps(c2))
    return c2


def night_conditions(station, date):
    """Hook for neurorover.envdata.EnvGenerator (real-data nightly wind, soil moisture -> friction, sensor noise).
    Returns a dict of FrostRoverEnv.dr overrides, or None to use the default domain randomisation."""
    try:
        from neurorover.envdata import EnvGenerator  # provided by another agent; may not exist yet
    except Exception:
        return None
    try:
        g = night_conditions.gen = getattr(night_conditions, "gen", None) or EnvGenerator()
        c = g.night(station, date)
        out = {}
        if "wind_ms" in c:      # dynamic pressure on a 0.12 m2 panel, Cd ~1.2 (EST)
            out["wind"] = 0.5 * 1.2 * 1.2 * c["wind_ms"] ** 2 * 0.12
        if "friction" in c:
            out["friction"] = float(c["friction"])
        if "noise_level" in c:
            out["noise"] = float(c["noise_level"])
        return out or None
    except Exception as e:  # never let the hook break the season loop
        print("envdata hook failed:", e)
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="ppo_s0")
    ap.add_argument("--nights", type=int, default=30)
    ap.add_argument("--station", default=None)
    ap.add_argument("--start", default="2016-12-01")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    from stable_baselines3 import PPO
    out = ROOT / "runs" / args.run
    model = PPO.load(out / "best_model", device="cpu")
    vn = pickle.loads((out / "vecnormalize.pkl").read_bytes()); vn.training = False
    print("training culture_teach on 100 samples ...", flush=True)
    culture = trained_culture()
    rng = np.random.default_rng(args.seed)
    world = FrostWorld(n_nights=args.nights)
    # pick a station whose season has at least a few freezes, deterministic
    stations = world.stations()
    st = args.station
    if st is None:
        for s in stations:
            world.reset(seed=args.seed, station=s, source="real", start_date=args.start)
            if 3 <= world.y.sum() <= len(world.y) - 3:
                st = s; break
    obs = world.reset(seed=args.seed, station=st, source="real", start_date=args.start)
    thr = fit_tmin_threshold()
    env = FrostRoverEnv(randomize=True)
    rows = []
    done = False
    night = 0
    while not done:
        levels = obs
        _, score, decision, _ = run_trial(culture, levels, rng)
        raw = world.raw
        obs, rew, done, info = world.step(int(decision))
        cond = night_conditions(st, info["date"])
        o, _ = env.reset(seed=10_000 + night, options={"deploy": bool(decision), "full_task": True, "dr_override": cond})
        ep_done = False
        while not ep_done:
            a, _ = model.predict(vn.normalize_obs(o[None]), deterministic=True)
            o, r, te, tr, inf = env.step(a[0])
            ep_done = te or tr
        m = inf["episode_metrics"]
        truth = info["truth"]
        sht45 = float(raw[0] + rng.normal(0, P.SHT45_NOISE))   # tmin_lag3 stands in for the dusk air reading
        protected = bool(decision and m["success"])
        rows.append(dict(night=night, date=info["date"], station=st, decision="DEPLOY" if decision else "HOLD",
                         culture_score=round(float(score), 2), truth_freeze=truth, rule_tmin=int(raw[0] < thr),
                         drive_success=int(m["success"]), cover_coverage=round(m["coverage"], 2),
                         time_s=round(m["time"], 1), energy_J=round(m["energy_J"], 0), crop_wheel_steps=m["crop_steps"],
                         tipped=int(m["tipped"]), sht45_reading_C=round(sht45, 2),
                         crop_outcome=("protected" if truth and protected else
                                       "DAMAGED (frost, no cover)" if truth and not protected else
                                       "trampled" if m["crop_steps"] > 0 else "ok"),
                         world_reward=rew, envdata_conditions=int(cond is not None), wind_N=round(env.dr["wind"], 2), friction=round(env.dr["friction"], 2), crop_health=round(info["crop_health"], 2)))
        print(rows[-1], flush=True)
        night += 1
    with open(out / "season.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    n = len(rows)
    dep = [r for r in rows if r["decision"] == "DEPLOY"]
    frz = [r for r in rows if r["truth_freeze"]]
    summ = dict(station=st, start=args.start, nights=n, freezes=len(frz), deploys=len(dep),
                decision_accuracy=float(np.mean([(r["decision"] == "DEPLOY") == bool(r["truth_freeze"]) for r in rows])),
                freezes_protected=sum(r["crop_outcome"] == "protected" for r in rows),
                freezes_damaged=sum(r["crop_outcome"].startswith("DAMAGED") for r in rows),
                drive_success_rate=float(np.mean([r["drive_success"] for r in rows])),
                deploy_drive_success_rate=float(np.mean([r["drive_success"] for r in dep])) if dep else None,
                nights_with_wheel_on_crops=sum(r["crop_wheel_steps"] > 0 for r in rows),
                total_energy_Wh=round(sum(r["energy_J"] for r in rows) / 3600, 2),
                battery_fraction_used=round(sum(r["energy_J"] for r in rows) / 3600 / (P.BATT_VOLTAGE * P.BATT_CAPACITY_AH), 4),
                season_reward_frostworld=world.total, final_crop_health=rows[-1]["crop_health"],
                tmin_rule_accuracy=float(np.mean([r["rule_tmin"] == r["truth_freeze"] for r in rows])))
    (out / "season_summary.json").write_text(json.dumps(summ, indent=1))
    md = ["| metric | value |", "|---|---|"] + [f"| {k} | {v} |" for k, v in summ.items()]
    (out / "season_summary.md").write_text("\n".join(md) + "\n")
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
