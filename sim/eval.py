"""Evaluate the PPO policy against the hand-written controller on 100 fixed-seed randomised episodes.

    .venv/Scripts/python -m sim.eval --run ppo_s0 [--model best_model]

Same seeds, same domain randomisation draw, same noisy/delayed observations for both. Half DEPLOY, half HOLD.
Also evaluates on the nominal (no-randomisation) model to show the sim-to-sim gap.
Writes sim/runs/<run>/eval.json and eval.md
"""
from __future__ import annotations

import argparse
import json
import pathlib
import pickle

import numpy as np

from .controller import PController
from .train_ppo import run_policy_episodes

ROOT = pathlib.Path(__file__).resolve().parent
EVAL_SEEDS = list(range(500_000, 500_100))


def summarize(ms):
    ok = [m for m in ms if m["success"]]
    dep = [m for m in ms if m["deploy"]]
    return dict(
        episodes=len(ms), success_rate=float(np.mean([m["success"] for m in ms])),
        success_deploy=float(np.mean([m["success"] for m in dep])) if dep else None,
        success_hold=float(np.mean([m["success"] for m in ms if not m["deploy"]])),
        time_to_goal_s=float(np.mean([m["time_to_goal"] for m in ok])) if ok else None,
        energy_J=float(np.mean([m["energy_J"] for m in ms])),
        energy_J_successful=float(np.mean([m["energy_J"] for m in ok])) if ok else None,
        crop_collision_rate=float(np.mean([m["crop_hit"] for m in ms])),
        crop_wheel_steps=float(np.mean([m["crop_steps"] for m in ms])),
        tip_rate=float(np.mean([m["tipped"] for m in ms])),
        cover_coverage_on_deploy=float(np.mean([m["coverage"] for m in dep])) if dep else None,
        final_dist_m=float(np.median([m["final_dist"] for m in ms])),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="ppo_s0")
    ap.add_argument("--model", default="best_model")
    args = ap.parse_args()
    from stable_baselines3 import PPO
    out = ROOT / "runs" / args.run
    model = PPO.load(out / args.model, device="cpu")
    vnf = "vecnormalize.pkl" if args.model == "best_model" else "vecnormalize_final.pkl"
    vn = pickle.loads((out / vnf).read_bytes()); vn.training = False

    def ppo(o):
        a, _ = model.predict(vn.normalize_obs(o[None]), deterministic=True)
        return a[0]

    ctrl = PController()

    def pc(o):
        return ctrl(o)

    opts = lambda s: {"deploy": bool(s % 2), "full_task": True}

    def pc_runner(randomize):
        from .rover_env import FrostRoverEnv
        env = FrostRoverEnv(randomize=randomize); res = []
        for s in EVAL_SEEDS:
            o, _ = env.reset(seed=s, options=opts(s)); ctrl.reset(); done = False; R = 0
            while not done:
                o, r, te, tr, info = env.step(pc(o)); R += r; done = te or tr
            m = dict(info["episode_metrics"]); m["reward"] = R; res.append(m)
        return res

    res = {}
    for rnd in (True, False):
        tag = "randomized" if rnd else "nominal"
        res[f"ppo_{tag}"] = summarize(run_policy_episodes(ppo, EVAL_SEEDS, opts, randomize=rnd))
        res[f"controller_{tag}"] = summarize(pc_runner(rnd))
        print(tag, json.dumps({k: v for k, v in res.items() if tag in k}, indent=1), flush=True)
    (out / "eval.json").write_text(json.dumps(res, indent=1))
    keys = ["success_rate", "success_deploy", "success_hold", "time_to_goal_s", "energy_J", "energy_J_successful",
            "crop_collision_rate", "tip_rate", "cover_coverage_on_deploy", "final_dist_m"]
    cols = list(res)
    fmt = lambda v: "-" if v is None else (f"{v:.2f}" if abs(v) < 10 else f"{v:.0f}")
    md = ["| metric | " + " | ".join(cols) + " |", "|---|" + "---:|" * len(cols)]
    md += [f"| {k} | " + " | ".join(fmt(res[c][k]) for c in cols) + " |" for k in keys]
    (out / "eval.md").write_text(f"100 fixed-seed episodes (seeds {EVAL_SEEDS[0]}..{EVAL_SEEDS[-1]}), half DEPLOY / half HOLD, "
                                 "full task (random headland start).\n\n" + "\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
