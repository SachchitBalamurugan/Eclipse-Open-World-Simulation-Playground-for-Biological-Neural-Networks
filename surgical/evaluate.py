"""Evaluate all controllers on the same 50 held-out targets and seeds -> results/eval.json."""
import json
import pathlib
import pickle
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from surgical import controllers as C  # noqa: E402
from surgical.arm import make_targets  # noqa: E402
from surgical.reach_env import SurgicalReachEnv  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent / "results"


class PPOPolicy:
    name = "ppo"

    def __init__(self, path):
        from stable_baselines3 import PPO
        self.model = PPO.load(path, device="cpu")

    def reset(self, env):
        pass

    def act(self, env, obs):
        return self.model.predict(obs, deterministic=True)[0]


def summarize(R):
    t = [r["reach_time_s"] for r in R if r["reach_time_s"] is not None]
    return dict(n=len(R), success_rate=float(np.mean([r["success"] for r in R])),
                final_err_mm_median=float(np.median([r["final_err_mm"] for r in R])),
                final_err_mm_mean=float(np.mean([r["final_err_mm"] for r in R])),
                time_to_reach_s_median=(float(np.median(t)) if t else None),
                reached_1cm_frac=len(t) / len(R),
                joint_limit_steps_total=int(sum(r["limit_hits"] for r in R)),
                collision_contacts_total=int(sum(r["collisions"] for r in R)),
                episodes_with_collision=int(sum(r["collisions"] > 0 for r in R)))


def main():
    env = SurgicalReachEnv(seed=0)
    held = make_targets(env.m, 50, seed=999)
    res = {}
    ctrls = [C.DLSIK(), C.RandomPolicy(seed=0)]
    if (OUT / "ppo_reach.zip").exists():
        ctrls.append(PPOPolicy(OUT / "ppo_reach.zip"))
    for c in ctrls:
        R = [C.run_episode(env, c, held[i], seed=i) for i in range(50)]
        res[c.name] = summarize(R)
        print(c.name, res[c.name], flush=True)
    for mode in ("teach", "frozen"):
        f = OUT / f"curve_{mode}.json"
        if f.exists():
            d = json.loads(f.read_text())
            res["culture_" + mode] = summarize(d["final"])
            res["culture_" + mode]["teaching_steps"] = d["teaching_steps"]
            print("culture_" + mode, res["culture_" + mode])
    meta = dict(held_out_targets=held.tolist(), success="tip within 10 mm for 0.5 s (10 control steps)",
                episode_s=4.0, control_hz=20, seeds="episode i uses env seed i (same DR draw for every controller)")
    (OUT / "eval.json").write_text(json.dumps(dict(meta=meta, results=res), indent=1))


if __name__ == "__main__":
    main()
