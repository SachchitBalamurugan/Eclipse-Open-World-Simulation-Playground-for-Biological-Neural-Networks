"""Train (paired-stimulation teaching) or freeze a simulated culture controller; learning curve.

usage: python surgical/train_culture.py teach|frozen [episodes] [lr]
Writes surgical/results/curve_<mode>.json and culture_<mode>.pkl.
"""
import json
import pathlib
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from surgical import controllers as C  # noqa: E402
from surgical.arm import make_targets  # noqa: E402
from surgical.culture_ctrl import CultureController  # noqa: E402
from surgical.reach_env import SurgicalReachEnv  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent / "results"
mode = sys.argv[1]
EPISODES = int(sys.argv[2]) if len(sys.argv) > 2 else 60
LR = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05
EVAL_EVERY, N_CURVE = 10, 6

env = SurgicalReachEnv(seed=0)
held = make_targets(env.m, 50, seed=999)          # 50 fixed held-out targets
cc = CultureController(seed=0, teach=(mode == "teach"), lr=LR)
train_rng = np.random.default_rng(1)


def evaluate(n):
    cc.learning = False
    R = [C.run_episode(env, cc, held[i], seed=i) for i in range(n)]
    cc.learning = cc.teach
    return R


curve = []
t0 = time.time()
for ep in range(EPISODES + 1):
    if ep % EVAL_EVERY == 0:
        R = evaluate(N_CURVE)
        row = dict(episode=ep, teaching_steps=cc.n_teach,
                   success=float(np.mean([r["success"] for r in R])),
                   median_err_mm=float(np.median([r["final_err_mm"] for r in R])),
                   mean_err_mm=float(np.mean([r["final_err_mm"] for r in R])),
                   wall_s=time.time() - t0)
        curve.append(row)
        print(json.dumps(row), flush=True)
    if ep < EPISODES:
        C.run_episode(env, cc, env.pool[train_rng.integers(len(env.pool))], seed=1000 + ep)

final = evaluate(50)
c = cc.culture
summary = dict(mode=mode, lr=LR, episodes=EPISODES, teaching_steps=cc.n_teach, curve=curve, final=final,
               plastic_w_mean=float(c.W[c.plastic].mean()))
(OUT / f"curve_{mode}.json").write_text(json.dumps(summary, indent=1))
cc.learning = False
with open(OUT / f"culture_{mode}.pkl", "wb") as fh:
    pickle.dump(cc, fh)
print("final success", np.mean([r["success"] for r in final]), "median err",
      np.median([r["final_err_mm"] for r in final]), "wall", time.time() - t0)
