"""Render a PPO rollout to MP4 (offscreen mujoco.Renderer) and export a replayable trajectory JSON.

    .venv/Scripts/python -m sim.render --run ppo_s0 --seed 500001 --deploy 1

Writes sim/runs/<run>/rollout.mp4, frames/frame_XXX.png (a few stills), trajectory.json
If offscreen GL fails, set MUJOCO_GL=egl or osmesa (Linux) / glfw (Windows default) before running.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import pickle

import imageio.v2 as imageio
import numpy as np

from . import params as P
from .rover_env import FrostRoverEnv

ROOT = pathlib.Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="ppo_s0")
    ap.add_argument("--model", default="best_model")
    ap.add_argument("--seed", type=int, default=500_001)
    ap.add_argument("--deploy", type=int, default=1)
    ap.add_argument("--policy", default="ppo", choices=["ppo", "controller"])
    ap.add_argument("--name", default="rollout")
    ap.add_argument("--no-props", action="store_true", help="omit the SimReady props (fence, yard, crates)")
    args = ap.parse_args()
    out = ROOT / "runs" / args.run
    if args.policy == "ppo":
        from stable_baselines3 import PPO
        model = PPO.load(out / args.model, device="cpu")
        vn = pickle.loads((out / "vecnormalize.pkl").read_bytes()); vn.training = False
        pol = lambda o: model.predict(vn.normalize_obs(o[None]), deterministic=True)[0][0]
    else:
        from .controller import PController
        c = PController(); pol = c
    env = FrostRoverEnv(randomize=True, props=not args.no_props and (ROOT / "assets" / "rover_props.xml").exists())
    o, info = env.reset(seed=args.seed, options={"deploy": bool(args.deploy), "full_task": True})
    frames, traj = [], []
    fdir = out / "frames"; fdir.mkdir(parents=True, exist_ok=True)
    done = False; t = 0; R = 0.0
    frames.append(env.render())
    while not done:
        a = pol(o)
        o, r, te, tr, inf = env.step(a)
        R += r; done = te or tr; t += 1
        frames.append(env.render())
        traj.append(dict(t=round(t * P.CTRL_DT, 3), action=[round(float(x), 3) for x in a], reward=round(r, 4),
                         boom_rad=round(float(env.d.qpos[env.boom_q]), 4), bodies=env.poses()))
    # hold the last frame for a second
    frames += [frames[-1]] * 20
    mp4 = out / f"{args.name}.mp4"
    imageio.mimsave(mp4, frames, fps=int(1 / P.CTRL_DT), codec="libx264", quality=7, macro_block_size=8)
    idx = sorted(set([0, len(frames) // 4, len(frames) // 2, (3 * len(frames)) // 4, len(frames) - 1]))
    for i in idx:
        imageio.imwrite(fdir / f"{args.name}_{i:04d}.png", frames[i])
    meta = dict(run=args.run, policy=args.policy, seed=args.seed, decision="DEPLOY" if args.deploy else "HOLD",
                dt=P.CTRL_DT, goal=[float(x) for x in env.goal], goal_yaw=env.goal_yaw, row_index=env.row,
                row_y=env.row_y, domain_randomization={k: (float(v) if np.isscalar(v) else [float(x) for x in v])
                                                       for k, v in env.dr.items()},
                episode_metrics=inf["episode_metrics"], total_reward=R,
                pose_format="per body: [x, y, z, qw, qx, qy, qz] in world frame (MuJoCo, z up, metres)",
                steps=traj)
    (out / f"trajectory{'' if args.name == 'rollout' else '_' + args.name}.json").write_text(json.dumps(meta))
    print("wrote", mp4, len(frames), "frames; stills", [f"{args.name}_{i:04d}.png" for i in idx]); print(inf["episode_metrics"])


if __name__ == "__main__":
    main()
