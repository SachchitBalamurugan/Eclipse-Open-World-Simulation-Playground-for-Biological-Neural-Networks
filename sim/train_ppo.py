"""Train the frost-rover drive+deploy policy with stable-baselines3 PPO on CPU.

    .venv/Scripts/python -m sim.train_ppo --run ppo_s0 --seed 0 --minutes 35 --envs 5

Writes sim/runs/<run>/: progress.csv, progress.png, checkpoints/, best_model.zip, vecnormalize.pkl, versions.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import platform
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
HELDOUT_SEEDS = list(range(900_000, 900_030))   # fixed held-out randomised episodes for the success curve


def versions():
    import gymnasium, mujoco, stable_baselines3, torch
    return {"python": sys.version.split()[0], "platform": platform.platform(), "mujoco": mujoco.__version__,
            "gymnasium": gymnasium.__version__, "stable_baselines3": stable_baselines3.__version__,
            "torch": torch.__version__, "numpy": np.__version__}


def run_policy_episodes(policy_fn, seeds, options_fn=None, randomize=True):
    """policy_fn(obs)->action. Returns list of episode_metrics dicts (one env in this process)."""
    from .rover_env import FrostRoverEnv
    env = FrostRoverEnv(randomize=randomize)
    out = []
    for s in seeds:
        opts = options_fn(s) if options_fn else {"deploy": bool(s % 2), "full_task": True}
        o, _ = env.reset(seed=s, options=opts)
        done = False
        R = 0.0
        while not done:
            o, r, te, tr, info = env.step(policy_fn(o))
            R += r
            done = te or tr
        m = dict(info["episode_metrics"]); m["reward"] = R; m["seed"] = s
        out.append(m)
    return out


def ppo_policy_fn(model, vecnorm):
    def f(o):
        on = vecnorm.normalize_obs(o[None]) if vecnorm is not None else o[None]
        a, _ = model.predict(on, deterministic=True)
        return a[0]
    return f


def bc_warm_start(model, venv, n_eps, seed, out):
    """Imitation warm start (cf. Open Duck's imitation reward, here as behaviour cloning): roll out the hand-written
    controller in the randomised env, keep successful episodes, fit the PPO actor mean by MSE, and seed the
    VecNormalize observation statistics from the demo data."""
    import torch
    from .controller import PController
    from .rover_env import FrostRoverEnv
    env = FrostRoverEnv(randomize=True); ctrl = PController()
    O, A, n_ok = [], [], 0
    for e in range(n_eps):
        o, _ = env.reset(seed=700_000 + seed * 10_000 + e); ctrl.reset(); done = False; eo, ea = [], []
        while not done:
            a = ctrl(o); eo.append(o); ea.append(a)
            o, r, te, tr, info = env.step(a); done = te or tr
        if info["episode_metrics"]["success"]:
            O += eo; A += ea; n_ok += 1
    O, A = np.array(O, np.float64), np.clip(np.array(A, np.float32), -1, 1)
    print(f"BC: {n_ok}/{n_eps} successful controller episodes, {len(O)} samples", flush=True)
    venv.obs_rms.mean = O.mean(0); venv.obs_rms.var = O.var(0) + 1e-6; venv.obs_rms.count = len(O)
    X = torch.tensor(np.clip((O - O.mean(0)) / np.sqrt(O.var(0) + 1e-6 + 1e-8), -10, 10), dtype=torch.float32)
    Y = torch.tensor(A)
    pol = model.policy
    opt = torch.optim.Adam(list(pol.mlp_extractor.policy_net.parameters()) + list(pol.action_net.parameters()), lr=1e-3)
    for ep in range(30):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 512):
            idx = perm[i:i + 512]
            lat = pol.mlp_extractor.forward_actor(pol.extract_features(X[idx], pol.pi_features_extractor))
            loss = ((pol.action_net(lat) - Y[idx]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        pol.log_std.fill_(-1.2)
    print(f"BC: final MSE {loss.item():.4f}", flush=True)
    (out / "bc.json").write_text(json.dumps({"episodes": n_eps, "successful": n_ok, "samples": len(O), "mse": loss.item()}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="ppo_s0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--minutes", type=float, default=35.0)
    ap.add_argument("--max-steps", type=int, default=30_000_000)
    ap.add_argument("--envs", type=int, default=5)
    ap.add_argument("--eval-every", type=int, default=300_000)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--bc", type=int, default=0, help="warm-start: behaviour-clone N controller episodes first")
    args = ap.parse_args()

    import torch
    torch.set_num_threads(args.threads)
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize
    from .rover_env import make_env

    out = ROOT / "runs" / args.run
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out / "versions.json").write_text(json.dumps(versions(), indent=1))
    venv = SubprocVecEnv([make_env(args.seed * 100 + i) for i in range(args.envs)])
    venv = VecMonitor(venv)
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)
    hp = dict(n_steps=1024, batch_size=1024, n_epochs=5, learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
              clip_range=0.2, ent_coef=0.003, vf_coef=0.5, max_grad_norm=0.5,
              policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128]), log_std_init=-0.5))
    (out / "hparams.json").write_text(json.dumps({**hp, "policy_kwargs": str(hp["policy_kwargs"]), "n_envs": args.envs,
                                                  "seed": args.seed}, indent=1))
    model = PPO("MlpPolicy", venv, seed=args.seed, verbose=0, device="cpu", **hp)
    if args.bc:
        bc_warm_start(model, venv, args.bc, args.seed, out)

    class Logger(BaseCallback):
        def __init__(self):
            super().__init__()
            self.t0 = time.time(); self.buf = []; self.next_eval = args.eval_every
            self.rows = []; self.best = -1.0; self.hist = []
            self.f = open(out / "progress.csv", "w", newline="")
            self.w = csv.writer(self.f)
            self.w.writerow(["timesteps", "wall_min", "ep_reward", "train_success", "train_energy_J", "train_crop_hit",
                             "heldout_success", "heldout_energy_J", "heldout_reward", "heldout_crop_hit"])
            self.stop = False

        def _on_step(self):
            for info in self.locals["infos"]:
                if "episode_metrics" in info:
                    em = dict(info["episode_metrics"]); em["reward"] = info.get("episode", {}).get("r", np.nan)
                    self.buf.append(em)
            if self.num_timesteps >= self.next_eval:
                self.next_eval += args.eval_every
                self._eval()
            if (time.time() - self.t0) / 60 > args.minutes:
                print("wall budget reached"); return False
            return not self.stop

        def _eval(self):
            ms = run_policy_episodes(ppo_policy_fn(self.model, venv), HELDOUT_SEEDS)
            hs = float(np.mean([m["success"] for m in ms]))
            b = self.buf[-300:] or [{"reward": np.nan, "success": np.nan, "energy_J": np.nan, "crop_hit": np.nan}]
            row = [self.num_timesteps, round((time.time() - self.t0) / 60, 2),
                   float(np.nanmean([m["reward"] for m in b])), float(np.nanmean([m["success"] for m in b])),
                   float(np.nanmean([m["energy_J"] for m in b])), float(np.nanmean([m["crop_hit"] for m in b])),
                   hs, float(np.mean([m["energy_J"] for m in ms])), float(np.mean([m["reward"] for m in ms])),
                   float(np.mean([m["crop_hit"] for m in ms]))]
            self.buf = []
            self.w.writerow([round(x, 4) if isinstance(x, float) else x for x in row]); self.f.flush()
            print("t=%d  %.1f min  train R %.1f succ %.2f | heldout succ %.2f E %.0f J crop %.2f" %
                  (row[0], row[1], row[2], row[3], hs, row[7], row[9]), flush=True)
            self.model.save(out / "checkpoints" / f"step_{self.num_timesteps}")
            venv.save(str(out / "checkpoints" / f"vecnorm_{self.num_timesteps}.pkl"))
            score = hs - 0.1 * row[9]
            if score >= self.best:
                self.best = score
                self.model.save(out / "best_model"); venv.save(str(out / "vecnormalize.pkl"))
            self.hist.append(hs)
            # plateau: no gain > 0.03 over the best of the last 5 evals, after reaching 0.6
            if len(self.hist) >= 8 and max(self.hist[-8:]) >= 0.6 and max(self.hist[-5:]) <= max(self.hist[:-5]) + 0.03:
                print("held-out success plateaued"); self.stop = True

    cb = Logger()
    model.learn(total_timesteps=args.max_steps, callback=cb)
    cb._eval()
    model.save(out / "final_model"); venv.save(str(out / "vecnormalize_final.pkl"))
    cb.f.close()
    venv.close()
    plot(out)


def plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    df = pd.read_csv(out / "progress.csv")
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    x = df.timesteps / 1e6
    ax[0].plot(x, df.ep_reward, color="#3b6ea8"); ax[0].set_title("training episode reward"); ax[0].set_xlabel("env steps (M)")
    ax[1].plot(x, df.train_success, label="training (randomised)", color="#3b6ea8")
    ax[1].plot(x, df.heldout_success, label="held-out 30 eps", color="#d9822b")
    ax[1].set_ylim(0, 1); ax[1].set_title("success rate"); ax[1].legend(frameon=False); ax[1].set_xlabel("env steps (M)")
    ax[2].plot(x, df.train_energy_J, color="#3b6ea8", label="training"); ax[2].plot(x, df.heldout_energy_J, color="#d9822b", label="held-out")
    ax[2].set_title("energy per episode (J)"); ax[2].set_xlabel("env steps (M)"); ax[2].legend(frameon=False)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(out / "progress.png", dpi=110)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--plot":
        plot(ROOT / "runs" / sys.argv[2])
    else:
        main()
