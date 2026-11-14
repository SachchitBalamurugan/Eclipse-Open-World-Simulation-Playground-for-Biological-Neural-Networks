"""PPO baseline (stable-baselines3, CPU). 4 envs in one process (DummyVecEnv), time-boxed."""
import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402

from surgical.reach_env import SurgicalReachEnv  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent / "results"
BUDGET_S = float(sys.argv[1]) if len(sys.argv) > 1 else 20 * 60
torch.set_num_threads(1)


class TimeBox(BaseCallback):
    def __init__(self, budget):
        super().__init__()
        self.t0, self.budget = time.time(), budget

    def _on_step(self):
        return time.time() - self.t0 < self.budget


def main():
    venv = DummyVecEnv([lambda i=i: Monitor(SurgicalReachEnv(seed=i)) for i in range(4)])
    model = PPO("MlpPolicy", venv, n_steps=512, batch_size=256, learning_rate=3e-4, gamma=0.98,
                ent_coef=0.0, policy_kwargs=dict(net_arch=[128, 128]), verbose=1, seed=0)
    t0 = time.time()
    model.learn(total_timesteps=10_000_000, callback=TimeBox(BUDGET_S))
    model.save(OUT / "ppo_reach")
    (OUT / "ppo_train_info.txt").write_text(
        f"timesteps={model.num_timesteps} wall_s={time.time()-t0:.0f} envs=4 (DummyVecEnv, 1 process)\n")
    print("done", model.num_timesteps)


if __name__ == "__main__":
    main()
