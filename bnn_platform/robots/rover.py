"""Frost-guard rover: the culture decides DEPLOY/HOLD each night, the MuJoCo rover executes it
(sim/night_loop.py)."""
from __future__ import annotations

import numpy as np

from ..registry import PY, RobotSpec, register_robot


def _train(a):
    # sim.night_loop trains (and caches) a culture_teach Culture on 100 samples, then runs a season.
    # It has no learning-rate flag; --episodes maps to --nights.
    cmd = [PY, "-m", "sim.night_loop", "--run", a.run or "ppo_s0", "--seed", str(a.seed)]
    if a.episodes is not None:
        cmd += ["--nights", str(a.episodes)]
    return cmd


def _eval(a):
    return [PY, "-m", "sim.eval", "--run", a.run or "ppo_s0"]


class RoverAdapter:
    """Mirrors neurorover/interface/codec.py: electrodes 0-13 sensory (0-4 pulses per feature),
    40-47 vote DEPLOY, 48-55 vote HOLD; 'teach' stimulates the correct vote region."""

    sensory = list(range(0, 14))
    motor = list(range(40, 56))

    def encode(self, obs):
        return np.clip(np.round(4 * np.asarray(obs, float)[:14]), 0, 4).astype(int)

    def decode(self, counts):
        return int(counts[40:48].sum() > counts[48:56].sum())  # 1 = DEPLOY

    def teach(self, env, obs, action):
        froze = bool(getattr(env, "label", 0))
        deploy, hold = list(range(40, 48)), list(range(48, 56))
        return {"good": deploy if froze else hold, "bad": hold if froze else deploy}


register_robot(RobotSpec(
    name="rover",
    description="Frost-guard rover: culture DEPLOY/HOLD decision + PPO-driven MuJoCo rover season",
    env_module="sim.rover_env:FrostRoverEnv",
    controller="neurorover.culture.core:Culture (rule='teach', via sim.night_loop.trained_culture)",
    train_cmd=_train,
    eval_cmd=_eval,
    fleet_job=None,
    adapter_factory=RoverAdapter,
    notes="needs a PPO body policy first (python -m sim.train_ppo --run ppo_s0); "
          "--lr is ignored (night_loop has no lr flag)",
))
