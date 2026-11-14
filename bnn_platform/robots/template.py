"""TEMPLATE: add a new robot to the BNN platform in one file.

1. Copy this file to bnn_platform/robots/<your_robot>.py and rename MyRobot*.
2. Fill in the three adapter methods (encode / decode / teach) for your env.
3. Point train_cmd / eval_cmd at your scripts (or use the generic loop below as-is).
4. Add `from . import <your_robot>` to bnn_platform/robots/__init__.py.
5. Check:  python -m bnn_platform robots
           python -m bnn_platform train --robot my_robot --episodes 20 --lr 0.02
           python -m bnn_platform fleet sweep --robot my_robot --lrs 0.02 0.05 --dry-run

This file is NOT imported by default, so it does not show up in `robots` until you copy it.
Run the generic loop directly with:  python -m bnn_platform.robots.template 5
"""
from __future__ import annotations

import sys

import numpy as np

from ..registry import PY, RobotSpec, register_robot, run_trial

# ---- electrode map (64-electrode MEA, zero-based) ----------------------------------------
SENSORY = list(range(0, 8))      # one electrode per observation feature
POS_POOL = [32, 34]              # action dim k increases when POS_POOL[k] out-fires NEG_POOL[k]
NEG_POOL = [33, 35]


class MyRobotAdapter:
    sensory = SENSORY
    motor = POS_POOL + NEG_POOL

    def encode(self, obs: np.ndarray) -> np.ndarray:
        """Observation -> 0..4 pulses per sensory electrode. Normalise each feature to [0, 1]."""
        levels = np.zeros(64, int)
        x = np.clip(np.asarray(obs, float)[: len(SENSORY)], 0.0, 1.0)
        levels[SENSORY[: len(x)]] = np.round(4 * x).astype(int)
        return levels

    def decode(self, counts: np.ndarray) -> np.ndarray:
        """Spike counts -> action in [-1, 1] per dim (push-pull motor pools with a deadzone)."""
        diff = counts[POS_POOL].astype(float) - counts[NEG_POOL]
        diff = np.sign(diff) * np.maximum(np.abs(diff) - 2.0, 0)
        return np.clip(diff / 8.0, -1, 1)

    def teach(self, env, obs, action) -> dict:
        """Teaching signal. Option A (paired stimulation, the repo's best rule): name the motor
        pools that *should* have fired. Option B: return {"reward": r} for R-STDP / FEP."""
        target = np.sign(-np.asarray(obs, float)[: len(POS_POOL)])  # e.g. move to reduce error
        good = [POS_POOL[k] if t > 0 else NEG_POOL[k] for k, t in enumerate(target) if t != 0]
        bad = [NEG_POOL[k] if t > 0 else POS_POOL[k] for k, t in enumerate(target) if t != 0]
        return {"good": good, "bad": bad}


def apply_teaching(culture, levels, signal: dict) -> None:
    """Paired stimulation: replay the sensory pattern, stimulate 'good' pools 3 ms after each
    volley (LTP) and 'bad' pools 3 ms before (LTD). Reward signals are left to the culture rule."""
    if "good" not in signal:
        return
    sens = [ch for ch, n in enumerate(levels) if n > 0]
    sched: dict = {}
    for k in range(4):
        t = 10 + 20 * k
        sched.setdefault(t - 3, []).extend(signal.get("bad", []))
        sched.setdefault(t, []).extend(sens)
        sched.setdefault(t + 3, []).extend(signal["good"])
    for t in range(90):
        for ch in sched.get(t, []):
            culture.stim(ch, 1.0)
        culture.step()


def train_loop(episodes: int = 20, lr: float = 0.02, seed: int = 0, steps: int = 20) -> None:
    """Generic closed loop: env -> encode -> culture -> decode -> env, plus teaching.
    Replace the toy env with your own (anything with reset() -> obs and step(a) -> obs, r, done)."""
    from neurorover.culture.core import Culture

    culture = Culture(seed=seed, rule="teach", lr=lr)
    ad = MyRobotAdapter()
    rng = np.random.default_rng(seed)
    for ep in range(episodes):
        obs = rng.uniform(-1, 1, 2)   # toy env: 2-d error that the action should drive to zero
        for _ in range(steps):
            levels = ad.encode(np.abs(np.r_[obs, obs]))
            a = ad.decode(run_trial(culture, levels))
            apply_teaching(culture, levels, ad.teach(None, obs, a))
            obs = obs + 0.1 * a
        print(f"episode {ep} final |err| {np.abs(obs).sum():.3f}", flush=True)


def _train(a):
    return [PY, "-m", "bnn_platform.robots.template", str(a.episodes or 20), str(a.lr or 0.02)]


def _eval(a):
    return [PY, "-m", "bnn_platform.robots.template", "3", "0"]


register_robot(RobotSpec(
    name="my_robot",
    description="TEMPLATE robot (toy 2-d error-nulling task)",
    env_module="bnn_platform.robots.template (toy env inside train_loop)",
    controller="bnn_platform.robots.template:MyRobotAdapter",
    train_cmd=_train,
    eval_cmd=_eval,
    # one shell line per VM, run from the repo root on a fresh Ubuntu box
    fleet_job=lambda lr, ep: (f"python3 -m venv .v && . .v/bin/activate && pip install -q numpy "
                              f"&& python -m bnn_platform.robots.template {ep} {lr:g}"),
    adapter_factory=MyRobotAdapter,
))


if __name__ == "__main__":
    train_loop(episodes=int(sys.argv[1]) if len(sys.argv) > 1 else 5,
               lr=float(sys.argv[2]) if len(sys.argv) > 2 else 0.02)
