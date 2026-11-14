"""Surgical cart arm-1 reach, driven by the simulated culture (surgical/culture_ctrl.py)."""
from __future__ import annotations

import numpy as np

from ..registry import PY, RobotSpec, register_robot

PIP = "python3 -m venv .v && . .v/bin/activate && pip install -q mujoco==4.0.1 gymnasium==1.4.0 numpy"


def _train(a):
    # surgical/train_culture.py takes positional args: teach|frozen [episodes] [lr]
    return [PY, "surgical/train_culture.py", a.mode or "teach",
            str(a.episodes if a.episodes is not None else 60),
            str(a.lr if a.lr is not None else 0.05)]


def _eval(a):
    return [PY, "-m", "surgical.research_eval", "--seed", str(a.seed),
            "--episodes", str(a.episodes if a.episodes is not None else 3)]


def _fleet_job(lr: float, episodes: int) -> str:
    # same shape as sweep/jobs_surgical.txt
    d = f"surgical/results/lr{lr:g}"
    return (f"{PIP} && python surgical/train_culture.py teach {episodes} {lr:g} && mkdir -p {d} "
            f"&& cp surgical/results/curve_teach.json {d}/ && cp surgical/results/culture_teach.pkl {d}/ "
            f"2>/dev/null; true")


class SurgicalAdapter:
    """Mirrors surgical.culture_ctrl.CultureController (which is what actually runs in training).
    obs[0:3] = tip->target error (m), obs[4:12] = joint angles normalised to [-1, 1]."""

    sensory = list(range(0, 14))
    motor = list(range(32, 48))

    def encode(self, obs):
        from surgical.culture_ctrl import encode_levels
        # The real controller uses the arm-1 base frame (env.err_arm_frame()); obs is world frame.
        return encode_levels(np.asarray(obs[0:3]), (np.asarray(obs[4:12]) + 1) / 2)

    def decode(self, counts):
        from surgical.culture_ctrl import CultureController
        return CultureController.decode(None, counts)

    def teach(self, env, obs, action):
        from surgical.culture_ctrl import NEG_POOL, POS_POOL
        dq = (env.ix.jac(env.m, env.d).T @ env.err()) / env.scale  # Jacobian-transpose direction
        mx = float(np.abs(dq).max()) or 1.0
        ks = [k for k in range(len(dq)) if abs(dq[k]) >= 0.15 * mx]
        return {"good": [POS_POOL[k] if dq[k] > 0 else NEG_POOL[k] for k in ks],
                "bad": [NEG_POOL[k] if dq[k] > 0 else POS_POOL[k] for k in ks]}


register_robot(RobotSpec(
    name="surgical_reach",
    description="SimReady surgical cart, arm 1 reaches a tissue target (8 joints, MuJoCo)",
    env_module="surgical.reach_env:SurgicalReachEnv",
    controller="surgical.culture_ctrl:CultureController (paired-stimulation 'teach')",
    train_cmd=_train,
    eval_cmd=_eval,
    fleet_job=_fleet_job,
    adapter_factory=SurgicalAdapter,
    notes="train -> surgical/results/curve_<mode>.json + culture_<mode>.pkl; "
          "eval -> procedural scenarios (surgical/scenarios.py) + IK world model",
    defaults=dict(lr=0.05, episodes=60),
))
