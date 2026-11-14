"""Non-culture controllers: damped-least-squares IK (engineering reference) and random."""
from __future__ import annotations

import numpy as np


class DLSIK:
    """dq = J^T (J J^T + lam^2 I)^-1 (k e) at the measured pose; the commanded servo
    target becomes q + dq (anti-windup: expressed as a delta from the current target),
    returned in normalised action units."""
    name = "ik_dls"

    def __init__(self, lam=0.05, k=0.8):
        self.lam, self.k = lam, k

    def reset(self, env):
        pass

    def act(self, env, obs):
        J = env.ix.jac(env.m, env.d)
        e = env.err()
        dq = J.T @ np.linalg.solve(J @ J.T + self.lam ** 2 * np.eye(3), self.k * e)
        return np.clip((env.q() + dq - env.qref) / env.scale, -1, 1)


class RandomPolicy:
    name = "random"

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def reset(self, env):
        pass

    def act(self, env, obs):
        return self.rng.uniform(-1, 1, env.action_space.shape)


def run_episode(env, ctrl, target, seed, record=None):
    obs, _ = env.reset(seed=seed, options={"target": target})
    ctrl.reset(env)
    info = {}
    for t in range(200):
        a = ctrl.act(env, obs)
        obs, r, term, trunc, info = env.step(a)
        if record is not None:
            record(env, a, info)
        if term or trunc:
            break
    return dict(success=bool(info["success"]), final_err_mm=1000 * info["dist"],
                reach_time_s=(info["reached_step"] * 0.05 if info["reached_step"] else None),
                limit_hits=info["limit_hits"], collisions=info["collisions"])
