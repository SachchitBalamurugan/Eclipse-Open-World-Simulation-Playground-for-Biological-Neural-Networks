"""Gymnasium env: arm 1 of the SimReady surgical cart reaches a tissue target.

obs    = [tip->target (3, m, world), |tip->target| (1), q (8, normalised to [-1,1]),
          qdot (8, scaled)]                                            -> 20-d
action = 8 joint-position deltas in [-1,1] for CONTROLLED joints, scaled to
         0.03 rad (hinge) / 0.006 m (slide) per 50 ms control step, low-pass
         filtered (a_f = 0.5 a_f + 0.5 a), added to the position-servo targets.
reward = -10*dist - 0.01*|a|^2 - 0.1*(#joints at a limit) - 0.5*(#arm-1 contacts
         with other arms / cart) + 10 on success (tip within 1 cm for 0.5 s).
episode: 4 s = 80 control steps (25 physics steps of 2 ms each).
DR:    arm-1 link masses x U(0.8,1.2), all joint damping x U(0.7,1.3), target
       position noise N(0, 3 mm) per axis at reset.
"""
from __future__ import annotations

import mujoco
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover
    gym = None

from .arm import ArmIndex, HOME_Q, load_model, make_targets

CTRL_DT = 0.05
SUBSTEPS = 25
EP_STEPS = 80
SUCCESS_R = 0.01
HOLD_STEPS = 10           # 0.5 s


class SurgicalReachEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, seed: int = 0, randomize: bool = True, target_pool: np.ndarray | None = None,
                 target_noise: float = 0.003):
        super().__init__()
        self.m = load_model()
        self.d = mujoco.MjData(self.m)
        self.ix = ArmIndex(self.m)
        self.rng = np.random.default_rng(seed)
        self.randomize = randomize
        self.target_noise = target_noise
        self.pool = target_pool if target_pool is not None else make_targets(self.m, 2000, seed=10_000 + seed)
        self.mass0 = self.m.body_mass.copy()
        self.inertia0 = self.m.body_inertia.copy()
        self.damp0 = self.m.dof_damping.copy()
        self.scale = np.where(self.ix.is_slide, 0.006, 0.03)
        self.mocap = self.m.body("target").mocapid[0]
        self.hand = self.m.body("handBase_01").id
        n = len(self.ix.jid)
        self.observation_space = spaces.Box(-np.inf, np.inf, (4 + 2 * n,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (n,), np.float32)

    # ---------------- helpers ----------------
    def err(self) -> np.ndarray:
        return self.target - self.ix.tip(self.d)

    def err_arm_frame(self) -> np.ndarray:
        """Tip->target error in the arm-1 base (handBase_01) frame."""
        R = self.d.xmat[self.hand].reshape(3, 3)
        return R.T @ self.err()

    def q(self):
        return self.d.qpos[self.ix.qadr].copy()

    def _obs(self):
        e = self.err()
        q = self.q()
        qn = 2 * (q - self.ix.lo) / (self.ix.hi - self.ix.lo) - 1
        qd = self.d.qvel[self.ix.dadr] * 0.5
        return np.concatenate([e, [np.linalg.norm(e)], qn, qd]).astype(np.float32)

    # ---------------- gym API ----------------
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        m = self.m
        m.body_mass[:] = self.mass0
        m.body_inertia[:] = self.inertia0
        m.dof_damping[:] = self.damp0
        if self.randomize:
            for b in self.ix.arm_bodies:
                s = self.rng.uniform(0.8, 1.2)
                m.body_mass[b] = self.mass0[b] * s
                m.body_inertia[b] = self.inertia0[b] * s
            m.dof_damping[:] = self.damp0 * self.rng.uniform(0.7, 1.3, m.nv)
        mujoco.mj_resetData(m, self.d)
        self.d.qpos[self.ix.qadr] = HOME_Q
        self.d.ctrl[:] = [self.d.qpos[m.jnt_qposadr[m.actuator_trnid[a, 0]]] for a in range(m.nu)]
        mujoco.mj_forward(m, self.d)
        opt = options or {}
        if "target" in opt:
            tgt = np.asarray(opt["target"], float)
        else:
            tgt = self.pool[self.rng.integers(len(self.pool))]
        if self.randomize and opt.get("noise", True):
            tgt = tgt + self.rng.normal(0, self.target_noise, 3)
        self.target = tgt
        self.d.mocap_pos[self.mocap] = tgt
        self.qref = self.q()
        self.a_f = np.zeros(len(self.ix.jid))
        self.t = 0
        self.hold = 0
        self.info_acc = dict(limit_hits=0, collisions=0, reached_step=None)
        return self._obs(), {}

    def step(self, action):
        a = np.clip(np.asarray(action, float), -1, 1)
        self.a_f = 0.5 * self.a_f + 0.5 * a
        self.qref = np.clip(self.qref + self.a_f * self.scale, self.ix.lo, self.ix.hi)
        self.d.ctrl[self.ix.act] = self.qref
        for _ in range(SUBSTEPS):
            mujoco.mj_step(self.m, self.d)
        self.t += 1
        e = self.err()
        dist = float(np.linalg.norm(e))
        q = self.q()
        tol = 0.005 * (self.ix.hi - self.ix.lo)
        at_lim = int(np.sum((q <= self.ix.lo + tol) | (q >= self.ix.hi - tol)))
        ncol = self.ix.collisions(self.m, self.d)
        self.info_acc["limit_hits"] += at_lim
        self.info_acc["collisions"] += ncol
        self.hold = self.hold + 1 if dist < SUCCESS_R else 0
        success = self.hold >= HOLD_STEPS
        if dist < SUCCESS_R and self.info_acc["reached_step"] is None:
            self.info_acc["reached_step"] = self.t
        r = -10 * dist - 0.01 * float(a @ a) - 0.1 * at_lim - 0.5 * ncol + (10.0 if success else 0.0)
        bad = not np.isfinite(self.d.qpos).all()
        terminated = success or bad
        truncated = self.t >= EP_STEPS
        info = dict(dist=dist, success=success, **self.info_acc)
        return self._obs(), r, terminated, truncated, info
