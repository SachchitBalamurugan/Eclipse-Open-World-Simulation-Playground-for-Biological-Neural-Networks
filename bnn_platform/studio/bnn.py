"""Generic culture (BNN) reach controller for any robot -- the surgical controller
(surgical/culture_ctrl.py) generalised from 8 fixed joints to 1..12 joints of any chain.

Simulated biology only: neurorover.culture.core.Culture (Izhikevich, 64-electrode MEA).

Electrode map (zero-based channels), n = number of controlled joints (<= 12):
  0-5          sensory: EE->target error in the chain base frame, (+x,-x,+y,-y,+z,-z),
               0-4 pulses from |e_axis| with robot-scaled thresholds (surgical: 2/8/25/60 mm)
  6..6+n-1     sensory context: normalised joint angle, 0-2 pulses
  32+2k/33+2k  motor pools of joint k: + / - direction (k < 12 -> channels 32..55)
Plastic projection: excitatory sensory neurons (0..max(15, 5+n)) -> motor pools, p = 0.15.

Control step (60 ms culture time): stimulate, count motor spikes for 50 ms,
  a_k = clip(deadzone(n+_k - n-_k, 2) / 8, -1, 1)  -> joint position delta a_k * scale_k.
Rule 'teach' (after every step, 85 ms): dq* = J^T e; for joints with |dq*_k| >= 15 % max,
replay the error pattern and pair it with the correct-direction pool 3 ms later (LTP) and the
opposite pool 3 ms earlier (LTD). Rule 'frozen': same culture, no plasticity, no teaching.
"""
from __future__ import annotations

import numpy as np

from neurorover.culture.core import Culture

ERR_CH = list(range(0, 6))
STIM_MS, RESP_MS, TRIAL_MS = 40, 50, 60
DEADZONE, GAIN = 2.0, 8.0
MAX_J = 12


class BNNReachController:
    def __init__(self, n_joints: int, seed: int = 0, rule: str = "teach", lr: float = 0.02,
                 npe: int = 16, thresh_mm=(2, 8, 25, 60)):
        if not 1 <= n_joints <= MAX_J:
            raise ValueError(f"1..{MAX_J} joints supported, got {n_joints}")
        self.nj = n_joints
        self.rule = rule
        self.teach = rule == "teach"
        self.learning = self.teach
        self.thresh = np.asarray(thresh_mm, float)
        self.ctx_ch = list(range(6, 6 + n_joints))
        self.pos_pool = [32 + 2 * k for k in range(n_joints)]
        self.neg_pool = [33 + 2 * k for k in range(n_joints)]
        self.n_teach = 0
        c = Culture(seed=seed, npe=npe, rule="teach" if self.teach else "frozen", lr=lr)
        # same rewiring as surgical.culture_ctrl.make_culture: drop the neurorover
        # sensory->(40-55) projection, wire sensory -> motor pools instead
        c.W[c.plastic] = 0.0
        c.mask[c.plastic] = False
        c.plastic[:] = False
        sens = np.isin(c.ch, list(range(0, max(16, 6 + n_joints)))) & c.exc
        mot = np.isin(c.ch, self.pos_pool + self.neg_pool)
        lr_mask = sens[:, None] & mot[None, :] & (c.rng.random((c.N, c.N)) < 0.15)
        c.mask |= lr_mask
        c.plastic[lr_mask] = True
        c.W[lr_mask] = c.rng.uniform(0.5, 1.5, lr_mask.sum())
        self.culture = c

    # ---------------- encoding / decoding ----------------
    def encode(self, err_base: np.ndarray, qn01: np.ndarray) -> np.ndarray:
        lv = np.zeros(64, int)
        for a in range(3):
            mm = abs(err_base[a]) * 1000
            lv[2 * a + (0 if err_base[a] >= 0 else 1)] = int(np.sum(mm >= self.thresh))
        lv[self.ctx_ch] = np.clip(np.round(2 * qn01), 0, 2).astype(int)
        return lv

    def decode(self, counts: np.ndarray) -> np.ndarray:
        diff = counts[self.pos_pool].astype(float) - counts[self.neg_pool]
        diff = np.sign(diff) * np.maximum(np.abs(diff) - DEADZONE, 0)
        return np.clip(diff / GAIN, -1, 1)

    def _run_trial(self, levels):
        c = self.culture
        sched: dict = {}
        for ch in np.flatnonzero(levels):
            for k in range(int(levels[ch])):
                sched.setdefault(k * 10, []).append(int(ch))
        counts = np.zeros(64, int)
        for t in range(TRIAL_MS):
            for ch in sched.get(t, []):
                c.stim(ch, 1.0)
            f = c.step()
            if t < RESP_MS and f.size:
                counts += np.bincount(c.ch[f], minlength=64)
        return counts

    # ---------------- control ----------------
    def act(self, err_base, qn01, dq_star=None):
        """-> (action in [-1,1]^n, (64,) spike counts). dq_star = J^T e in action units (teach)."""
        levels = self.encode(err_base, qn01)
        self.culture.rule = "teach" if (self.teach and self.learning) else "frozen"
        counts = self._run_trial(levels)
        a = self.decode(counts)
        if self.teach and self.learning and dq_star is not None:
            self._teach(levels, dq_star)
            self.n_teach += 1
        return a, counts

    def _teach(self, levels, dq):
        c = self.culture
        mx = float(np.abs(dq).max())
        if mx < 1e-9:
            return
        good, bad = [], []
        for k in range(self.nj):
            if abs(dq[k]) < 0.15 * mx:
                continue
            good.append(self.pos_pool[k] if dq[k] > 0 else self.neg_pool[k])
            bad.append(self.neg_pool[k] if dq[k] > 0 else self.pos_pool[k])
        sched: dict = {}
        for k in range(4):
            t = 10 + k * 20
            sens = [ch for ch in ERR_CH if levels[ch] > k]
            if not sens:
                continue
            sched.setdefault(t - 3, []).extend(bad)
            sched.setdefault(t, []).extend(sens)
            sched.setdefault(t + 3, []).extend(good)
        for t in range(85):
            for ch in sched.get(t, []):
                c.stim(ch, 1.0)
            c.step()


class KinematicReach:
    """Reach task on a RobotModel. The joint-position servo is treated as ideal (kinematic:
    qpos follows the commanded reference), so any imported robot -- fixed or floating base,
    with or without tuned gains -- behaves the same and nothing falls over.

    action -> low-pass a_f = 0.5 a_f + 0.5 a -> qref += a_f * scale (clipped to limits).
    success: EE within success_r of the target for HOLD consecutive steps (50 ms each)."""

    HOLD = 5

    def __init__(self, rm, steps: int = 60):
        self.rm = rm
        self.steps = steps
        self.d = rm.d

    def reset(self, target):
        rm = self.rm
        self.target = np.asarray(target, float)
        self.qref = rm.home.copy()
        self.a_f = np.zeros(len(rm.jid))
        self.t = 0
        self.hold = 0
        rm.set_q(self.qref, self.d, target=self.target)
        self.dist = float(np.linalg.norm(self.err()))
        self.success = False

    def tip(self):
        return self.d.site_xpos[self.rm.site].copy()

    def err(self):
        return self.target - self.tip()

    def err_base(self):
        R = self.d.xmat[self.rm.base_body].reshape(3, 3)
        return R.T @ self.err()

    def qn01(self):
        rm = self.rm
        return (self.qref - rm.lo) / (rm.hi - rm.lo)

    def dq_star(self):
        J = self.rm.jac(self.d)
        return (J.T @ self.err()) / self.rm.scale

    def step(self, a):
        rm = self.rm
        a = np.clip(np.asarray(a, float), -1, 1)
        self.a_f = 0.5 * self.a_f + 0.5 * a
        self.qref = np.clip(self.qref + self.a_f * rm.scale, rm.lo, rm.hi)
        rm.set_q(self.qref, self.d, target=self.target)
        self.t += 1
        self.dist = float(np.linalg.norm(self.err()))
        self.hold = self.hold + 1 if self.dist < rm.success_r else 0
        self.success = self.hold >= self.HOLD
        return self.success or self.t >= self.steps

    def body_poses(self):
        d = self.d
        return np.round(np.concatenate([d.xpos, d.xquat], 1), 5).tolist()
