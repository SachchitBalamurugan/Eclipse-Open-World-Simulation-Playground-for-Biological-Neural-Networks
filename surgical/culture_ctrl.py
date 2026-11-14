"""Simulated-culture reach controller for arm 1 (neurorover.culture.core.Culture).

Electrode map (64-electrode MEA, zero-based channels):
  0-5    sensory: tip->target error in the arm-1 base frame, (+x,-x,+y,-y,+z,-z);
         pulse count 0-4 from |e_axis| with thresholds 2 / 8 / 25 / 60 mm
  6-13   sensory context: normalised joint angle of each controlled joint, 0-2 pulses
  32-47  motor pools: joint k (k = 0..7 over CONTROLLED) -> + pool 32+2k, - pool 33+2k
  48-55  unused (spare motor row)
Long-range plastic projection: sensory 0-15 (excitatory) -> motor 32-47, p = 0.15,
w ~ U(0.5, 1.5); this replaces the neurorover sensory -> 40-55 projection.

Control step (60 ms of culture time):
  0-40 ms  sensory pulses (10 ms spacing, as neurorover codec.encode)
  0-50 ms  spike counts on the motor pools
  delta_k = clip((n+ - n-) deadzoned by DEADZONE, / GAIN, -1, 1)

Teaching (rule "teach", after every control step, ~85 ms): the joint direction that
would reduce the error is dq* = J^T e (Jacobian transpose at the current pose).
For each joint with |dq*_k| above 15 % of max|dq*|: replay the sensory pattern
(4 volleys, 20 ms apart); 3 ms after each volley stimulate the correct-direction
pool (pre->post, LTP), 3 ms before it the opposite pool (post->pre, LTD).
Exactly the neurorover codec rule "teach", per joint instead of per class.
"""
from __future__ import annotations

import numpy as np

from neurorover.culture.core import Culture

ERR_CH = list(range(0, 6))
CTX_CH = list(range(6, 14))
SENS_ALL = list(range(0, 16))
N_J = 8
POS_POOL = [32 + 2 * k for k in range(N_J)]
NEG_POOL = [33 + 2 * k for k in range(N_J)]
MOTOR = POS_POOL + NEG_POOL
THRESH_MM = [2, 8, 25, 60]
STIM_MS, RESP_MS, TRIAL_MS = 40, 50, 60
DEADZONE, GAIN = 2.0, 8.0


def make_culture(seed: int, rule: str, lr: float = 0.02) -> Culture:
    c = Culture(seed=seed, rule=rule, lr=lr)
    # remove the neurorover sensory->(40-55) projection ...
    c.W[c.plastic] = 0.0
    c.mask[c.plastic] = False
    c.plastic[:] = False
    # ... and wire sensory 0-15 -> motor pools 32-47 instead
    sens = np.isin(c.ch, SENS_ALL) & c.exc
    mot = np.isin(c.ch, MOTOR)
    lr_mask = sens[:, None] & mot[None, :] & (c.rng.random((c.N, c.N)) < 0.15)
    c.mask |= lr_mask
    c.plastic[lr_mask] = True
    c.W[lr_mask] = c.rng.uniform(0.5, 1.5, lr_mask.sum())
    return c


def encode_levels(err_arm: np.ndarray, qn01: np.ndarray) -> np.ndarray:
    """-> (14,) pulse counts for channels 0-13."""
    lv = np.zeros(14, int)
    for a in range(3):
        mm = abs(err_arm[a]) * 1000
        L = int(np.sum(mm >= np.array(THRESH_MM)))
        lv[2 * a + (0 if err_arm[a] >= 0 else 1)] = L
    lv[6:14] = np.clip(np.round(2 * qn01), 0, 2).astype(int)
    return lv


class CultureController:
    def __init__(self, seed: int = 0, teach: bool = True, lr: float = 0.02):
        self.teach = teach
        self.culture = make_culture(seed, "teach" if teach else "frozen", lr)
        self.learning = teach
        self.n_teach = 0
        self.name = "culture_teach" if teach else "culture_frozen"
        self.last = {}

    def reset(self, env):
        pass

    def _run_trial(self, levels):
        c = self.culture
        sched = {}
        for ch, L in enumerate(levels):
            for k in range(int(L)):
                sched.setdefault(k * 10, []).append(ch)
        counts = np.zeros(64, int)
        stims = []
        for t in range(TRIAL_MS):
            for ch in sched.get(t, []):
                c.stim(ch, 1.0)
                stims.append((t, ch))
            f = c.step()
            if t < RESP_MS and f.size:
                counts += np.bincount(c.ch[f], minlength=64)
        return counts, stims

    def decode(self, counts):
        diff = counts[POS_POOL].astype(float) - counts[NEG_POOL]
        diff = np.sign(diff) * np.maximum(np.abs(diff) - DEADZONE, 0)
        return np.clip(diff / GAIN, -1, 1)

    def act(self, env, obs):
        qn01 = (env.q() - env.ix.lo) / (env.ix.hi - env.ix.lo)
        e_arm = env.err_arm_frame()
        levels = encode_levels(e_arm, qn01)
        # plasticity only while teaching
        self.culture.rule = "teach" if (self.teach and self.learning) else "frozen"
        counts, stims = self._run_trial(levels)
        a = self.decode(counts)
        teach_stims = []
        if self.teach and self.learning:
            J = env.ix.jac(env.m, env.d)
            dq = (J.T @ env.err()) / env.scale  # in action units
            teach_stims = self._teach(levels, dq)
            self.n_teach += 1
        self.last = dict(levels=levels, counts=counts, stims=stims, teach_stims=teach_stims, action=a)
        return a

    def _teach(self, levels, dq):
        c = self.culture
        mx = np.abs(dq).max()
        if mx < 1e-9:
            return []
        good, bad = [], []
        for k in range(N_J):
            if abs(dq[k]) < 0.15 * mx:
                continue
            good.append(POS_POOL[k] if dq[k] > 0 else NEG_POOL[k])
            bad.append(NEG_POOL[k] if dq[k] > 0 else POS_POOL[k])
        sched = {}
        for k in range(4):
            t = 10 + k * 20
            sens = [ch for ch in ERR_CH if levels[ch] > k]
            if not sens:
                continue
            sched.setdefault(t - 3, []).extend(bad)
            sched.setdefault(t, []).extend(sens)
            sched.setdefault(t + 3, []).extend(good)
        out = []
        for t in range(85):
            for ch in sched.get(t, []):
                c.stim(ch, 1.0)
                out.append((t, ch))
            c.step()
        return out
