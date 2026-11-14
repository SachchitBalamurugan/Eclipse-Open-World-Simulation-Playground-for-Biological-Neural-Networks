"""Simulated cortical culture on a 64-electrode MEA (numpy, 1 ms steps).

This is a *model* of a dissociated culture, not a culture. It exists so the
closed-loop pipeline (CL API -> culture -> decode -> robot -> feedback) can be
built and tested without wetware, and as the in-silico control arm.

Layout: 8x8 electrode grid, NPE neurons per electrode (Izhikevich RS excitatory /
FS inhibitory, 80/20). Local distance-dependent recurrent connectivity plus a
sparse long-range projection from the sensory rows to the motor rows, which is
the substrate the reward-modulated rule can shape.
"""
from __future__ import annotations

import numpy as np

GRID = 8
N_CH = GRID * GRID

# Electrode roles (zero-based CL channels)
SENSORY = list(range(0, 16))      # rows 0-1: observation encoding
MOTOR_DEPLOY = list(range(40, 48))  # row 5
MOTOR_HOLD = list(range(48, 56))    # row 6
REWARD = list(range(60, 64))      # row 7 right half: reward/"dopamine" stim (R-STDP only)
FEEDBACK = list(range(56, 60))    # row 7 left half: DishBrain-style feedback stim


class Culture:
    def __init__(self, seed: int = 0, npe: int = 16, rule: str = "frozen",
                 noise: float = 4.0, w_exc: float = 2.0, w_inh: float = -6.0,
                 stim_gain: float = 30.0, lr: float = 0.02):
        self.rng = np.random.default_rng(seed)
        self.npe = npe
        self.N = N_CH * npe
        self.rule = rule
        self.noise = noise
        self.stim_gain = stim_gain
        self.lr = lr
        N = self.N
        self.ch = np.repeat(np.arange(N_CH), npe)          # electrode of each neuron
        exc = self.rng.random(N) < 0.8
        self.exc = exc
        r = self.rng.random(N)
        # Izhikevich parameters (RS for exc, FS for inh), with heterogeneity
        self.a = np.where(exc, 0.02, 0.02 + 0.08 * r)
        self.b = np.where(exc, 0.2, 0.25 - 0.05 * r)
        self.c = np.where(exc, -65 + 15 * r**2, -65.0)
        self.d = np.where(exc, 8 - 6 * r**2, 2.0)
        self.v = np.full(N, -65.0)
        self.u = self.b * self.v

        # Connectivity: distance-dependent local + sensory->motor long-range
        xy = np.stack([self.ch % GRID, self.ch // GRID], 1).astype(float)
        xy += self.rng.random((N, 2)) * 0.9
        dist = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2)
        p = 0.25 * np.exp(-dist / 1.2)
        mask = self.rng.random((N, N)) < p
        np.fill_diagonal(mask, False)
        sens = np.isin(self.ch, SENSORY) & exc
        motor = np.isin(self.ch, MOTOR_DEPLOY + MOTOR_HOLD)
        self.plastic = np.zeros((N, N), bool)
        lr_mask = (sens[:, None] & motor[None, :]) & (self.rng.random((N, N)) < 0.15)
        mask |= lr_mask
        self.plastic[lr_mask] = True
        W = np.where(mask, 1.0, 0.0) * self.rng.uniform(0.5, 1.0, (N, N))
        W[exc] *= w_exc
        W[~exc] *= -abs(w_inh)
        # long-range projections start weak and unbiased
        W[lr_mask] = self.rng.uniform(0.5, 1.5, lr_mask.sum())
        self.W = W
        self.mask = mask
        self.w_max = 6.0
        # STDP traces / eligibility
        self.x_pre = np.zeros(N)
        self.x_post = np.zeros(N)
        self.elig = np.zeros((N, N), np.float32) if rule == "rstdp" else None
        self.I_syn = np.zeros(N)
        self.pending = np.zeros(N)  # stim current to inject next step
        self.t_ms = 0

    # ---------------- stimulation ----------------
    def stim(self, channel: int, current_uA: float = 1.0):
        """Inject a biphasic-pulse-equivalent current near an electrode."""
        if channel in REWARD:
            # 60,61 = reward, 62,63 = punishment (modelling assumption, see codec.feedback)
            self.reward((1.0 if channel in REWARD[:2] else -1.0) * abs(current_uA))
            return
        sel = self.ch == channel
        self.pending[sel] += self.stim_gain * abs(current_uA)

    def reward(self, r: float):
        """Three-factor update: dW = lr * r * eligibility (R-STDP only)."""
        if self.rule != "rstdp" or self.elig is None:
            return
        dW = self.lr * r * self.elig
        self.W[self.plastic] = np.clip(self.W[self.plastic] + dW[self.plastic], 0.0, self.w_max)

    # ---------------- dynamics ----------------
    def step(self) -> np.ndarray:
        """Advance 1 ms. Returns indices of neurons that spiked."""
        I = self.I_syn + self.pending + self.noise * self.rng.standard_normal(self.N)
        self.pending[:] = 0.0
        v, u = self.v, self.u
        for _ in range(2):  # two 0.5 ms half steps for stability
            v += 0.5 * (0.04 * v * v + 5 * v + 140 - u + I)
        u += self.a * (self.b * v - u)
        fired = np.flatnonzero(v >= 30)
        v[fired] = self.c[fired]
        u[fired] += self.d[fired]
        np.clip(v, -90, 30, out=v)
        # synaptic current (exponential decay ~ 5 ms)
        self.I_syn *= 0.8
        if fired.size:
            self.I_syn += self.W[fired].sum(0)
        if self.rule in ("stdp", "fep", "rstdp", "teach"):
            self._plasticity(fired)
        self.t_ms += 1
        return fired

    def _plasticity(self, fired: np.ndarray):
        self.x_pre *= 0.95   # tau ~ 20 ms
        self.x_post *= 0.95
        if fired.size == 0:
            return
        a_plus, a_minus = 0.01, 0.012
        if self.rule == "teach":
            # STDP only on the sensory->motor projection; step size scales with lr
            k = self.lr / 0.02
            P = self.plastic[:, fired]
            if P.any():
                Wc = self.W[:, fired] + k * a_plus * self.x_pre[:, None] * P
                self.W[:, fired] = np.where(P, np.clip(Wc, 0.0, self.w_max), Wc)
            Pr = self.plastic[fired, :]
            if Pr.any():
                Wr = self.W[fired, :] - k * a_minus * self.x_post[None, :] * Pr
                self.W[fired, :] = np.where(Pr, np.clip(Wr, 0.0, self.w_max), Wr)
        elif self.rule == "rstdp":
            # accumulate into eligibility on plastic synapses only
            self.elig *= 0.995
            self.elig[:, fired] += a_plus * self.x_pre[:, None] * self.plastic[:, fired]
            self.elig[fired, :] -= a_minus * self.x_post[None, :] * self.plastic[fired, :]
        else:
            # plain STDP on all excitatory synapses
            col = self.mask[:, fired] & self.exc[:, None]
            Wc = self.W[:, fired] + a_plus * self.x_pre[:, None] * col
            self.W[:, fired] = np.where(col, np.clip(Wc, 0.0, self.w_max), Wc)
            ex_f = fired[self.exc[fired]]
            if ex_f.size:
                row = self.mask[ex_f, :]
                Wr = self.W[ex_f, :] - a_minus * self.x_post[None, :] * row
                self.W[ex_f, :] = np.where(row, np.clip(Wr, 0.0, self.w_max), Wr)
        self.x_pre[fired] += 1.0
        self.x_post[fired] += 1.0

    def run(self, ms: int) -> list[tuple[int, int]]:
        """Run `ms` steps, return (t_ms, channel) per neuron spike."""
        out = []
        for _ in range(ms):
            f = self.step()
            if f.size:
                t = self.t_ms
                out.extend((t, int(c)) for c in self.ch[f])
        return out
