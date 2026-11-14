"""Encoding/decoding and the trial protocol, shared by the fast harness and the CL-API path.

Trial timeline (ms from trial start):
  0-40    sensory stim: electrode f gets L_f pulses (L in 0..4) at 100 Hz
  0-50    response window: spike counts on every electrode
  60-160  feedback (condition-dependent)
  160-400 rest
"""
from __future__ import annotations

import numpy as np

from ..culture.core import FEEDBACK, MOTOR_DEPLOY, MOTOR_HOLD, REWARD, SENSORY

STIM_MS = 40
RESP_MS = 50
FB_START, FB_MS = 60, 100
TRIAL_MS = 400
EVAL_TRIAL_MS = 250


def encode(levels: np.ndarray) -> list[tuple[int, int]]:
    """levels: (F,) ints 0..4 -> [(t_ms, channel)] pulse schedule."""
    out = []
    for f, L in enumerate(levels):
        ch = SENSORY[f]
        for k in range(int(L)):
            out.append((k * 10, ch))
    out.sort()
    return out


def decode(counts: np.ndarray) -> float:
    """Spike counts per channel (64,) in the response window -> DEPLOY score."""
    return float(counts[MOTOR_DEPLOY].sum() - counts[MOTOR_HOLD].sum())


def feedback(rule: str, correct: bool, rng: np.random.Generator, levels=None, label=None) -> list[tuple[int, int, float]]:
    """[(t_ms_from_FB_START, channel, current_uA)] feedback stim schedule.

    fep   : DishBrain-style. Correct -> predictable: synchronous 100 Hz train on the
            feedback electrodes. Wrong -> unpredictable: random-time pulses on random
            sensory+feedback electrodes. The culture never sees a scalar reward.
    rstdp : reward electrodes (60,61 = +1 reward, 62,63 = punishment). This is a
            modelling assumption: real hardware has no dopamine channel.
    """
    if rule == "fep":
        if correct:
            return [(t, ch, 1.0) for t in range(0, FB_MS, 10) for ch in FEEDBACK]
        n = 40
        ts = rng.integers(0, FB_MS, n)
        chs = rng.choice(SENSORY + FEEDBACK, n)
        return sorted((int(t), int(c), 1.0) for t, c in zip(ts, chs))
    if rule == "teach":
        # Paired-stimulation teaching (stim-only, so it ports to real MEAs):
        # replay the sensory pattern; 3 ms after each sensory pulse stimulate the
        # CORRECT motor pool (pre->post, LTP); 3 ms before it stimulate the other
        # pool (post->pre, LTD). Runs on every trial, right or wrong.
        good = MOTOR_DEPLOY if label == 1 else MOTOR_HOLD
        bad = MOTOR_HOLD if label == 1 else MOTOR_DEPLOY
        out = []
        for k in range(4):
            t = 10 + k * 20
            sens = [SENSORY[f] for f, L in enumerate(levels) if int(L) > k]
            if not sens:
                continue
            out += [(t - 3, ch, 1.0) for ch in bad]
            out += [(t, ch, 1.0) for ch in sens]
            out += [(t + 3, ch, 1.0) for ch in good]
        return sorted(out)
    if rule == "rstdp":
        return [(0, REWARD[0] if correct else REWARD[2], 1.0)]
    return []
