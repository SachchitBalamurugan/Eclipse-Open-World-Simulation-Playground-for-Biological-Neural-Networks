"""Robot/task registry and the adapter interface between a robot and a simulated culture.

A robot plugs in with three things:

1. An env (MuJoCo/Gymnasium or anything with reset/step), named by ``env_module``.
2. A ``BNNAdapter``: the three functions that make a task trainable by a culture
     encode(obs)             -> per-electrode pulse counts   (observation -> stimulation)
     decode(counts)          -> action                       (spikes -> action)
     teach(env, obs, action) -> teaching signal              (reward, or target motor pools)
3. Entry points: how to train/evaluate locally, and the shell command one Freestyle VM
   runs for a fleet sweep (``fleet_job``).

Register with ``register_robot(RobotSpec(...))`` in one file under bnn_platform/robots/
and import that file from bnn_platform/robots/__init__.py.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol, Sequence

import numpy as np

PY = sys.executable


class BNNAdapter(Protocol):
    """Interface between one robot and the 64-electrode simulated culture
    (neurorover.culture.core.Culture).

    sensory   electrodes that receive encoded observations
    motor     electrodes whose spike counts are decoded into actions
    """

    sensory: Sequence[int]
    motor: Sequence[int]

    def encode(self, obs: np.ndarray) -> np.ndarray:
        """Observation -> integer pulse count per electrode (index = channel, 0..4 typical)."""

    def decode(self, counts: np.ndarray) -> Any:
        """(64,) spike counts in the response window -> action for env.step()."""

    def teach(self, env: Any, obs: np.ndarray, action: Any) -> dict:
        """Teaching signal after a step. Return either
        {"reward": float}                       (R-STDP / FEP-style feedback), or
        {"good": [motor chans], "bad": [...]}   (paired-stimulation 'teach' rule)."""


def run_trial(culture, levels: Sequence[int], trial_ms: int = 60, resp_ms: int = 50,
              spacing_ms: int = 10) -> np.ndarray:
    """Stimulate electrode ch with levels[ch] pulses (spacing_ms apart) and count spikes per
    electrode for resp_ms. Same protocol as surgical/culture_ctrl.CultureController._run_trial,
    reusable by any adapter. Returns (64,) int spike counts."""
    sched: dict[int, list[int]] = {}
    for ch, n in enumerate(levels):
        for k in range(int(n)):
            sched.setdefault(k * spacing_ms, []).append(ch)
    counts = np.zeros(64, int)
    for t in range(trial_ms):
        for ch in sched.get(t, []):
            culture.stim(ch, 1.0)
        fired = culture.step()
        if t < resp_ms and fired.size:
            counts += np.bincount(culture.ch[fired], minlength=64)
    return counts


@dataclass
class RobotSpec:
    """Everything the CLI needs to know about one robot/task.

    train_cmd(args)          -> argv list for local training (args has lr, episodes, seed, mode, run)
    eval_cmd(args)           -> argv list for local evaluation
    fleet_job(lr, episodes)  -> one shell line, run from the repo root on a fresh Ubuntu VM
                                (None = no fleet sweep support yet)
    adapter_factory()        -> BNNAdapter (documents/implements the encoding and decoding)
    """

    name: str
    description: str
    env_module: str
    controller: str
    train_cmd: Callable[[Any], list]
    eval_cmd: Callable[[Any], list]
    fleet_job: Optional[Callable[[float, int], str]] = None
    adapter_factory: Optional[Callable[[], BNNAdapter]] = None
    notes: str = ""
    defaults: dict = field(default_factory=dict)


_ROBOTS: dict = {}


def register_robot(spec: RobotSpec) -> RobotSpec:
    """Add (or replace) a robot in the registry; returns the spec."""
    _ROBOTS[spec.name] = spec
    return spec


def _load_builtin() -> None:
    from . import robots  # noqa: F401  (importing registers the built-in robots)


def list_robots() -> list:
    _load_builtin()
    return list(_ROBOTS.values())


def get_robot(name: str) -> RobotSpec:
    _load_builtin()
    if name not in _ROBOTS:
        raise SystemExit(f"unknown robot {name!r}; registered: {', '.join(_ROBOTS)}")
    return _ROBOTS[name]
