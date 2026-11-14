"""Seed-addressable procedural scenarios for open-world surgical reaching.

The stream varies target regions and controller-facing calibration/noise and actuator
conditions while keeping the underlying robot and scoring rule fixed. These are
state-based reach tests, not photorealistic scenes or clinical validation.

Design cues: SuFIA evaluates randomized object/gripper starts and estimation/control
noise; Orbit-Surgical randomizes initial and goal states; SuFIA-BC measures generalization
to unseen instances and small viewpoint changes. We translate those axes to this cart's
available state-based model (no gripper, tissue contact, or camera).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence

import mujoco
import numpy as np

from .arm import ArmIndex, HOME_Q, TABLE_Z, make_targets

SCENARIO_NAMES = (
    "nominal",
    "novel_targets",
    "calibration_bias",
    "measurement_noise",
    "actuator_mismatch",
    "combined_stress",
)


@dataclass(frozen=True)
class Scenario:
    name: str
    family: str
    episode: int
    seed: int
    target: tuple[float, float, float]
    randomize: bool
    error_bias_m: tuple[float, float, float]
    measurement_sigma_m: float
    action_gain: float

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "family": self.family,
            "episode": self.episode,
            "seed": self.seed,
            "target_xyz_m": list(self.target),
            "randomize": self.randomize,
            "error_bias_m": list(self.error_bias_m),
            "measurement_sigma_m": self.measurement_sigma_m,
            "action_gain": self.action_gain,
        }


def _novel_targets(m: mujoco.MjModel, count: int, seed: int) -> list[np.ndarray]:
    """Reachable targets 25-38 cm from the home pose, beyond the training sampler's 25 cm cap."""
    rng = np.random.default_rng(seed)
    ix = ArmIndex(m)
    d = mujoco.MjData(m)
    d.qpos[:] = m.qpos0
    d.qpos[ix.qadr] = HOME_Q
    mujoco.mj_forward(m, d)
    home = ix.tip(d)
    span = 0.65 * (ix.hi - ix.lo)
    out: list[np.ndarray] = []
    attempts = 0
    limit = max(12_000, count * 12_000)
    while len(out) < count and attempts < limit:
        attempts += 1
        q = np.clip(HOME_Q + rng.uniform(-span, span), ix.lo, ix.hi)
        d.qpos[:] = m.qpos0
        d.qpos[ix.qadr] = q
        mujoco.mj_forward(m, d)
        p = ix.tip(d)
        dist = float(np.linalg.norm(p - home))
        if ix.collisions(m, d):
            continue
        if TABLE_Z + 0.02 <= p[2] <= home[2] - 0.03 and 0.25 <= dist <= 0.38:
            out.append(p.copy())
    if len(out) != count:
        raise RuntimeError(
            f"could sample only {len(out)}/{count} novel targets after {attempts} attempts; "
            "the extended workspace filter may need adjustment"
        )
    return out


def _bias(rng: np.random.Generator, magnitude_m: float) -> tuple[float, float, float]:
    v = rng.normal(size=3)
    v /= max(float(np.linalg.norm(v)), 1e-12)
    return tuple(float(x) for x in v * magnitude_m)


def sample_scenario(m: mujoco.MjModel, seed: int, episode: int, family: str) -> Scenario:
    """Create one reproducible scenario; any nonnegative episode index is valid."""
    if family not in SCENARIO_NAMES:
        raise ValueError(f"unknown scenario {family!r}; choose from {', '.join(SCENARIO_NAMES)}")
    # Family-specific streams ensure the same seed and episode always produce the same target.
    family_index = SCENARIO_NAMES.index(family)
    ss = np.random.SeedSequence([int(seed), family_index, int(episode), 0x53554649])
    target_seed, disturb_seed = (int(x) for x in ss.generate_state(2))
    target_rng = np.random.default_rng(target_seed)
    disturbance_rng = np.random.default_rng(disturb_seed)

    if family in ("novel_targets", "combined_stress"):
        target = _novel_targets(m, 1, target_seed)[0]
    else:
        target = make_targets(m, 1, seed=target_seed)[0]

    bias = (0.0, 0.0, 0.0)
    sigma = 0.0
    gain = 1.0
    randomize = family != "nominal"
    if family == "calibration_bias":
        bias = _bias(disturbance_rng, 0.010)  # 10 mm constant target-frame bias (stress assumption)
    elif family == "measurement_noise":
        sigma = 0.003  # 3 mm independent measurement noise (stress assumption)
    elif family == "actuator_mismatch":
        gain = float(disturbance_rng.uniform(0.80, 1.20))
    elif family == "combined_stress":
        bias = _bias(disturbance_rng, 0.010)
        sigma = 0.003
        gain = float(disturbance_rng.uniform(0.80, 1.20))
    # Episode seed is distinct across distributed workers but fully reproducible.
    episode_seed = int(np.random.SeedSequence([int(seed), family_index, int(episode), 0x0B17]).generate_state(1)[0])
    return Scenario(
        name=f"{family}_{episode:04d}", family=family, episode=int(episode), seed=episode_seed,
        target=tuple(float(x) for x in target), randomize=randomize,
        error_bias_m=bias, measurement_sigma_m=sigma, action_gain=gain,
    )


def iter_scenarios(
    m: mujoco.MjModel,
    seed: int = 0,
    episodes: int | None = None,
    families: Sequence[str] | None = None,
) -> Iterator[Scenario]:
    """Yield a deterministic, open-ended stream (or `episodes` per family if bounded).

    An unbounded call can keep generating new in-domain and out-of-domain targets by seed;
    the finite evaluator uses the same generator and records all seeds for replay.
    """
    names = tuple(families) if families is not None else SCENARIO_NAMES
    if not names or any(n not in SCENARIO_NAMES for n in names):
        raise ValueError("families must be a non-empty subset of SCENARIO_NAMES")
    if episodes is not None and episodes < 1:
        raise ValueError("episodes must be positive or None for an open-ended stream")
    i = 0
    while episodes is None or i < episodes * len(names):
        family = names[i % len(names)]
        episode = i // len(names)
        yield sample_scenario(m, seed, episode, family)
        i += 1

