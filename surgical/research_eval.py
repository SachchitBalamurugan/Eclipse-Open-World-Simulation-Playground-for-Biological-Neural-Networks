"""Matched culture/IK evaluation on the procedural surgical scenario stream.

Example: python -m surgical.research_eval --seed 100 --episodes 3 --out surgical/results/research_100.json
The model checkpoints are fixed; no held-out target is used to adapt them.
"""
from __future__ import annotations

import argparse
import copy
import json
import pathlib
import pickle
from collections import defaultdict

import numpy as np

from .controllers import DLSIK
from .reach_env import SurgicalReachEnv
from .scenarios import SCENARIO_NAMES, Scenario, iter_scenarios

ROOT = pathlib.Path(__file__).resolve().parent
RESULTS = ROOT / "results"


class ScenarioView:
    """Controller view with biased/noisy perception and actuator gain mismatch."""

    def __init__(self, env: SurgicalReachEnv, spec: Scenario):
        self.base, self.spec = env, spec
        self.rng = np.random.default_rng(spec.seed)
        self.last_applied_action = np.zeros(8, dtype=float)

    def __getattr__(self, name):
        return getattr(self.base, name)

    def err(self):
        # CultureController and DLSIK both call this method; neither sees true error.
        return (self.base.err() + np.asarray(self.spec.error_bias_m)
                + self.rng.normal(0.0, self.spec.measurement_sigma_m, 3))

    def err_arm_frame(self):
        rot = self.base.d.xmat[self.base.hand].reshape(3, 3)
        return rot.T @ self.err()

    def _obs(self, obs):
        out = np.asarray(obs, dtype=np.float32).copy()
        out[:3] = self.err()
        out[3] = float(np.linalg.norm(out[:3]))
        return out

    def reset(self, seed=None, options=None):
        seed = self.spec.seed if seed is None else int(seed)
        self.rng = np.random.default_rng(seed ^ 0x51F1A)
        opts = dict(options or {})
        opts.update(target=np.asarray(self.spec.target, dtype=float), noise=False)
        self.base.randomize = self.spec.randomize
        obs, info = self.base.reset(seed=seed, options=opts)
        return self._obs(obs), info

    def step(self, action):
        requested = np.clip(np.asarray(action, dtype=float), -1, 1)
        self.last_applied_action = np.clip(requested * self.spec.action_gain, -1, 1)
        obs, reward, term, trunc, info = self.base.step(self.last_applied_action)
        return self._obs(obs), reward, term, trunc, info


def _load_checkpoints():
    out = {}
    for name in ("culture_teach", "culture_frozen"):
        path = RESULTS / f"{name}.pkl"
        if not path.is_file():
            raise FileNotFoundError(f"fixed controller checkpoint missing: {path}")
        out[name] = pickle.loads(path.read_bytes())
    return out


def _fit_world_model(args):
    if args.skip_world_model:
        return None, {"status": "skipped_by_flag"}
    try:
        from .world_model import fit_ik_world_model
    except ImportError as exc:
        return None, {"status": "unavailable", "detail": str(exc)}
    try:
        model, metrics = fit_ik_world_model(
            train_episodes=args.world_model_train_episodes,
            eval_episodes=args.world_model_eval_episodes,
            train_seed_start=700000,
            eval_seed_start=800000,
        )
        return model, {"status": "fit", "train_seed_start": 700000,
                       "eval_seed_start": 800000, "metrics": metrics}
    except Exception as exc:  # keep the culture evaluation usable if model fitting fails
        return None, {"status": "error", "detail": f"{type(exc).__name__}: {exc}"}


def _run_episode(env, spec: Scenario, ctrl, world_model=None):
    view = ScenarioView(env, spec)
    obs, _ = view.reset(seed=spec.seed)
    ctrl.reset(view)
    if hasattr(ctrl, "learning"):
        ctrl.learning = False
    wm_err, wm_unc, wm_novelty, wm_state = [], [], [], []
    last = {}
    for _ in range(200):
        true_obs = env._obs().astype(float, copy=True)  # 20D simulator state, reserved for dynamics audit
        action = np.clip(np.asarray(ctrl.act(view, obs), dtype=float), -1, 1)
        applied = np.clip(action * spec.action_gain, -1, 1)
        before_tip = env.ix.tip(env.d).copy()
        prediction = world_model.predict_step(env, true_obs, applied) if world_model is not None else None
        obs, _, term, trunc, last = view.step(action)
        true_next = env._obs().astype(float, copy=True)
        if prediction is not None:
            predicted_next = np.asarray(prediction["next_obs"], dtype=float).reshape(-1)
            if predicted_next.size == true_next.size:
                wm_state.append(float(np.sqrt(np.mean((predicted_next - true_next) ** 2))))
            pred_tip_delta = np.asarray(prediction["tip_delta_m"], dtype=float).reshape(3)
            true_tip_delta = env.ix.tip(env.d) - before_tip
            wm_err.append(float(np.linalg.norm(pred_tip_delta - true_tip_delta) * 1000))
            wm_unc.append(float(prediction["tip_delta_uncertainty_mm"]))
            wm_novelty.append(float(prediction["novelty"]))
        if term or trunc:
            break
    row = {
        "scenario": spec.family,  # stable grouping key for the multi-VM aggregator
        "scenario_id": spec.name,
        "episode": spec.episode,
        "seed": spec.seed,
        "target_xyz_m": list(spec.target),
        "family": spec.family,
        "controller": ctrl.name,
        "success": bool(last.get("success", False)),
        "final_err_mm": float(last.get("dist", np.nan)) * 1000,
        "reach_time_s": (float(last["reached_step"]) * 0.05 if last.get("reached_step") else None),
        "limit_hits": int(last.get("limit_hits", 0)),
        "collisions": int(last.get("collisions", 0)),
    }
    if wm_err:
        row["world_model"] = {
            "steps": len(wm_err),
            "tip_prediction_error_mm_mean": float(np.mean(wm_err)),
            "tip_prediction_error_mm_median": float(np.median(wm_err)),
            "tip_delta_uncertainty_mm_mean": float(np.mean(wm_unc)),
            "novelty_mean": float(np.mean(wm_novelty)),
            "next_obs_rmse_mean": float(np.mean(wm_state)) if wm_state else None,
        }
    return row


def _summary(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["scenario"], r["controller"])].append(r)
    out = {}
    for (scenario, controller), group in groups.items():
        errors = [r["final_err_mm"] for r in group if np.isfinite(r["final_err_mm"])]
        times = [r["reach_time_s"] for r in group if r["reach_time_s"] is not None]
        out.setdefault(scenario, {})[controller] = {
            "n": len(group),
            "success_rate": float(np.mean([r["success"] for r in group])),
            "median_error_mm": float(np.median(errors)) if errors else None,
            "mean_error_mm": float(np.mean(errors)) if errors else None,
            "reach_rate": len(times) / len(group),
            "median_reach_time_s": float(np.median(times)) if times else None,
            "collision_episode_rate": float(np.mean([r["collisions"] > 0 for r in group])),
            "mean_collisions": float(np.mean([r["collisions"] for r in group])),
            "mean_joint_limit_steps": float(np.mean([r["limit_hits"] for r in group])),
            "world_model_tip_error_mm_mean": float(np.mean([
                r["world_model"]["tip_prediction_error_mm_mean"] for r in group if "world_model" in r
            ])) if any("world_model" in r for r in group) else None,
        }
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=3, help="episodes per scenario family")
    ap.add_argument("--out", default=None, help="output JSON path")
    ap.add_argument("--scenarios", nargs="+", choices=SCENARIO_NAMES, default=list(SCENARIO_NAMES))
    ap.add_argument("--skip-world-model", action="store_true")
    ap.add_argument("--world-model-train-episodes", type=int, default=12)
    ap.add_argument("--world-model-eval-episodes", type=int, default=3)
    args = ap.parse_args(argv)
    if args.episodes < 1:
        ap.error("--episodes must be positive")

    checkpoints = _load_checkpoints()
    # A single model/pool is reused; every episode reset is keyed by the recorded seed.
    env = SurgicalReachEnv(seed=args.seed, randomize=True, target_noise=0.0)
    specs = list(iter_scenarios(env.m, args.seed, args.episodes, args.scenarios))
    world_model, wm_meta = _fit_world_model(args)
    rows = []
    for spec in specs:
        for name in ("culture_teach", "culture_frozen", "ik_dls"):
            # A fresh deep copy per episode preserves the fixed checkpoint weights and
            # prevents spike/adaptation state from leaking across test targets.
            ctrl = copy.deepcopy(checkpoints[name]) if name.startswith("culture_") else DLSIK()
            ctrl.name = name
            rows.append(_run_episode(env, spec, ctrl, world_model))
    env.close()

    out_obj = {
        "schema_version": 1,
        "seed": args.seed,
        "episodes_per_scenario": args.episodes,
        "controllers": ["culture_teach", "culture_frozen", "ik_dls"],
        "biology": "simulated Izhikevich culture only; no living neural tissue",
        "scenario_definitions": [s.as_dict() for s in specs],
        "summary": _summary(rows),
        "rows": rows,
        "world_model": wm_meta,
        "limitations": [
            "state-based single-arm reaching only; no tissue contact, instruments, or camera simulation",
            "bias/noise/action-gain ranges are procedural stress assumptions, not measured clinical distributions",
            "DLSIK is a geometry-aware reference and uses the simulator Jacobian",
            "three episodes per family are robustness checks, not precise success estimates",
        ],
    }
    out = pathlib.Path(args.out) if args.out else RESULTS / f"research_eval_seed{args.seed}.json"
    if out.exists() and out.is_dir():
        out = out / f"research_eval_seed{args.seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(out_obj, indent=1, allow_nan=False))
    success = {scenario: {controller: round(metrics["success_rate"], 3)
                          for controller, metrics in by_controller.items()}
               for scenario, by_controller in out_obj["summary"].items()}
    print(json.dumps({"seed": args.seed, "episodes_per_scenario": args.episodes,
                      "scenario_count": len(specs), "out": str(out), "success_rates": success,
                      "world_model": wm_meta}, separators=(",", ":")))


if __name__ == "__main__":
    main()

