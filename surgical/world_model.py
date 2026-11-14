"""A small supervised forward model for the simulated surgical reach task.

This predicts one MuJoCo control-step transition from IK-generated synthetic
data. It is not a policy, patient model, or generative simulator. The surgical
learning papers used as context discuss simulation/robot-learning workflows
(arXiv:2404.16027), language-guided planning and re-planning (2405.05226), and
visuomotor behavior cloning from demonstrations (2504.14857); none is called a
generative dynamics model here.

Run from the repository root: python -m surgical.world_model
Uses only NumPy and this repository's MuJoCo/Gymnasium environment.
"""
from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "results" / "world_model_metrics.json"
OBS_DIM, STATE_DIM, N_JOINTS, DR_DIM = 20, 19, 8, 17
FEATURE_DIM = OBS_DIM + 3 * N_JOINTS + DR_DIM


def state_from_obs(obs: np.ndarray) -> np.ndarray:
    """Project obs=[error xyz, distance, qnorm(8), qdot-scaled(8)] to 19-D."""
    o = np.asarray(obs, dtype=float).reshape(-1)
    if o.size != OBS_DIM:
        raise ValueError(f"expected {OBS_DIM}-D SurgicalReachEnv observation, got {o.size}")
    return np.r_[o[:3], o[4:]]


def obs_from_state(state: np.ndarray) -> np.ndarray:
    s = np.asarray(state, dtype=float).reshape(-1)
    if s.size != STATE_DIM:
        raise ValueError(f"expected {STATE_DIM}-D state, got {s.size}")
    return np.r_[s[:3], np.linalg.norm(s[:3]), s[3:]].astype(np.float32)


def _dr_context(env) -> np.ndarray:
    bodies = np.asarray(sorted(env.ix.arm_bodies), dtype=int)
    m0, m = np.asarray(env.mass0[bodies], float), np.asarray(env.m.body_mass[bodies], float)
    mr = np.divide(m, m0, out=np.ones_like(m), where=m0 > 0)
    d = np.asarray(env.ix.dadr, dtype=int)
    d0, damp = np.asarray(env.damp0[d], float), np.asarray(env.m.dof_damping[d], float)
    dr = np.divide(damp, d0, out=np.ones_like(damp), where=d0 > 1e-9)
    return np.r_[mr, dr]


def features_from_components(state: np.ndarray, action: np.ndarray, qref: np.ndarray,
                             action_filter: np.ndarray, dr_context: np.ndarray,
                             lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Build a 61-D pre-step input from true state plus actuator/filter state."""
    s = np.asarray(state, dtype=float).reshape(-1)
    a = np.clip(np.asarray(action, dtype=float).reshape(-1), -1, 1)
    qref, af = np.asarray(qref, float).reshape(-1), np.asarray(action_filter, float).reshape(-1)
    lo, hi, dr = np.asarray(lo, float), np.asarray(hi, float), np.asarray(dr_context, float).reshape(-1)
    if (s.size, a.size, qref.size, af.size, dr.size, lo.size, hi.size) != (19, 8, 8, 8, 17, 8, 8):
        raise ValueError("unexpected state/action/filter/domain-randomization dimensions")
    qref_norm = 2 * (qref - lo) / np.maximum(hi - lo, 1e-9) - 1
    return np.r_[obs_from_state(s), a, qref_norm, af, dr]


def features_from_env(env, obs: np.ndarray, action: np.ndarray) -> np.ndarray:
    """Build a model input before ``env.step``; obs must be the true 20-D state."""
    return features_from_components(state_from_obs(obs), action, env.qref, env.a_f,
                                    _dr_context(env), env.ix.lo, env.ix.hi)


class SurgicalWorldModel:
    """Bootstrap ensemble of random-Fourier-feature ridge transition regressors.

    Ensemble spread and nearest-training distance are diagnostics, not calibrated
    confidence. This NumPy-only model approximates a smooth nonlinear transition.
    """
    def __init__(self, n_features: int = 128, n_models: int = 5,
                 ridge: float = 1e-2, seed: int = 0):
        if n_features < 8 or n_models < 1 or ridge <= 0:
            raise ValueError("need n_features>=8, n_models>=1, and ridge>0")
        self.n_features, self.n_models = int(n_features), int(n_models)
        self.ridge, self.seed, self.fitted = float(ridge), int(seed), False

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "SurgicalWorldModel":
        X, Y = np.asarray(X, float), np.asarray(Y, float)
        if X.ndim != 2 or X.shape[1] != FEATURE_DIM or Y.shape != (len(X), STATE_DIM):
            raise ValueError(f"expected X=(n,{FEATURE_DIM}), Y=(n,{STATE_DIM}); got {X.shape}, {Y.shape}")
        if len(X) < 12 or not np.isfinite(X).all() or not np.isfinite(Y).all():
            raise ValueError("fit needs at least 12 finite transitions")
        self.x_mean, self.x_scale = X.mean(0), X.std(0)
        self.x_scale[self.x_scale < 1e-6] = 1.0
        Xn = (X - self.x_mean) / self.x_scale
        self.y_mean, self.y_scale = Y.mean(0), Y.std(0)
        self.y_scale[self.y_scale < 1e-5] = 1.0
        Yn = (Y - self.y_mean) / self.y_scale
        rng = np.random.default_rng(self.seed)
        ia, ib = rng.integers(0, len(Xn), 2048), rng.integers(0, len(Xn), 2048)
        ds = np.linalg.norm(Xn[ia] - Xn[ib], axis=1)
        ds = ds[ds > 1e-9]
        self.length_scale = max(float(np.median(ds)) if len(ds) else 1.0, 1e-3)
        self.rff_w = rng.normal(0, 1 / self.length_scale, (FEATURE_DIM, self.n_features))
        self.rff_b = rng.uniform(0, 2 * np.pi, self.n_features)
        Phi = self._map(Xn)
        eye = np.eye(Phi.shape[1])
        self.models = []
        for _ in range(self.n_models):
            ids = rng.integers(0, len(Phi), len(Phi))
            p, y = Phi[ids], Yn[ids]
            self.models.append(np.linalg.solve(p.T @ p + self.ridge * eye, p.T @ y))
        self.x_train_norm = Xn
        self.n_train, self.fitted = int(len(X)), True
        return self

    def _map(self, Xn: np.ndarray) -> np.ndarray:
        Xn = np.atleast_2d(Xn)
        rff = np.sqrt(2 / self.n_features) * np.cos(Xn @ self.rff_w + self.rff_b)
        return np.c_[np.ones(len(Xn)), Xn, rff]

    def predict_features(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return predicted 19-D delta, ensemble std, and nearest-train novelty."""
        if not self.fitted:
            raise RuntimeError("fit the model before prediction")
        X = np.asarray(X, float)
        one = X.ndim == 1
        Xn = (np.atleast_2d(X) - self.x_mean) / self.x_scale
        if Xn.shape[1] != FEATURE_DIM:
            raise ValueError(f"expected {FEATURE_DIM} model inputs, got {Xn.shape}")
        Phi = self._map(Xn)
        P = np.stack([Phi @ w for w in self.models])
        mean = P.mean(0) * self.y_scale + self.y_mean
        std = P.std(0) * self.y_scale
        d2 = ((Xn ** 2).sum(1, keepdims=True) + (self.x_train_norm ** 2).sum(1)[None, :]
              - 2 * Xn @ self.x_train_norm.T)
        novelty = np.sqrt(np.maximum(d2.min(1), 0))
        if one:
            return mean[0], std[0], novelty[0]
        return mean, std, novelty

    def predict_step(self, env, obs: np.ndarray, action: np.ndarray) -> dict[str, Any]:
        """Predict one step. ``obs`` is the true simulator state; controller obs may be noisy."""
        delta, std, novelty = self.predict_features(features_from_env(env, obs, action))
        return {"next_obs": obs_from_state(state_from_obs(obs) + delta),
                "delta_state": delta, "tip_delta_m": -delta[:3].copy(),
                "tip_delta_uncertainty_mm": float(np.linalg.norm(std[:3]) * 1000),
                "novelty": float(novelty)}


def collect_ik_transitions(episodes: int, seed_start: int, action_noise: float = 0.05,
                           randomize: bool = True) -> dict[str, Any]:
    """Collect synthetic IK trajectories with disjoint, explicit episode seeds."""
    from .arm import make_targets
    from .controllers import DLSIK
    from .reach_env import EP_STEPS, SurgicalReachEnv
    if episodes < 1 or action_noise < 0:
        raise ValueError("episodes must be positive and action_noise nonnegative")
    env = SurgicalReachEnv(seed=seed_start, randomize=randomize,
                           target_pool=np.zeros((1, 3), dtype=float))
    targets = make_targets(env.m, episodes, seed=seed_start + 1234567)
    ctrl, rng = DLSIK(), np.random.default_rng(seed_start + 7654321)
    X, S, S1, rollouts, seeds = [], [], [], [], []
    for i, target in enumerate(targets):
        seed = int(seed_start + i)
        seeds.append(seed)
        obs, _ = env.reset(seed=seed, options={"target": target, "noise": False})
        ctrl.reset(env)
        roll = {"seed": seed, "states": [state_from_obs(obs)], "actions": [],
                "qref": [env.qref.copy()], "action_filter": [env.a_f.copy()],
                "dr_context": _dr_context(env), "lo": env.ix.lo.copy(),
                "hi": env.ix.hi.copy(), "scale": env.scale.copy()}
        for _ in range(EP_STEPS):
            a = ctrl.act(env, obs)
            if action_noise:
                a = np.clip(a + rng.normal(0, action_noise, a.shape), -1, 1)
            state = state_from_obs(obs)
            X.append(features_from_components(state, a, env.qref, env.a_f,
                                              roll["dr_context"], env.ix.lo, env.ix.hi))
            next_obs, _, term, trunc, _ = env.step(a)
            s1 = state_from_obs(next_obs)
            S.append(state); S1.append(s1)
            roll["actions"].append(a.copy()); roll["states"].append(s1.copy())
            roll["qref"].append(env.qref.copy()); roll["action_filter"].append(env.a_f.copy())
            obs = next_obs
            if term or trunc:
                break
        rollouts.append(roll)
    S, S1 = np.asarray(S), np.asarray(S1)
    return {"X": np.asarray(X), "Y": S1 - S, "states": S, "next_states": S1,
            "rollouts": rollouts, "seeds": seeds, "randomize": randomize,
            "action_noise": float(action_noise)}


def _tip_rmse_mm(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((pred[:, :3] - truth[:, :3]) ** 2, axis=1))) * 1000)


def evaluate_transitions(model: SurgicalWorldModel, data: dict[str, Any], split: str) -> dict[str, Any]:
    """Teacher-forced one-step test, compared with predicting zero state change."""
    mean, std, novelty = model.predict_features(data["X"])
    y = data["Y"]
    rmse, base = _tip_rmse_mm(mean, y), _tip_rmse_mm(np.zeros_like(y), y)
    unc = np.linalg.norm(std[:, :3], axis=1) * 1000
    err = np.linalg.norm((mean - y)[:, :3], axis=1) * 1000
    return {"split": split, "evaluation_type": "teacher_forced_one_step",
            "transitions": int(len(y)), "tip_displacement_rmse_mm": round(rmse, 3),
            "persistence_zero_delta_tip_rmse_mm": round(base, 3),
            "rmse_improvement_vs_persistence_pct": round(100 * (base - rmse) / max(base, 1e-9), 2),
            "tip_delta_abs_error_mm_median": round(float(np.median(err)), 3),
            "joint_q_delta_mae_norm": round(float(np.mean(np.abs(mean[:, 3:11] - y[:, 3:11]))), 6),
            "ensemble_tip_uncertainty_mm_p50_p95": [round(float(x), 3) for x in np.percentile(unc, [50, 95])],
            "nearest_train_feature_distance_p50_p95": [round(float(x), 3) for x in np.percentile(novelty, [50, 95])]}


def evaluate_open_loop(model: SurgicalWorldModel, rollouts: list[dict[str, Any]],
                       horizons: tuple[int, ...] = (1, 5, 10)) -> dict[str, Any]:
    """Multi-step state predictions under recorded IK actions (not policy rollouts)."""
    out = []
    for h in horizons:
        errors, persist = [], []
        for r in rollouts:
            states, actions = np.asarray(r["states"]), np.asarray(r["actions"])
            qrefs, filters = np.asarray(r["qref"]), np.asarray(r["action_filter"])
            if len(actions) < h:
                continue
            starts = range(0, len(actions) - h + 1)
            for start in starts:
                if start + h > len(actions):
                    continue
                s, qref, af = states[start].copy(), qrefs[start].copy(), filters[start].copy()
                for k in range(h):
                    a = actions[start + k]
                    x = features_from_components(s, a, qref, af, r["dr_context"], r["lo"], r["hi"])
                    d, _, _ = model.predict_features(x)
                    s += d
                    af = 0.5 * af + 0.5 * a
                    qref = np.clip(qref + af * r["scale"], r["lo"], r["hi"])
                truth = states[start + h]
                errors.append(float(np.linalg.norm(s[:3] - truth[:3]) * 1000))
                persist.append(float(np.linalg.norm(states[start, :3] - truth[:3]) * 1000))
        if errors:
            out.append({"horizon_steps": int(h), "evaluation_type": "recorded_action_open_loop",
                        "windows": int(len(errors)), "tip_state_rmse_mm": round(float(np.sqrt(np.mean(np.square(errors)))), 3),
                        "persistence_tip_state_rmse_mm": round(float(np.sqrt(np.mean(np.square(persist)))), 3),
                        "tip_state_error_median_mm": round(float(np.median(errors)), 3)})
    return {"horizons": out,
            "note": "Actions are from the held-out IK trajectory; this is a model test, not a policy rollout."}


def fit_ik_world_model(train_episodes: int = 12, eval_episodes: int = 3, seed: int = 0,
                       train_seed_start: int = 700000, eval_seed_start: int = 800000,
                       action_noise: float = 0.05, n_features: int = 128,
                       n_models: int = 5) -> tuple[SurgicalWorldModel, dict[str, Any]]:
    """Fit on IK transitions and report on held-out IK seeds plus prediction diagnostics."""
    if set(range(train_seed_start, train_seed_start + train_episodes)) & set(range(eval_seed_start, eval_seed_start + eval_episodes)):
        raise ValueError("training and evaluation seeds must be disjoint")
    train = collect_ik_transitions(train_episodes, train_seed_start, action_noise, True)
    test = collect_ik_transitions(eval_episodes, eval_seed_start, action_noise, True)
    model = SurgicalWorldModel(n_features, n_models, seed=seed).fit(train["X"], train["Y"])
    metrics = {
        "method": "supervised random-Fourier-feature ridge ensemble on synthetic IK transitions",
        "model_role": "one-step forward dynamics predictor; not a controller or generative model",
        "input_features": FEATURE_DIM, "predicted_delta_dimensions": STATE_DIM,
        "training": {"episodes": int(train_episodes), "transitions": int(len(train["X"])),
                     "seed_start": int(train_seed_start), "seed_end": int(train_seed_start + train_episodes - 1),
                     "controller": "DLSIK + clipped Gaussian action noise", "action_noise_std": float(action_noise),
                     "domain_randomization": "enabled"},
        "heldout_ik": {"episodes": int(eval_episodes), "transitions": int(len(test["X"])),
                       "seed_start": int(eval_seed_start), "seed_end": int(eval_seed_start + eval_episodes - 1)},
        "one_step": evaluate_transitions(model, test, "heldout_ik"),
        "open_loop": evaluate_open_loop(model, test["rollouts"]),
        "uncertainty_note": "Ensemble disagreement and nearest standardized training distance are diagnostics, not calibrated confidence."}
    return model, metrics


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-episodes", type=int, default=12)
    ap.add_argument("--eval-episodes", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-seed-start", type=int, default=700000)
    ap.add_argument("--eval-seed-start", type=int, default=800000)
    ap.add_argument("--action-noise", type=float, default=0.05)
    ap.add_argument("--features", type=int, default=128)
    ap.add_argument("--ensemble", type=int, default=5)
    ap.add_argument("--output", type=pathlib.Path, default=OUT)
    a = ap.parse_args()
    model, metrics = fit_ik_world_model(a.train_episodes, a.eval_episodes, a.seed,
                                        a.train_seed_start, a.eval_seed_start,
                                        a.action_noise, a.features, a.ensemble)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
