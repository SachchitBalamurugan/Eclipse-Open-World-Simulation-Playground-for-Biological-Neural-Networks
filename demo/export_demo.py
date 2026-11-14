"""Export a small, deterministic playback bundle for the table demo.

Run from the repository root with:
    .venv/Scripts/python demo/export_demo.py

Only validation nights are included in the playback JSON. Each culture is first
presented the same 100-sample training stream used by neurorover.bench, then the
readout is evaluated on a stratified 40-night subset of the fixed validation
set. The frozen control uses the benchmark's direct motor decoder.
"""
from __future__ import annotations

import copy
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neurorover import bench
from neurorover.culture.core import Culture, FEEDBACK, MOTOR_DEPLOY, MOTOR_HOLD, REWARD, SENSORY
from neurorover.interface import codec
from neurorover.env.frost import FrostTask

OUT = pathlib.Path(__file__).resolve().parent / "demo_data.json"
SEED = 0
TRAIN_N = 100
PLAYBACK_N = 40
MAX_JSON_BYTES = 5_800_000
TRAINING_PLAYBACK_N = 40
SPIKES_PER_TRAINING_TRIAL = 2500


def _summary_numbers() -> dict:
    grouped: dict[str, dict[str, list[float]]] = {}
    results = ROOT / "neurorover" / "results"
    if results.exists():
        for path in sorted(results.glob("*.json")):
            try:
                row = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if row.get("tag"):
                continue
            condition = row.get("condition")
            if not condition:
                continue
            for budget, value in (row.get("auc") or {}).items():
                try:
                    grouped.setdefault(str(condition), {}).setdefault(str(int(budget)), []).append(float(value))
                except (TypeError, ValueError):
                    continue
    return {
        condition: {
            budget: {
                "mean_auc": float(np.mean(values)),
                "n": len(values),
            }
            for budget, values in sorted(points.items(), key=lambda item: int(item[0]))
        }
        for condition, points in sorted(grouped.items())
    }


def _record_trial(c: Culture, levels: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, float, int, list[list[int]]]:
    """Run one benchmark trial and capture its 1 ms electrode spike events."""
    spikes: list[list[int]] = []
    local_t = 0
    original_step = c.step

    def recording_step():
        nonlocal local_t
        fired = original_step()
        if fired.size:
            channels = c.ch[fired]
            spikes.extend([[local_t, int(channel)] for channel in channels])
        local_t += 1
        return fired

    c.step = recording_step  # type: ignore[method-assign]
    try:
        counts, raw_score, raw_action, _ = bench.run_trial(c, levels, rng)
    finally:
        c.step = original_step  # type: ignore[method-assign]
    return counts, float(raw_score), int(raw_action), spikes


def _trial_record(row: dict, levels: np.ndarray, counts: np.ndarray, score: float,
                  action: int, spikes: list[list[int]]) -> dict:
    deploy = counts[MOTOR_DEPLOY].astype(int).tolist()
    hold = counts[MOTOR_HOLD].astype(int).tolist()
    features = row["features"]
    return {
        "station_id": row["station"],
        "date": row["date"],
        "last_input_date": row.get("last_input_date"),
        "feature_levels": np.asarray(levels, dtype=int).tolist(),
        "spikes": spikes,
        "motor_counts": {
            "deploy": deploy,
            "hold": hold,
            "deploy_total": int(sum(deploy)),
            "hold_total": int(sum(hold)),
        },
        "readout_score": float(score),
        "action": int(action),
        "action_label": "DEPLOY" if action else "HOLD",
        "target": int(row["target"]),
        "target_label": "Freeze" if row["target"] else "No freeze",
        "correct": int(action == int(row["target"])),
        "weather": {
            "tmin_lag3_c": float(features[0]),
            "tmin_lag4_c": float(features[1]),
            "tmin_lag7_c": float(features[2]),
            "tmin_7d_mean_c": float(features[3]),
        },
    }


def _validation_indices(task: FrostTask, rows: list[dict]) -> list[int]:
    """Pick and shuffle 20 freeze + 20 non-freeze rows from FrostTask's set."""
    y = np.asarray([row["target"] for row in rows], dtype=int)
    erng = np.random.default_rng(12345)
    pos = erng.choice(np.flatnonzero(y == 1), 150, replace=False)
    neg = erng.choice(np.flatnonzero(y == 0), 150, replace=False)
    fixed_subset = np.concatenate([pos, neg])
    selected = np.concatenate([fixed_subset[:20], fixed_subset[150:170]])
    np.random.default_rng(2026).shuffle(selected)
    # Guard the metadata alignment against any accidental mismatch with FrostTask.
    if not np.array_equal(np.asarray([rows[i]["target"] for i in fixed_subset]), task.yev):
        raise RuntimeError("Validation row order no longer matches FrostTask")
    return [int(i) for i in selected]


def _run_culture(condition: str, task: FrostTask, train_x: np.ndarray,
                 train_y: np.ndarray, validation_rows: list[dict], eval_indices: list[int]) -> dict:
    rng = np.random.default_rng(SEED)
    levels = task.levels(train_x)
    culture = Culture(seed=SEED, rule="frozen")
    culture.run(1000)  # Same settling period as bench.run_job.

    train_counts = np.zeros((TRAIN_N, 64))
    for i in range(TRAIN_N):
        train_counts[i], _, _, _ = bench.run_trial(
            culture, levels[i], rng, label=int(train_y[i]), rule="frozen"
        )

    readout = None
    if condition == "culture_reservoir":
        mu = train_counts.mean(0)
        sd = train_counts.std(0) + 1e-6
        readout = bench.fit_logistic((train_counts - mu) / sd, train_y, lam=10.0)

    # Match bench.eval_culture's frozen evaluation copy and decode path.
    eval_culture = copy.deepcopy(culture)
    eval_culture.rule = "frozen"
    eval_culture.elig = None
    trials = []
    for original_index in eval_indices:
        source = validation_rows[original_index]
        x = np.asarray(source["features"], dtype=float)[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14]]
        encoded = task.levels(x)[0]
        counts, raw_score, raw_action, spikes = _record_trial(eval_culture, encoded, rng)
        if readout is None:
            score, action = raw_score, raw_action
        else:
            score = float(readout(((counts - mu) / sd)[None, :])[0])
            action = int(score > 0) if score != 0 else int(rng.random() < 0.5)
        trials.append(_trial_record(source, encoded, counts, score, action, spikes))

    return {
        "condition": condition,
        "seed": SEED,
        "training_samples": TRAIN_N,
        "readout": "logistic on standardized electrode counts" if readout else "motor DEPLOY minus HOLD",
        "trials": trials,
    }


def _plastic_means(c: Culture, mask_d: np.ndarray, mask_h: np.ndarray) -> tuple[float, float]:
    return float(c.W[mask_d].mean()), float(c.W[mask_h].mean())


def _record_training(condition: str, task: FrostTask, train_x: np.ndarray, train_y: np.ndarray) -> dict:
    """Replay bench.run_job's first training trials, capturing spikes, stims and weights."""
    rule = bench.RULE[condition]
    rng = np.random.default_rng(SEED)
    c = Culture(seed=SEED, rule="stdp" if rule == "fep" else rule)
    c.run(1000)
    levels = task.levels(train_x)
    sens_rows = np.isin(c.ch, SENSORY)[:, None]
    mask_d = c.plastic & sens_rows & np.isin(c.ch, MOTOR_DEPLOY)[None, :]
    mask_h = c.plastic & sens_rows & np.isin(c.ch, MOTOR_HOLD)[None, :]
    track_w = condition == "culture_fep"
    w0 = _plastic_means(c, mask_d, mask_h) if track_w else None
    original_step, original_stim = c.step, c.stim
    trials = []
    for i in range(TRAINING_PLAYBACK_N):
        spikes: list[list[int]] = []
        stims: list[list] = []
        local_t = 0

        def recording_step():
            nonlocal local_t
            fired = original_step()
            if fired.size:
                spikes.extend([[local_t, int(ch)] for ch in c.ch[fired]])
            local_t += 1
            return fired

        def recording_stim(channel, current_uA=1.0):
            if channel in REWARD:
                kind = "reward"
            elif local_t >= codec.FB_START:
                kind = "feedback"
            else:
                kind = "sensory"
            stims.append([local_t, int(channel), kind])
            return original_stim(channel, current_uA)

        c.step, c.stim = recording_step, recording_stim  # type: ignore[method-assign]
        try:
            counts, score, action, ok = bench.run_trial(c, levels[i], rng, label=int(train_y[i]), rule=rule)
        finally:
            c.step, c.stim = original_step, original_stim  # type: ignore[method-assign]
        all_counts = np.bincount([ch for _, ch in spikes], minlength=64).astype(int) if spikes else np.zeros(64, int)
        if len(spikes) > SPIKES_PER_TRAINING_TRIAL:
            keep = np.linspace(0, len(spikes) - 1, SPIKES_PER_TRAINING_TRIAL).astype(int)
            spikes_out = [spikes[k] for k in keep]
        else:
            spikes_out = spikes
        rec = {
            "index": i,
            "feature_levels": np.asarray(levels[i], dtype=int).tolist(),
            "tmin_c": [round(float(v), 2) for v in train_x[i, :4]],
            "spikes": spikes_out,
            "spikes_total": len(spikes),
            "spike_counts": all_counts.tolist(),
            "stims": stims,
            "readout_score": float(score),
            "action": int(action),
            "action_label": "DEPLOY" if action else "HOLD",
            "target": int(train_y[i]),
            "correct": int(bool(ok)),
            "motor_counts": {"deploy_total": int(counts[MOTOR_DEPLOY].sum()), "hold_total": int(counts[MOTOR_HOLD].sum())},
        }
        if track_w:
            wd, wh = _plastic_means(c, mask_d, mask_h)
            rec["w_deploy"], rec["w_hold"] = wd, wh
        trials.append(rec)
    return {
        "condition": condition,
        "rule": rule,
        "seed": SEED,
        "feedback": "DishBrain-style: correct = predictable 100 Hz train on electrodes 56-59; wrong = 40 random pulses on sensory+feedback electrodes"
        if rule == "fep" else "none (frozen reservoir; readout trained offline)",
        "w_initial": {"deploy": w0[0], "hold": w0[1]} if w0 else None,
        "trials": trials,
    }


def main() -> None:
    task = FrostTask(seed=SEED)
    train_x, train_y = task.train_stream(TRAIN_N)
    validation_rows = json.loads(__import__("gzip").decompress(
        (ROOT / "resources" / "challenges" / "freeze-preparation" / "09-freeze" / "data.json.gz").read_bytes()
    ))["validation"]
    eval_indices = _validation_indices(task, validation_rows)

    data = {
        "schema_version": 1,
        "simulation": "Izhikevich neuron culture model; 64-electrode MEA; no living cells",
        "seed": SEED,
        "training_samples": TRAIN_N,
        "playback_trials_per_culture": PLAYBACK_N,
        "feature_names": [
            "tmin_lag3", "tmin_lag4", "tmin_lag7", "tmin_7d", "tmax_lag3", "tmax_lag4",
            "tmax_lag7", "tmax_7d", "season1", "season2", "season3", "season4", "lat", "elev",
        ],
        "level_range": [0, 4],
        "trial_duration_ms": 250,
        "conditions": [
            _run_culture("culture_reservoir", task, train_x, train_y, validation_rows, eval_indices),
            _run_culture("culture_frozen", task, train_x, train_y, validation_rows, eval_indices),
        ],
        "training_trial_duration_ms": codec.TRIAL_MS,
        "training_timeline_ms": {"stim": [0, codec.STIM_MS], "response": [0, codec.RESP_MS],
                                 "feedback": [codec.FB_START, codec.FB_START + codec.FB_MS], "rest": [codec.FB_START + codec.FB_MS, codec.TRIAL_MS]},
        "training": [
            _record_training("culture_fep", task, train_x, train_y),
            _record_training("culture_reservoir", task, train_x, train_y),
        ],
        "summary": _summary_numbers(),
    }

    stride = 1
    payload = json.dumps(data, separators=(",", ":"), ensure_ascii=False)
    while len(payload.encode("utf-8")) > MAX_JSON_BYTES:
        stride *= 2
        for condition in data["conditions"]:
            for trial in condition["trials"]:
                trial["spikes"] = trial["spikes"][::2]
        data["spike_sampling_stride"] = stride
        payload = json.dumps(data, separators=(",", ":"), ensure_ascii=False)
    OUT.write_text(payload + "\n", encoding="utf-8")
    print(f"Wrote {OUT} ({OUT.stat().st_size:,} bytes; spike stride {stride})")


if __name__ == "__main__":
    main()
