"""Train a neurorover model and export freeze-preparation held-out predictions."""
from __future__ import annotations

import argparse
import copy
import json
import math
import multiprocessing as mp
import pathlib
import time

import numpy as np

from . import bench
from .culture.core import Culture
from .env.frost import ENCODED, FrostTask, load


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_OUTPUTS = {
    "culture_reservoir": ROOT / "neurorover" / "submissions" / "freeze_culture_reservoir.json",
    "logistic": ROOT / "neurorover" / "submissions" / "freeze_logistic.json",
}
CHUNK_SIZE = 128

_WORKER_CULTURE: Culture | None = None
_WORKER_RNG: np.random.Generator | None = None
_WORKER_LEVELS: np.ndarray | None = None


def train_model(model: str, budget: int, seed: int):
    """Train with the same stream, settling, trials, and readout as bench.run_job."""
    task = FrostTask(seed)
    rng = np.random.default_rng(seed)
    Xs, ys = task.train_stream(budget)
    levels = task.levels(Xs)

    if model == "logistic":
        return task, None, None, None, bench.fit_logistic(levels, ys), None

    rule = bench.RULE["culture_reservoir"]
    culture = Culture(seed=seed, rule="frozen")
    culture.run(1000)  # settle, matching bench.run_job
    tr_counts = np.zeros((len(levels), bench.N_CH))
    for i in range(budget):
        tr_counts[i], _, _, _ = bench.run_trial(
            culture, levels[i], rng, label=int(ys[i]), rule=rule
        )
    mu = tr_counts[:budget].mean(0)
    sd = tr_counts[:budget].std(0) + 1e-6
    readout = bench.fit_logistic((tr_counts[:budget] - mu) / sd, ys[:budget], lam=10.0)
    return task, culture, mu, sd, readout, tr_counts


def _init_worker(culture: Culture, levels: np.ndarray, seed: int):
    global _WORKER_CULTURE, _WORKER_RNG, _WORKER_LEVELS
    # Each process starts from the same trained frozen state, then carries it
    # across the chunks assigned to that process.
    _WORKER_CULTURE = copy.deepcopy(culture)
    _WORKER_CULTURE.rule = "frozen"
    _WORKER_CULTURE.elig = None
    _WORKER_RNG = np.random.default_rng(seed)
    _WORKER_LEVELS = levels


def _infer_chunk(bounds: tuple[int, int]):
    if _WORKER_CULTURE is None or _WORKER_RNG is None or _WORKER_LEVELS is None:
        raise RuntimeError("inference worker was not initialized")
    start, end = bounds
    counts = np.zeros((end - start, bench.N_CH))
    for offset, i in enumerate(range(start, end)):
        counts[offset], _, _, _ = bench.run_trial(
            _WORKER_CULTURE, _WORKER_LEVELS[i], _WORKER_RNG
        )
    return start, counts


def infer_culture(culture: Culture, levels: np.ndarray, mu: np.ndarray,
                  sd: np.ndarray, readout, procs: int, seed: int) -> np.ndarray:
    n = len(levels)
    logits = np.empty(n, dtype=float)
    chunks = [(i, min(i + CHUNK_SIZE, n)) for i in range(0, n, CHUNK_SIZE)]
    if procs == 1:
        _init_worker(culture, levels, seed)
        results = map(_infer_chunk, chunks)
        pool = None
    else:
        pool = mp.get_context("spawn").Pool(
            processes=procs,
            initializer=_init_worker,
            initargs=(culture, levels, seed),
        )
        results = pool.imap_unordered(_infer_chunk, chunks, chunksize=1)

    started = time.perf_counter()
    last_report = started
    done = 0
    try:
        for start, counts in results:
            end = start + len(counts)
            logits[start:end] = readout((counts - mu) / sd)
            done += len(counts)
            now = time.perf_counter()
            if now - last_report >= 15:
                print(f"inferred {done}/{n} cases ({now - started:.1f}s)", flush=True)
                last_report = now
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    return logits


def sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))


def make_submission(cases: list[dict], probabilities: np.ndarray,
                    model: str, budget: int, seed: int, limit: int | None,
                    procs: int) -> dict:
    by_date: dict[str, list[int]] = {}
    for i, case in enumerate(cases):
        by_date.setdefault(case["date"], []).append(i)
    review = np.zeros(len(cases), dtype=bool)
    for indices in by_date.values():
        count = math.ceil(0.2 * len(indices))
        chosen = sorted(indices, key=lambda i: (-probabilities[i], cases[i]["id"]))[:count]
        review[chosen] = True

    rows = [
        {
            "id": case["id"],
            "probability": float(probabilities[i]),
            "review": bool(review[i]),
            "status": "ok",
            "last_input_date": case["last_input_date"],
        }
        for i, case in enumerate(cases)
    ]
    if model == "culture_reservoir":
        model_name = "neurorover-culture-reservoir"
        training = (
            f"Participant data.json train only; class-balanced stream of {budget} samples. "
            "A simulated Izhikevich culture (not living neurons) was settled for 1000 ms "
            "and trained with the neurorover frozen rule; a logistic readout was fit to "
            "training spike counts."
        )
        hyperparameters = {
            "budget": budget,
            "seed": seed,
            "culture": "simulated Izhikevich culture; 64 electrodes x 16 neurons",
            "culture_rule": "frozen",
            "settle_ms": 1000,
            "readout": "bench.fit_logistic on standardized training spike counts",
            "readout_lambda": 10.0,
            "probability": "sigmoid of readout logit",
            "inference_procs": procs,
            "limit": limit,
        }
    else:
        model_name = "neurorover-logistic"
        training = (
            f"Participant data.json train only; class-balanced stream of {budget} samples. "
            "bench.fit_logistic was fit to task.levels of the selected weather features."
        )
        hyperparameters = {
            "budget": budget,
            "seed": seed,
            "features": "task.levels (train-quantile pulse levels)",
            "readout": "bench.fit_logistic",
            "probability": "sigmoid of readout logit",
            "limit": limit,
        }
    return {
        "packet_version": "0.5.0-internal",
        "route": "09-freeze",
        "model": model_name,
        "provenance": {
            "training": training,
            "external_models": [],
            "hyperparameters": hyperparameters,
            "pre_event_work": "Existing neurorover simulator and benchmark code reused; no external models used.",
        },
        "rows": rows,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--budget", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--limit", type=int, help="infer only the first N held-out cases (quick run)")
    ap.add_argument("--model", choices=["culture_reservoir", "logistic"], default="culture_reservoir")
    ap.add_argument("--output", type=pathlib.Path)
    args = ap.parse_args()
    if args.budget < 1 or args.budget > 500:
        ap.error("--budget must be between 1 and 500")
    if args.procs < 1:
        ap.error("--procs must be at least 1")
    if args.limit is not None and args.limit < 1:
        ap.error("--limit must be at least 1")

    started = time.perf_counter()
    print(f"training {args.model}: budget={args.budget}, seed={args.seed}", flush=True)
    task, culture, mu, sd, readout, _ = train_model(args.model, args.budget, args.seed)
    data = load()
    cases = data["cases"]
    if args.limit is not None:
        cases = cases[:args.limit]
    X = np.array([case["features"] for case in cases], dtype=float)[:, ENCODED]
    levels = task.levels(X)

    infer_started = time.perf_counter()
    if args.model == "logistic":
        logits = readout(levels)
    else:
        logits = infer_culture(culture, levels, mu, sd, readout, args.procs, args.seed)
    probabilities = sigmoid(logits)
    submission = make_submission(cases, probabilities, args.model, args.budget,
                                 args.seed, args.limit, args.procs)

    output = args.output or DEFAULT_OUTPUTS[args.model]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(submission, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        f"wrote {len(submission['rows'])} rows to {output}; "
        f"inference={time.perf_counter() - infer_started:.1f}s, total={time.perf_counter() - started:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()
