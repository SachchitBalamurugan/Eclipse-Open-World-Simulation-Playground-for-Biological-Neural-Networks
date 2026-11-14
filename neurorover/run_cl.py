"""Closed loop through the real Cortical Labs CL API (hardware-portable).

Runs the same trial protocol as bench.run_trial (codec.encode -> stim, count
spikes per electrode in the 0-50 ms response window, codec.decode, codec.feedback
stims), but every stim goes through `neurons.stim()` and every spike comes from
`tick.analysis.spikes`. With --sim (default) the Izhikevich culture is registered
as the CL SDK simulator source; with --hardware nothing is registered, so on a
CL1 / Cortical Cloud the identical loop drives living neurons.

Usage:
  python -m neurorover.run_cl --condition culture_reservoir --seed 0 --budget 25 --eval 100
  python -m neurorover.run_cl --condition culture_fep --budget 50 --record
  python -m neurorover.run_cl --condition culture_reservoir --budget 25 --eval 100 --direct   # reference harness
  python -m neurorover.run_cl ... --realtime      # disable accelerated time (sim only)

Differences from the direct harness (bench.py), by construction of real hardware:
  * Stims are pre-scheduled on the 25 kHz device clock (one-shot StimPlans,
    SDK 80 us lead) so a pulse for trial-ms t lands in culture ms t; spikes are
    binned by their 25 kHz timestamps, so the loop tick (default 5 ms) only sets
    latency, not resolution. Feedback is committed on the first tick after the
    response window closes (<= 1 tick before FB_START at 5 ms ticks).
  * The SDK enforces <=200 Hz per channel: colliding pulses (only possible in the
    random fep "wrong" feedback) are queued 5 ms later instead of summed.
  * Evaluation cannot deepcopy a living culture: eval trials run on the same
    culture, without feedback. For rule=fep/stdp plasticity keeps running during
    eval (as it would in a dish); frozen/reservoir are unaffected.
  * culture_rstdp's reward electrodes (60-63) are a modelling assumption; on
    hardware those stims are just ordinary stims.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import time

import numpy as np

from .culture.core import N_CH
from .env.frost import FrostTask, auc
from .interface import codec

BUDGETS = [25, 50, 100, 200]
RULE = {"culture_frozen": "frozen", "culture_fep": "fep", "culture_rstdp": "rstdp",
        "culture_reservoir": "frozen"}
FRAMES_PER_MS = 25
REC_DIR = pathlib.Path(__file__).resolve().parent / "recordings"


def eval_subset(task: FrostTask, n: int):
    """First n/2 positives + first n/2 negatives of the fixed stratified eval set."""
    half = len(task.yev) // 2
    k = min(n // 2, half)
    idx = np.r_[np.arange(k), half + np.arange(k)]
    return task.Xev[idx], task.yev[idx]


class CLTrialRunner:
    """Presents trials to `neurons` through `neurons.loop()`.

    Stims are pre-scheduled on the 25 kHz device clock with one-shot StimPlans
    (`plan.run(at_timestamp=...)`), so the loop can tick coarsely (default 5 ms)
    while every pulse still lands in its intended millisecond. The trial's t=0 is
    placed `lookahead` ticks in the future so all sensory pulses are committed
    before the device reaches them. Feedback is scheduled the first tick after the
    response window closes; any pulse whose time has already passed (or that is
    inside the tick the producer may already be generating) is counted in
    `late_stims` and fires as soon as possible.
    """

    def __init__(self, neurons, cl, rng, tick_ms: int = 5, lookahead_ticks: int = 2):
        self.n = neurons
        self.cl = cl
        self.rng = rng
        self.tick_ms = tick_ms
        self.tick_frames = tick_ms * FRAMES_PER_MS
        self.lookahead_frames = lookahead_ticks * self.tick_frames
        self.design = cl.StimDesign(160, -1.0, 160, 1.0)
        self.ticks = iter(neurons.loop(ticks_per_second=1000 / tick_ms, ignore_jitter=True))
        self.n_stims = 0
        self.late_stims = 0

    def _stim_at(self, frame: int, ch: int, cur: float, T: int):
        d = self.design if cur == 1.0 else self.cl.StimDesign(160, -abs(cur), 160, abs(cur))
        if frame < T + self.tick_frames:
            self.late_stims += 1
        plan = self.n.create_stim_plan()
        plan.stim(self.cl.ChannelSet(int(ch)), d)
        plan.run(at_timestamp=int(frame))  # +80 us lead -> lands 2 frames later, same ms
        self.n_stims += 1

    def idle(self, ms: int):
        for _ in range(max(1, ms // self.tick_ms)):
            next(self.ticks)

    def trial(self, levels, label=None, rule="frozen", total_ms=None):
        """Same contract as bench.run_trial: (counts[64], score, action, correct)."""
        total_ms = total_ms or (codec.TRIAL_MS if label is not None else codec.EVAL_TRIAL_MS)
        counts = np.zeros(N_CH)
        score = action = correct = None
        tick = next(self.ticks)
        T = tick.iteration_timestamp
        m0 = -(-(T + self.lookahead_frames) // FRAMES_PER_MS)  # culture ms of trial t=0
        f0 = m0 * FRAMES_PER_MS
        for t, ch in codec.encode(levels):
            self._stim_at(f0 + t * FRAMES_PER_MS, ch, 1.0, T)
        end_frame = f0 + total_ms * FRAMES_PER_MS
        resp_end_frame = f0 + codec.RESP_MS * FRAMES_PER_MS
        decided = False
        while True:
            tick = next(self.ticks)
            T = tick.iteration_timestamp
            # spikes detected over the previous tick (timestamps at 25 kHz)
            for s in tick.analysis.spikes:
                t = (s.timestamp - f0) // FRAMES_PER_MS
                if 0 <= t < codec.RESP_MS:
                    counts[s.channel] += 1
            if not decided and T >= resp_end_frame:
                score = codec.decode(counts)
                action = int(score > 0) if score != 0 else int(self.rng.random() < 0.5)
                if label is not None:
                    correct = action == label
                    for t, ch, cur in codec.feedback(rule, correct, self.rng):
                        self._stim_at(f0 + (codec.FB_START + t) * FRAMES_PER_MS, ch, cur, T)
                decided = True
            if decided and tick.iteration_next_timestamp >= end_frame:
                return counts, score, action, correct


def eval_set(run: CLTrialRunner, task, Xev):
    L = task.levels(Xev)
    counts = np.zeros((len(L), N_CH))
    scores = np.zeros(len(L))
    for i, lv in enumerate(L):
        counts[i], scores[i], _, _ = run.trial(lv)
    return counts, scores


def score_checkpoint(condition, tr_counts, ys, n, ev_counts, ev_scores):
    if condition == "culture_reservoir":
        from .bench import fit_logistic
        mu, sd = tr_counts[:n].mean(0), tr_counts[:n].std(0) + 1e-6
        f = fit_logistic((tr_counts[:n] - mu) / sd, ys[:n], lam=10.0)
        return f((ev_counts - mu) / sd)
    return ev_scores


def run_cl(condition, seed, budget, n_eval, sim=True, accelerated=True, record=False, tick_ms=5):
    os.environ.setdefault("CL_SDK_VISUALISATION", "0")
    if sim:
        os.environ["CL_SDK_ACCELERATED_TIME"] = "1" if accelerated else "0"
    import cl
    rule = RULE[condition]
    if sim:
        from .culture.izh_source import register
        # same construction as bench.run_job (fep runs plain STDP in the culture)
        register(seed=seed, rule="stdp" if rule == "fep" else rule)
    task = FrostTask(seed)
    rng = np.random.default_rng(seed)
    budgets = [b for b in BUDGETS if b <= budget] or [budget]
    Xs, ys = task.train_stream(max(budgets))
    Xev, yev = eval_subset(task, n_eval)
    L = task.levels(Xs)
    res = {"condition": condition, "seed": seed, "path": "cl_api", "sim": sim,
           "accelerated": accelerated and sim, "n_eval": len(yev), "auc": {}, "train_acc": {}}
    t0 = time.perf_counter()
    with cl.open() as neurons:
        rec = None
        if record:
            REC_DIR.mkdir(exist_ok=True)
            rec = neurons.record(file_location=str(REC_DIR),
                                 file_suffix=f"{condition}_s{seed}_b{budget}",
                                 include_raw_samples=False,
                                 attributes={"condition": condition, "seed": seed, "budget": budget})
        run = CLTrialRunner(neurons, cl, rng, tick_ms=tick_ms)
        ts0 = neurons.timestamp()
        tr_counts = np.zeros((len(L), N_CH))
        correct = []
        done = 0
        for n in budgets:
            for i in range(done, n):
                tr_counts[i], _, _, ok = run.trial(L[i], label=int(ys[i]), rule=rule)
                correct.append(bool(ok))
            done = n
            ev_counts, ev_scores = eval_set(run, task, Xev)
            s = score_checkpoint(condition, tr_counts, ys, n, ev_counts, ev_scores)
            res["auc"][n] = auc(s, yev)
            res["train_acc"][n] = float(np.mean(correct[-25:]))
            print(f"[cl] {condition} s{seed} n={n}: AUC={res['auc'][n]:.3f} "
                  f"train_acc={res['train_acc'][n]:.2f}  wall={time.perf_counter()-t0:.1f}s", flush=True)
        sim_s = (neurons.timestamp() - ts0) / neurons.get_frames_per_second()
        if rec is not None:
            rec.stop()
            res["recording"] = str(getattr(rec, "_file_path", REC_DIR))
    wall = time.perf_counter() - t0
    res.update(wall_s=round(wall, 1), sim_s=round(sim_s, 1),
               sim_s_per_wall_s=round(sim_s / wall, 2), n_stims=run.n_stims,
               late_stims=run.late_stims, tick_ms=tick_ms)
    return res


def run_direct(condition, seed, budget, n_eval):
    """Reference: bench.run_trial harness on the same eval subset."""
    import copy
    from .bench import run_trial
    from .culture.core import Culture
    rule = RULE[condition]
    task = FrostTask(seed)
    rng = np.random.default_rng(seed)
    budgets = [b for b in BUDGETS if b <= budget] or [budget]
    Xs, ys = task.train_stream(max(budgets))
    Xev, yev = eval_subset(task, n_eval)
    L, Lev = task.levels(Xs), task.levels(Xev)
    c = Culture(seed=seed, rule="stdp" if rule == "fep" else rule)
    c.run(1000)
    res = {"condition": condition, "seed": seed, "path": "direct", "n_eval": len(yev), "auc": {}}
    t0 = time.perf_counter()
    tr_counts = np.zeros((len(L), N_CH))
    done = 0
    for n in budgets:
        for i in range(done, n):
            tr_counts[i], _, _, _ = run_trial(c, L[i], rng, label=int(ys[i]), rule=rule)
        done = n
        c2 = copy.deepcopy(c); c2.rule = "frozen"; c2.elig = None
        ev_counts = np.zeros((len(Lev), N_CH)); ev_scores = np.zeros(len(Lev))
        for i, lv in enumerate(Lev):
            ev_counts[i], ev_scores[i], _, _ = run_trial(c2, lv, rng)
        s = score_checkpoint(condition, tr_counts, ys, n, ev_counts, ev_scores)
        res["auc"][n] = auc(s, yev)
        print(f"[direct] {condition} s{seed} n={n}: AUC={res['auc'][n]:.3f}", flush=True)
    res["wall_s"] = round(time.perf_counter() - t0, 1)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=list(RULE), default="culture_reservoir")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=int, default=25)
    ap.add_argument("--eval", type=int, default=300, help="eval rows (balanced subset of the 300)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--sim", dest="sim", action="store_true", default=True)
    g.add_argument("--hardware", dest="sim", action="store_false")
    ap.add_argument("--realtime", action="store_true", help="sim: disable accelerated time")
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--tick-ms", type=int, default=5, help="loop tick period (stims are pre-scheduled)")
    ap.add_argument("--direct", action="store_true", help="run the bench harness instead (reference)")
    ap.add_argument("--out", type=str, default=None, help="write result JSON here")
    a = ap.parse_args()
    if a.direct:
        r = run_direct(a.condition, a.seed, a.budget, a.eval)
    else:
        r = run_cl(a.condition, a.seed, a.budget, a.eval, sim=a.sim,
                   accelerated=not a.realtime, record=a.record, tick_ms=a.tick_ms)
    print(json.dumps(r))
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
