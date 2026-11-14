"""Generate learning-curve and simulated-culture raster figures.

Run from the repository root with:
    .venv/Scripts/python -m neurorover.analysis.plots

Learning curves re-read neurorover/results on every invocation.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "neurorover" / "results"
OUTDIR = Path(__file__).resolve().parent
BUDGETS = (25, 50, 100, 200)
CULTURE_CONDS = (
    "culture_frozen",
    "culture_fep",
    "culture_rstdp",
    "culture_reservoir",
    "culture_teach",
)
PALETTE = {
    "culture_frozen": "#0072B2",
    "culture_fep": "#D55E00",
    "culture_rstdp": "#009E73",
    "culture_reservoir": "#CC79A7",
    "culture_teach": "#F0E442",
    "heuristic_tmin": "#E69F00",
    "logistic": "#56B4E9",
    "mlp": "#6B6B6B",
    "logistic_rawfeat": "#8C6D31",
    "random": "#222222",
}
N_BOOT = 20_000


def _auc_values(record: dict) -> dict[int, float]:
    values: dict[int, float] = {}
    nested = record.get("auc", {})
    if isinstance(nested, dict):
        for key, value in nested.items():
            match = re.search(r"\d+", str(key))
            if match:
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(number):
                    values[int(match.group())] = number
    for key, value in record.items():
        match = re.fullmatch(r"auc[_-]?(\d+)", str(key), flags=re.IGNORECASE)
        if match:
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                values[int(match.group(1))] = number
    return values


def load_results() -> dict[str, dict[int, dict[int, float]]]:
    """Take a fresh snapshot; ignore files caught during an in-progress write."""
    data: dict[str, dict[int, dict[int, float]]] = {}
    if not RESULTS.exists():
        return data
    for path in sorted(RESULTS.glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            condition = record.get("condition")
            seed = int(record["seed"])
            aucs = _auc_values(record)
            if condition and aucs and not record.get("tag"):  # skip hyperparameter variants
                data.setdefault(str(condition), {})[seed] = aucs
        except (OSError, ValueError, TypeError, KeyError, AttributeError, json.JSONDecodeError):
            continue
    return data


def _rng(tag: str) -> np.random.Generator:
    seed = 0x5EED
    for byte in tag.encode("utf-8"):
        seed = (seed * 131 + byte) % (2**32)
    return np.random.default_rng(seed)


def mean_ci(values, tag: str):
    x = np.asarray(values, dtype=float)
    if len(x) == 0:
        return None
    mean = float(x.mean())
    if len(x) < 2:
        return mean, None
    draws = _rng(tag).integers(0, len(x), size=(N_BOOT, len(x)))
    boot = x[draws].mean(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return mean, (float(lo), float(hi))


def _plot_series(ax, data, condition: str, budgets, *, linestyle, color, label=None):
    xs, means, lows, highs = [], [], [], []
    for budget in budgets:
        vals = [
            row[budget]
            for row in data.get(condition, {}).values()
            if budget in row
        ]
        result = mean_ci(vals, f"curve:{condition}:{budget}")
        if result is None:
            continue
        mean, ci = result
        xs.append(budget)
        means.append(mean)
        lows.append(mean if ci is None else ci[0])
        highs.append(mean if ci is None else ci[1])
    if not xs:
        return None
    ax.plot(
        xs,
        means,
        marker="o",
        linewidth=2.1,
        markersize=5,
        linestyle=linestyle,
        color=color,
        label=label or condition,
    )
    if any(lo != hi for lo, hi in zip(lows, highs)):
        ax.fill_between(xs, lows, highs, color=color, alpha=0.14, linewidth=0)
    return xs, means, lows, highs


def make_learning_curves(data):
    fig, ax = plt.subplots(figsize=(10.2, 6.2), constrained_layout=True)
    for condition in CULTURE_CONDS:
        _plot_series(
            ax,
            data,
            condition,
            BUDGETS,
            linestyle="-",
            color=PALETTE[condition],
        )
    # The heuristic uses the same feature score at every budget. Pool the
    # available per-seed checkpoint AUCs, then draw that score as a flat line.
    heuristic_rows = data.get("heuristic_tmin", {})
    per_seed = [
        float(np.mean([row[b] for b in BUDGETS if b in row]))
        for row in heuristic_rows.values()
        if any(b in row for b in BUDGETS)
    ]
    heuristic = mean_ci(per_seed, "curve:heuristic_tmin:flat")
    if heuristic is not None:
        mean, ci = heuristic
        ax.axhline(
            mean,
            color=PALETTE["heuristic_tmin"],
            linestyle="--",
            linewidth=2.0,
            label="heuristic_tmin",
        )
        if ci is not None:
            ax.axhspan(ci[0], ci[1], color=PALETTE["heuristic_tmin"], alpha=0.12)
    for condition in ("logistic", "mlp", "logistic_rawfeat"):
        _plot_series(
            ax,
            data,
            condition,
            BUDGETS,
            linestyle="--",
            color=PALETTE[condition],
        )
    ax.axhline(
        0.5,
        color=PALETTE["random"],
        linestyle=":",
        linewidth=1.8,
        label="random (AUC = 0.50)",
    )
    ax.set_xscale("log", base=2)
    ax.set_xticks(BUDGETS, labels=[str(v) for v in BUDGETS])
    ax.set_xlim(22, 225)
    ax.set_ylim(0.3, 0.95)
    ax.set_xlabel("Training samples")
    ax.set_ylabel("Validation AUC")
    ax.set_title("Simulated culture learning curves on Frost")
    ax.grid(True, which="major", color="#D9DEE5", linewidth=0.8)
    ax.grid(True, which="minor", axis="x", color="#ECEFF3", linewidth=0.5)
    ax.legend(frameon=False, ncol=2, loc="best")
    fig.savefig(OUTDIR / "fig_learning_curves.png", dpi=200, facecolor="white")
    plt.close(fig)


def make_raster():
    from neurorover.bench import run_trial
    from neurorover.culture.core import (
        FEEDBACK,
        N_CH,
        SENSORY,
        Culture,
    )
    from neurorover.env.frost import FrostTask

    task = FrostTask(seed=0)
    X, y = task.train_stream(5)
    levels = task.levels(X)
    rng = np.random.default_rng(0)
    culture = Culture(seed=0)

    spike_times: list[int] = []
    spike_channels: list[int] = []
    stim_events: list[tuple[int, int]] = []
    original_step = culture.step
    original_stim = culture.stim

    def recording_step():
        t = culture.t_ms
        fired = original_step()
        if fired.size:
            spike_times.extend([t] * int(fired.size))
            spike_channels.extend(culture.ch[fired].astype(int).tolist())
        return fired

    def recording_stim(channel, current_uA=1.0):
        stim_events.append((culture.t_ms, int(channel)))
        return original_stim(channel, current_uA)

    culture.step = recording_step
    culture.stim = recording_stim
    culture.run(3000)
    trial_starts = []
    for i in range(5):
        trial_starts.append(culture.t_ms)
        # FEP feedback makes feedback electrode stim onsets visible; the
        # default Culture(seed=0) keeps its weights frozen for the raster.
        run_trial(
            culture,
            levels[i],
            rng,
            label=int(y[i]),
            rule="fep",
        )

    fig, ax = plt.subplots(figsize=(14.4, 7.2), constrained_layout=True)
    band_colors = {
        "sensory": "#F3D9A6",
        "motor": "#C9DDED",
        "feedback": "#E8CDE2",
    }
    ax.axhspan(-0.5, 15.5, color=band_colors["sensory"], alpha=0.42, zorder=0)
    ax.axhspan(39.5, 55.5, color=band_colors["motor"], alpha=0.38, zorder=0)
    ax.axhspan(55.5, 59.5, color=band_colors["feedback"], alpha=0.48, zorder=0)
    if spike_times:
        ax.scatter(
            np.asarray(spike_times) / 1000.0,
            spike_channels,
            s=2.2,
            color="#243447",
            alpha=0.48,
            linewidths=0,
            rasterized=True,
            zorder=2,
        )
    for start in trial_starts:
        ax.axvline(
            start / 1000.0,
            color="#59636E",
            linestyle="--",
            linewidth=0.8,
            alpha=0.48,
            zorder=1,
        )

    sensory_times = sorted({t for t, ch in stim_events if ch in SENSORY})
    feedback_times = sorted({t for t, ch in stim_events if ch in FEEDBACK})
    ax.vlines(
        np.asarray(sensory_times) / 1000.0,
        63.35,
        64.15,
        color="#B97800",
        linewidth=1.0,
        zorder=3,
    )
    ax.vlines(
        np.asarray(feedback_times) / 1000.0,
        63.35,
        64.15,
        color="#A44B88",
        linewidth=1.0,
        zorder=3,
    )
    handles = [
        Line2D([], [], marker=".", linestyle="None", color="#243447",
               markersize=6, label="neuron spike"),
        Line2D([], [], marker="|", linestyle="None", color="#B97800",
               markersize=10, label="sensory stim onset"),
        Line2D([], [], marker="|", linestyle="None", color="#A44B88",
               markersize=10, label="feedback stim onset"),
        Line2D([], [], linestyle="--", color="#59636E",
               label="encoded trial start"),
        Patch(facecolor=band_colors["sensory"], alpha=0.65, label="sensory electrodes 0–15"),
        Patch(facecolor=band_colors["motor"], alpha=0.65, label="motor electrodes 40–55"),
        Patch(facecolor=band_colors["feedback"], alpha=0.75, label="feedback electrodes 56–59"),
    ]
    ax.set_xlim(0, culture.t_ms / 1000.0)
    ax.set_ylim(-0.5, 64.8)
    ax.set_yticks(np.arange(N_CH))
    ax.set_yticklabels([str(i) if i % 4 == 0 else "" for i in range(N_CH)])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Electrode index (0–63; stimulus ticks above 63)")
    ax.set_title("Simulated culture spontaneous activity and five Frost trials")
    ax.grid(axis="x", color="#E1E5EA", linewidth=0.7)
    ax.legend(
        handles=handles,
        frameon=False,
        ncol=1,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize=9,
    )
    fig.savefig(OUTDIR / "fig_raster.png", dpi=200, facecolor="white")
    plt.close(fig)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    data = load_results()
    make_learning_curves(data)
    make_raster()
    print(f"Wrote {OUTDIR / 'fig_learning_curves.png'}")
    print(f"Wrote {OUTDIR / 'fig_raster.png'}")
    print(f"Learning curves used a fresh snapshot of {RESULTS}")


if __name__ == "__main__":
    main()
