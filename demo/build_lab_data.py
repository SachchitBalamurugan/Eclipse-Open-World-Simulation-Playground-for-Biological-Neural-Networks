"""Build compact JSON for demo/lab.html from recorded culture runs.

Reads runs/<condition>_s<seed>/ (written by neurorover/record_run.py). If no runs
exist yet, it generates equivalent CSVs itself (SIMULATED culture, no living cells)
into demo/lab/raw/. Raw CSVs are copied to demo/lab/raw/<run>/ for download.
Re-runnable:  .venv/Scripts/python demo/build_lab_data.py [--regen] [--budget 50 --eval 20]
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
LAB = ROOT / "demo" / "lab"
RAW = LAB / "raw"
RUNS = ROOT / "runs"
FILES = ["spikes.csv", "stims.csv", "trials.csv", "electrode_rates.csv", "weights.csv", "meta.json"]
WINDOWS = ["stim", "response", "feedback", "rest"]
KINDS = {"sensory": 0, "feedback": 1, "reward": 2}


def generate(condition: str, seed: int, budget: int, eval_n: int, out: pathlib.Path):
    """Run the simulated culture and log CSVs in the same format as record_run.py."""
    import numpy as np
    from neurorover import bench
    from neurorover.culture.core import Culture, MOTOR_DEPLOY, MOTOR_HOLD, N_CH, REWARD
    from neurorover.env.frost import FrostTask
    from neurorover.interface import codec

    out.mkdir(parents=True, exist_ok=True)
    task = FrostTask(seed)
    rng = np.random.default_rng(seed)
    Xs, ys = task.train_stream(max(bench.BUDGETS))
    rule = bench.RULE[condition]
    plastic = rule in ("fep", "rstdp")
    c = Culture(seed=seed, rule="stdp" if rule == "fep" else rule)
    W0 = c.W.copy()
    c.run(1000)
    st = {"active": False}
    fs = {n: (out / n).open("w", newline="", encoding="utf-8") for n in FILES[:5]}
    w = {n: csv.writer(f) for n, f in fs.items()}
    w["spikes.csv"].writerow(["trial", "phase", "t_ms", "channel"])
    w["stims.csv"].writerow(["trial", "phase", "t_ms", "channel", "kind"])
    w["trials.csv"].writerow(["trial", "phase", "station_label", "action", "correct", "score",
                              "deploy_spikes", "hold_spikes", "total_spikes"])
    w["electrode_rates.csv"].writerow(["trial", "phase", "window"] + [f"ch{i}" for i in range(N_CH)])
    w["weights.csv"].writerow(["trial", "mean_W_sens_to_deploy", "mean_W_sens_to_hold", "mean_abs_dW_since_start"])
    orig_step, orig_stim = Culture.step, Culture.stim

    def step(self):
        fired = orig_step(self)
        if st["active"] and fired.size:
            t = int(self.t_ms - st["t0"]) - 1
            chs = self.ch[fired].astype(int)
            for ch in chs:
                w["spikes.csv"].writerow([st["trial"], st["phase"], t, int(ch)])
            cnt = np.bincount(chs, minlength=N_CH)
            st["total"] += int(fired.size)
            if t < codec.STIM_MS: st["win"][0] += cnt
            if t < codec.RESP_MS: st["win"][1] += cnt
            if codec.FB_START <= t < codec.FB_START + codec.FB_MS: st["win"][2] += cnt
            if t >= codec.FB_START + codec.FB_MS: st["win"][3] += cnt
        return fired

    def stim(self, channel, current_uA=1.0):
        if st["active"]:
            t = int(self.t_ms - st["t0"])
            ch = int(channel)
            kind = "reward" if ch in REWARD else ("sensory" if t < codec.STIM_MS else "feedback")
            w["stims.csv"].writerow([st["trial"], st["phase"], t, ch, kind])
        return orig_stim(self, channel, current_uA)

    Culture.step, Culture.stim = step, stim
    dmask = c.plastic & np.isin(c.ch[None, :], MOTOR_DEPLOY)
    hmask = c.plastic & np.isin(c.ch[None, :], MOTOR_HOLD)

    def trial(cul, i, phase, lv, label, rl):
        st.update(active=True, t0=cul.t_ms, trial=i, phase=phase, total=0, win=np.zeros((4, N_CH), int))
        counts, score, action, correct = bench.run_trial(cul, lv, rng, label=label, rule=rl)
        st["active"] = False
        for name, row in zip(WINDOWS, st["win"]):
            w["electrode_rates.csv"].writerow([i, phase, name] + row.tolist())
        d, h = int(counts[MOTOR_DEPLOY].sum()), int(counts[MOTOR_HOLD].sum())
        return counts, score, action, d, h, st["total"]

    try:
        L = task.levels(Xs)
        for i in range(budget):
            _, score, action, d, h, tot = trial(c, i, "train", L[i], int(ys[i]), rule)
            w["trials.csv"].writerow([i, "train", int(ys[i]), action, int(action == int(ys[i])), score, d, h, tot])
            if plastic:
                w["weights.csv"].writerow([i, float(c.W[dmask].mean()), float(c.W[hmask].mean()),
                                           float(np.abs(c.W[c.plastic] - W0[c.plastic]).mean())])
        import copy
        c2 = copy.deepcopy(c); c2.rule = "frozen"; c2.elig = None
        idx = np.concatenate((np.arange(eval_n // 2), 150 + np.arange(eval_n - eval_n // 2)))
        Le = task.levels(task.Xev[idx])
        for j, lv in enumerate(Le):
            lab = int(task.yev[idx][j])
            _, score, action, d, h, tot = trial(c2, j, "eval", lv, None, "frozen")
            w["trials.csv"].writerow([j, "eval", lab, action, int(action == lab), score, d, h, tot])
    finally:
        Culture.step, Culture.stim = orig_step, orig_stim
        for f in fs.values():
            f.close()
    if not plastic:
        (out / "weights.csv").unlink()
    meta = {"condition": condition, "seed": seed, "budget": budget, "eval": eval_n,
            "generated_by": "demo/build_lab_data.py (fallback; runs/ absent)",
            "note": "Simulated Izhikevich culture model. No living neurons."}
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def rows(p):
    with p.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def num(v):
    try:
        x = float(v)
        return int(x) if x.is_integer() else round(x, 4)
    except (TypeError, ValueError):
        return v


def convert(d: pathlib.Path, max_spikes_per_trial=6000):
    meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
    trials = [{k: num(v) for k, v in r.items()} for r in rows(d / "trials.csv")]
    key = lambda r: f"{r['phase']}:{int(float(r['trial']))}"
    order = {key(t): i for i, t in enumerate(trials)}
    sp = [[] for _ in trials]
    for r in rows(d / "spikes.csv"):
        i = order.get(key(r))
        if i is not None and len(sp[i]) < max_spikes_per_trial:
            sp[i].append(int(float(r["t_ms"])) * 64 + int(r["channel"]))
    stm = [[] for _ in trials]
    if (d / "stims.csv").exists():
        for r in rows(d / "stims.csv"):
            i = order.get(key(r))
            if i is not None:
                stm[i].append((int(float(r["t_ms"])) * 64 + int(r["channel"])) * 4 + KINDS.get(r["kind"], 0))
    rates = [[None] * 4 for _ in trials]
    if (d / "electrode_rates.csv").exists():
        for r in rows(d / "electrode_rates.csv"):
            i = order.get(key(r))
            if i is not None and r["window"] in WINDOWS:
                rates[i][WINDOWS.index(r["window"])] = [int(float(r[f"ch{c}"])) for c in range(64)]
    weights = [{k: num(v) for k, v in r.items()} for r in rows(d / "weights.csv")] if (d / "weights.csv").exists() else []
    for t, s, m, r in zip(trials, sp, stm, rates):
        s.sort(); m.sort()
        t["spk"], t["stm"], t["rates"] = s, m, r
    return {"id": d.name, "meta": meta, "trials": trials, "weights": weights,
            "files": [f for f in FILES if (d / f).exists()]}


def benchmark():
    out = {}
    p = ROOT / "neurorover" / "analysis" / "summary.md"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            m = re.match(r"\|\s*([a-z_]+)\s*\|(.*)", line)
            if m and m.group(1) != "condition":
                cells = [c.strip() for c in m.group(2).split("|")]
                vals = [re.match(r"([0-9.]+)", c) for c in cells]
                v = [float(x.group(1)) for x in vals if x]
                if v and m.group(1) not in out:
                    out[m.group(1)] = v[-1]
    return {"auc_at_max_budget": out, "source": "neurorover/analysis/summary.md"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regen", action="store_true")
    ap.add_argument("--budget", type=int, default=50)
    ap.add_argument("--eval", type=int, default=20)
    a = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    srcs = sorted(p for p in RUNS.glob("*_s*") if (p / "trials.csv").exists()) if RUNS.exists() else []
    for s in srcs:
        dst = RAW / s.name
        dst.mkdir(exist_ok=True)
        for f in FILES:
            if (s / f).exists():
                shutil.copy2(s / f, dst / f)
    if not srcs:
        for cond in ("culture_reservoir", "culture_fep"):
            dst = RAW / f"{cond}_s0"
            if a.regen or not (dst / "trials.csv").exists():
                print("generating", dst.name, flush=True)
                generate(cond, 0, a.budget, a.eval, dst)
    names = [s.name for s in srcs] or sorted(p.name for p in RAW.iterdir() if (p / "trials.csv").exists())
    index = {"runs": [], "benchmark": benchmark(), "source": "runs/" if srcs else "generated fallback"}
    for n in names:
        data = convert(RAW / n)
        (LAB / f"run_{n}.json").write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
        index["runs"].append({"id": n, "condition": data["meta"].get("condition", n.rsplit("_s", 1)[0]),
                              "seed": data["meta"].get("seed", n.rsplit("_s", 1)[-1]), "n_trials": len(data["trials"]),
                              "plastic": bool(data["weights"])})
    (LAB / "index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    total = sum(p.stat().st_size for p in LAB.glob("*.json"))
    print(f"wrote {len(names)} runs, json total {total/1e6:.2f} MB")


if __name__ == "__main__":
    main()
