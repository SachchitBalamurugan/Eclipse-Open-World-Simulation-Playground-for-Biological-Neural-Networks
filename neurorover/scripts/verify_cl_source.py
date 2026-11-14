"""Verify the Izhikevich culture through the CL API: reactivity, latency, speed.

Stimulates electrode 5 (via StimPlan at exact device timestamps) on alternating
100 ms epochs; the other epochs are sham (no stim). Counts spikes on electrodes
4,5,6,13 in the 20 ms after each (real or sham) stim time and measures
stim->first-evoked-spike latency from the stim's own committed timestamp.

  python -m neurorover.scripts.verify_cl_source [--epochs 100] [--realtime]
"""
import argparse
import os
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--realtime", action="store_true")
    a = ap.parse_args()
    os.environ["CL_SDK_VISUALISATION"] = "0"
    os.environ["CL_SDK_ACCELERATED_TIME"] = "0" if a.realtime else "1"
    import cl
    from neurorover.culture.izh_source import register
    register(seed=0, rule="frozen")
    CH, NEIGH, WIN = 5, [4, 5, 6, 13], 20 * 25  # 20 ms window in frames
    EPOCH = 100 * 25
    design = cl.StimDesign(160, -1.0, 160, 1.0)
    with cl.open() as n:
        t_start = None
        events = []           # (frame, is_stim)
        stim_ts = []          # committed stim timestamps seen by analysis
        spikes = []           # (ts, ch) on NEIGH
        t0 = time.perf_counter()
        ts0 = n.timestamp()
        for tick in n.loop(ticks_per_second=200, ignore_jitter=True):
            T = tick.iteration_timestamp
            if t_start is None:
                t_start = T + 4 * 125
                for k in range(a.epochs):
                    f = t_start + k * EPOCH
                    is_stim = k % 2 == 0
                    events.append((f, is_stim))
                    if is_stim:
                        p = n.create_stim_plan()
                        p.stim(cl.ChannelSet(CH), design)
                        p.run(at_timestamp=f)
            stim_ts += [s.timestamp for s in tick.analysis.stims if s.channel == CH]
            spikes += [(s.timestamp, s.channel) for s in tick.analysis.spikes if s.channel in NEIGH]
            if T > t_start + a.epochs * EPOCH + WIN:
                break
        wall = time.perf_counter() - t0
        sim = (n.timestamp() - ts0) / 25000
    sp = np.array(spikes)
    stim_ts = np.array(sorted(stim_ts))
    res = {"stim": {c: [] for c in NEIGH}, "sham": {c: [] for c in NEIGH}}
    lat = []
    for (f, is_stim) in events:
        onset = f
        if is_stim:
            j = np.searchsorted(stim_ts, f)
            onset = int(stim_ts[j]) if j < len(stim_ts) else f
        m = (sp[:, 0] >= onset) & (sp[:, 0] < onset + WIN)
        for c in NEIGH:
            res["stim" if is_stim else "sham"][c].append(int((m & (sp[:, 1] == c)).sum()))
        if is_stim:
            m5 = m & (sp[:, 1] == CH)
            if m5.any():
                lat.append((sp[m5, 0].min() - onset) / 25.0)
    print(f"stims committed on ch{CH}: {len(stim_ts)} (scheduled {sum(e[1] for e in events)}); "
          f"commit offset vs requested: {np.mean(stim_ts - np.array([e[0] for e in events if e[1]][:len(stim_ts)])):.1f} frames")
    for c in NEIGH:
        s, h = np.mean(res["stim"][c]), np.mean(res["sham"][c])
        print(f"  electrode {c:2d}: stim {s:6.2f} vs sham {h:5.2f} spikes / 20 ms")
    if lat:
        print(f"  stim->first spike on ch{CH}: median {np.median(lat):.2f} ms, "
              f"min {np.min(lat):.2f}, max {np.max(lat):.2f} (sim ms; n={len(lat)})")
    print(f"speed: {sim:.1f} sim-s in {wall:.1f} wall-s = {sim / wall:.2f}x "
          f"({'real time' if a.realtime else 'accelerated'})")


if __name__ == "__main__":
    main()
