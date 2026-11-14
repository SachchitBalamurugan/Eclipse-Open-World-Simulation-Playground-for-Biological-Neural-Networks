"""Simulated reference recording: always a compressed .npz; also NWB 2.x when pynwb is installed
(`python -m bnn_platform.transfer to-nwb --bundle DIR` converts later)."""
from __future__ import annotations

import datetime as _dt
import json
import pathlib

import numpy as np

NPZ = "sim_reference.npz"
NWB = "sim_reference.nwb"
DOC = "sim_reference_README.md"

DOC_TEXT = """# sim_reference -- SIMULATED reference recording

Recorded by the exporter: the trained *simulated* culture driven by this bundle's own runtime (mock
backend, teaching off) on one held-out target, preceded by spontaneous activity with no stimulation.
It is NOT data from living neurons. Use it to compare a living culture's recordings with what the
simulation assumed (spontaneous rates per electrode, evoked responses after encoder pulses, motor-pool
counts per step, behaviour).

## sim_reference.npz (numpy.load)
| key | shape | meaning |
|---|---|---|
| spike_times_ms, spike_electrode, spike_sim_channel | (n_spikes,) | multi-unit spikes (all model neurons at an electrode) |
| stim_times_ms, stim_electrode, stim_amplitude, stim_kind | (n_stims,) | delivered pulses (kind: encode / teach_*) |
| segment_names, segment_start_ms, segment_stop_ms | (n_seg,) | 'spontaneous', 'episode' |
| step_t_ms, step_err_mm, step_action, step_counts, step_tip, step_target | per control step | trial start, error after the step, decoded action, spike counts per decoder input electrode, EE and target (m) |
| decoder_inputs | (n_inputs,) | electrode ids for step_counts columns |
| meta_json | () | JSON string: bundle, target, timing, units |

## Converting to NWB
`pip install pynwb` then `python -m bnn_platform.transfer to-nwb --bundle <bundle dir>`. Mapping:
electrodes -> NWB electrodes table (x/y in um, group = MEA, location = 'simulated culture'); spikes ->
Units table, one multi-unit per electrode (spike_times in s); pulses -> TimeIntervals 'stimulation'
(start/stop, electrode, amplitude, kind); behaviour -> processing/behavior TimeSeries (err_mm,
action, tip, target). Real recordings exported by vendor software (MaxLab, MCS DataManager ->
HDF5/NWB, CL recordings) can then be loaded with the same tools for side-by-side comparison.
"""


def write_reference_files(bundle_dir: pathlib.Path, b, rec, res: dict, spont_ms: int) -> list:
    sp = np.array(rec.spikes, dtype=np.int64).reshape(-1, 2)
    st = rec.stims
    steps = rec.steps
    meta = dict(bundle=bundle_dir.name, target=b.target, biology="simulated", timing=b.timing,
                units=dict(time="ms", position="m", err="mm"), episode_result=res)
    arrays = dict(
        spike_times_ms=sp[:, 0], spike_electrode=sp[:, 1],
        spike_sim_channel=np.array([b.sim_channel[int(e)] for e in sp[:, 1]], np.int64),
        stim_times_ms=np.array([s[0] for s in st], np.int64), stim_electrode=np.array([s[1] for s in st], np.int64),
        stim_amplitude=np.array([s[2] for s in st], float), stim_kind=np.array([s[3] for s in st], dtype="U16"),
        segment_names=np.array(["spontaneous", "episode"]), segment_start_ms=np.array([0, spont_ms]),
        segment_stop_ms=np.array([spont_ms, rec.t_ms]),
        step_t_ms=np.array([s["t_ms"] for s in steps]), step_err_mm=np.array([s["err_mm"] for s in steps]),
        step_action=np.array([s["action"] for s in steps]), step_counts=np.array([s["counts"] for s in steps]),
        step_tip=np.array([s["tip"] for s in steps]), step_target=np.array([s["target"] for s in steps]),
        decoder_inputs=np.array([int(e) for e in b.decoder["input"]["electrodes"]]),
        meta_json=np.array(json.dumps(meta)))
    np.savez_compressed(bundle_dir / NPZ, **arrays)
    (bundle_dir / DOC).write_text(DOC_TEXT, encoding="utf-8")
    files = [NPZ, DOC]
    try:
        import pynwb  # noqa: F401
    except ImportError:
        return files
    to_nwb(bundle_dir)
    return files + [NWB]


def to_nwb(bundle_dir: pathlib.Path) -> pathlib.Path:
    from pynwb import NWBHDF5IO, NWBFile, TimeSeries
    from pynwb.epoch import TimeIntervals
    import csv
    bundle_dir = pathlib.Path(bundle_dir)
    z = np.load(bundle_dir / NPZ, allow_pickle=False)
    meta = json.loads(str(z["meta_json"]))
    nwb = NWBFile(session_description="SIMULATED reference episode (Izhikevich culture model, not living neurons) "
                                      f"for bundle {meta['bundle']}",
                  identifier=f"bnn-sim-reference-{meta['bundle']}",
                  session_start_time=_dt.datetime.now(_dt.timezone.utc),
                  experiment_description="sim-to-real export reference; biology=simulated",
                  keywords=["simulated", "MEA", "closed-loop"])
    dev = nwb.create_device(name="simulated_MEA", description=f"target layout {meta['target']}")
    grp = nwb.create_electrode_group(name="MEA", description="simulated culture electrodes", location="simulated culture",
                                     device=dev)
    nwb.add_electrode_column(name="electrode_id", description="bundle logical electrode id")
    nwb.add_electrode_column(name="role", description="electrode_map.csv role")
    with open(bundle_dir / "electrode_map.csv", newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["role"] != "unavailable"]
    row_of = {}
    for i, r in enumerate(rows):
        nwb.add_electrode(x=float(r["x_um"]), y=float(r["y_um"]), z=0.0, group=grp, location="simulated culture",
                          electrode_id=int(r["electrode_id"]), role=r["role"])
        row_of[int(r["electrode_id"])] = i
    st, se = z["spike_times_ms"], z["spike_electrode"]
    for e, i in row_of.items():
        nwb.add_unit(spike_times=np.sort(st[se == e]) / 1000.0, electrodes=[i])
    ti = TimeIntervals(name="stimulation", description="delivered biphasic pulses (simulated)")
    ti.add_column(name="electrode", description="logical electrode id")
    ti.add_column(name="amplitude", description="per-phase amplitude (encoder.json units)")
    ti.add_column(name="kind", description="encode / teach_*")
    for t, e, a, k in zip(z["stim_times_ms"], z["stim_electrode"], z["stim_amplitude"], z["stim_kind"]):
        ti.add_interval(start_time=t / 1000.0, stop_time=t / 1000.0 + 0.001, electrode=int(e), amplitude=float(a),
                        kind=str(k))
    nwb.add_time_intervals(ti)
    beh = nwb.create_processing_module(name="behavior", description="robot behaviour per control step (sim)")
    ts = z["step_t_ms"] / 1000.0
    beh.add(TimeSeries(name="err_mm", data=z["step_err_mm"], unit="mm", timestamps=ts))
    beh.add(TimeSeries(name="action", data=z["step_action"], unit="normalised", timestamps=ts))
    beh.add(TimeSeries(name="tip", data=z["step_tip"], unit="m", timestamps=ts))
    beh.add(TimeSeries(name="target", data=z["step_target"], unit="m", timestamps=ts))
    out = bundle_dir / NWB
    with NWBHDF5IO(str(out), "w") as io:
        io.write(nwb)
    return out
