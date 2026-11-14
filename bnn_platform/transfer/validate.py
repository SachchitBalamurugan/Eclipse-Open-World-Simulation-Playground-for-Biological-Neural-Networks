"""Validate a bundle: structure/schema, internal consistency, and stimulation/robot safety bounds."""
from __future__ import annotations

import csv
import json
import pathlib

from . import SCHEMA_VERSION
from .source import ROOT
from .targets import TARGETS

REQUIRED = ["manifest.json", "electrode_map.csv", "encoder.json", "decoder.json", "training_protocol.json",
            "stim_schedule.csv", "calibration_protocol.json", "safety.json", "README.md"]
KEYS = {
    "manifest.json": ["schema_version", "biology_of_source", "source_run", "robot", "task", "observation_space",
                      "action_space", "electrodes", "timing", "target", "provenance", "warnings"],
    "encoder.json": ["schema_version", "channels", "pulse_timing", "waveform"],
    "decoder.json": ["schema_version", "type", "input", "output", "weights", "bias", "input_offset", "deadzone",
                     "gain", "refit"],
    "training_protocol.json": ["schema_version", "teach", "teacher", "session_plan", "controls"],
    "calibration_protocol.json": ["schema_version", "baseline", "probe", "remap", "decoder_refit",
                                  "mock_compressed"],
    "safety.json": ["schema_version", "status", "stimulation", "robot"],
}
MAP_COLS = ["electrode_id", "role", "role_name", "pool", "joint", "sign", "grid_x", "grid_y", "x_um", "y_um",
            "sim_channel"]
ROLES = {"sensory", "motor", "reward", "feedback", "unused", "unavailable"}


def validate(bundle) -> dict:
    p = pathlib.Path(bundle)
    if not p.is_absolute() and not p.exists():
        p = ROOT / p
    p = p.resolve()
    checks, errors, warnings = [], [], []

    def check(name, ok, detail="", warn=False):
        checks.append(dict(name=name, ok=bool(ok), detail=detail, level="warning" if warn else "error"))
        if not ok:
            (warnings if warn else errors).append(f"{name}: {detail}")
        return ok

    missing = [f for f in REQUIRED if not (p / f).exists()]
    if not check("files_present", not missing, f"missing {missing}" if missing else "all required files"):
        return dict(ok=False, bundle=(p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else str(p)), errors=errors, warnings=warnings, checks=checks)
    J = {}
    for f, keys in KEYS.items():
        try:
            J[f] = json.loads((p / f).read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            check(f"json:{f}", False, f"invalid JSON: {e}")
            continue
        miss = [k for k in keys if k not in J[f]]
        check(f"schema:{f}", not miss, f"missing keys {miss}" if miss else "ok")
        check(f"schema_version:{f}", J[f].get("schema_version") == SCHEMA_VERSION,
              f"{J[f].get('schema_version')} (expected {SCHEMA_VERSION})")
    if errors:
        return dict(ok=False, bundle=(p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else str(p)), errors=errors, warnings=warnings, checks=checks)
    man, enc, dec, tp, saf = (J["manifest.json"], J["encoder.json"], J["decoder.json"], J["training_protocol.json"],
                              J["safety.json"])
    check("biology_of_source", man["biology_of_source"] in ("simulated", "living"), man["biology_of_source"])
    check("non_transfer_warning", any("not transferable" in w for w in man["warnings"]),
          "manifest must state that simulated weights are not transferable")
    tid = man["target"]["id"]
    tgt = TARGETS.get(tid)
    check("target_known", tgt is not None, tid)
    nj = len(man["action_space"]["joints"])

    # ---- electrode map ----
    with open(p / "electrode_map.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    cols = rows[0].keys() if rows else []
    check("map_columns", all(c in cols for c in MAP_COLS), f"columns {list(cols)}")
    ids = [r["electrode_id"] for r in rows]
    check("map_unique_ids", len(ids) == len(set(ids)), f"{len(ids)} rows")
    check("map_roles_valid", all(r["role"] in ROLES for r in rows), "")
    names = [r["role_name"] for r in rows if r["role_name"]]
    check("map_unique_role_names", len(names) == len(set(names)), "")
    need = {f"err{s}{a}" for a in "xyz" for s in "+-"} | {f"ctx:{k}" for k in range(nj)} | \
           {f"motor{s}:{k}" for k in range(nj) for s in "+-"}
    check("map_roles_complete", need <= set(names), f"missing {sorted(need - set(names))}")
    if tgt:
        bad = [r["electrode_id"] for r in rows
               if int(r["sim_channel"]) in tgt.unavailable and r["role"] not in ("unavailable", "unused")]
        check("roles_off_unavailable_electrodes", not bad, f"roles on unavailable electrodes {bad}")
    usable = {int(r["electrode_id"]) for r in rows if r["role"] != "unavailable"}
    role_e = {r["role_name"]: int(r["electrode_id"]) for r in rows if r["role_name"]}
    n_stim = sum(1 for r in rows if r["stimulated"] == "True")

    # ---- encoder ----
    wf, pt = enc["waveform"], enc["pulse_timing"]
    st = saf["stimulation"]
    ok = all(role_e.get(c["role_name"]) == int(c["electrode"]) for c in enc["channels"])
    check("encoder_electrodes_match_map", ok, "")
    max_p = max(c["max_pulses"] for c in enc["channels"])
    check("encoder_fits_stim_window", (max_p - 1) * pt["spacing_ms"] < pt["stim_window_ms"],
          f"{max_p} pulses x {pt['spacing_ms']} ms in {pt['stim_window_ms']} ms")
    check("stim_burst_rate", 1000 / pt["spacing_ms"] <= st["max_burst_hz"],
          f"{1000 / pt['spacing_ms']:.0f} Hz <= {st['max_burst_hz']} Hz")
    for k in ("amplitude", "teach_amplitude"):
        check(f"stim_{k}_limit", 0 < wf[k] <= st["max_amplitude"], f"{wf[k]} <= {st['max_amplitude']} {wf['unit']}")
    check("stim_unit_matches_limits", wf["unit"] == st["amplitude_unit"], f"{wf['unit']} vs {st['amplitude_unit']}")
    check("stim_biphasic", "biphasic" in wf.get("shape", ""), wf.get("shape", ""))
    if wf["mode"] == "current":
        q = max(wf["amplitude"], wf["teach_amplitude"]) * wf["phase_width_us"] / 1000.0
        if st.get("max_charge_per_phase_nC"):
            check("stim_charge_per_phase", q <= st["max_charge_per_phase_nC"],
                  f"{q:.3f} nC <= {st['max_charge_per_phase_nC']} nC")
        if st.get("max_total_charge_nC"):
            check("stim_total_charge", 2 * q <= st["max_total_charge_nC"],
                  f"{2 * q:.3f} nC <= {st['max_total_charge_nC']} nC")
    if tgt and tgt.limits.get("phase_width_step_us"):
        s = tgt.limits["phase_width_step_us"]
        check("stim_phase_width_granularity", wf["phase_width_us"] % s == 0, f"{wf['phase_width_us']} us % {s}")
    check("stim_params_flagged", "must be validated" in wf.get("status", ""), "waveform must carry the "
                                                                               "'must be validated' flag")
    T = man["timing"]
    t_teach = tp["teach"]
    worst = max_p + (t_teach["volleys"] if t_teach["enabled"] else 0)
    period = T["trial_ms"] + (T["teach_ms"] if t_teach["enabled"] else 0)
    rate = worst * 1000.0 / period
    check("stim_mean_rate", rate <= st["max_mean_rate_hz_per_electrode"],
          f"worst case {rate:.1f} Hz <= {st['max_mean_rate_hz_per_electrode']} Hz per electrode")
    if t_teach["enabled"]:
        check("teach_pairing_rate", 1000 / t_teach["volley_period_ms"] <= st["max_burst_hz"], "")
        last = t_teach["volley_start_ms"] + (t_teach["volleys"] - 1) * t_teach["volley_period_ms"] + t_teach["pre_post_ms"]
        check("teach_fits_block", last < T["teach_ms"], f"last pulse {last} ms < {T['teach_ms']} ms")
        check("teach_pre_after_zero", t_teach["volley_start_ms"] - t_teach["pre_post_ms"] >= 0, "")
    if st.get("max_stim_electrodes"):
        check("stim_units", n_stim <= st["max_stim_electrodes"],
              f"{n_stim} stimulated electrodes <= {st['max_stim_electrodes']} stimulation units")

    # ---- decoder ----
    W = dec["weights"]
    inp = [int(e) for e in dec["input"]["electrodes"]]
    check("decoder_shape", len(W) == nj and all(len(r) == len(inp) for r in W) and len(dec["bias"]) == nj
          and len(dec["input_offset"]) == len(inp), f"{len(W)} x {len(inp)}")
    check("decoder_inputs_usable", set(inp) <= usable, "")
    check("decoder_window", 0 <= dec["input"]["window_ms"][0] < dec["input"]["window_ms"][1] <= T["trial_ms"], "")
    check("decoder_gain_positive", dec["gain"] > 0, "")

    # ---- schedule ----
    with open(p / "stim_schedule.csv", newline="", encoding="utf-8") as fh:
        sched = list(csv.DictReader(fh))
    span = T["trial_ms"] + T["teach_ms"]
    ok = all(int(s["electrode_id"]) in usable and 0 <= int(s["t_ms"]) < span for s in sched)
    check("stim_schedule_valid", ok and len(sched) > 0, f"{len(sched)} rows within 0..{span} ms")

    # ---- robot safety ----
    r = saf["robot"]
    check("watchdog", T["trial_ms"] < r["watchdog_timeout_ms"] <= 1000, f"{r['watchdog_timeout_ms']} ms")
    check("missing_data_holds", r["on_missing_data"] == "hold_position", r["on_missing_data"])
    joints = man["action_space"]["joints"]
    lo, hi = r["joint_soft_limits"]["lo"], r["joint_soft_limits"]["hi"]
    ok = all(j["range"][0] <= lo[i] < hi[i] <= j["range"][1] for i, j in enumerate(joints))
    check("joint_soft_limits", ok and len(lo) == nj, "inside joint ranges")
    v = r["joint_velocity_limits"]
    check("joint_velocity_limits", len(v) == nj and all(x > 0 for x in v), "")
    check("action_step_bounded", all(0 < x <= 1 for x in r["max_step_normalised"]), "")
    ws = r.get("workspace")
    check("workspace", ws is not None and all(a < b for a, b in zip(ws["min"], ws["max"])), "")

    # ---- reference ----
    check("sim_reference", (p / "sim_reference.npz").exists() or (p / "sim_reference.nwb").exists(),
          "sim_reference.npz / .nwb", warn=True)
    if tgt and tgt.status.startswith("stub"):
        check("target_backend", False, f"{tid}: stub-needs-vendor-sdk (layout/format only, no runtime backend)",
              warn=True)
    return dict(ok=not errors, bundle=(p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else str(p)), target=tid, errors=errors, warnings=warnings, checks=checks)
