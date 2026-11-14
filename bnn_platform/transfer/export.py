"""Build a sim-to-real deployment bundle from a trained source run."""
from __future__ import annotations

import copy
import csv
import datetime as _dt
import json
import pathlib
import shutil
import zipfile

import numpy as np

from . import BIOLOGY_OF_SOURCE, SCHEMA_VERSION
from .source import ROOT, Source, TransferError, load_source, sha256
from .targets import GRID, Target, get_target, remap_roles

MUST_VALIDATE = "must be validated against vendor/lab limits"
NOT_TRANSFERABLE = "simulated weights are not transferable; run calibration on the living culture first"


def _rel(p: pathlib.Path) -> str:
    p = pathlib.Path(p).resolve()
    return p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else str(p)


def _dump(path: pathlib.Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=1), encoding="utf-8")


def trained_roles(src: Source) -> dict:
    """trained sim channel -> role record."""
    L = src.layout
    roles = {}
    axes = [("x", +1), ("x", -1), ("y", +1), ("y", -1), ("z", +1), ("z", -1)]
    for i, ch in enumerate(L.err_ch):
        ax, sg = axes[i]
        roles[ch] = dict(role="sensory", role_name=f"err{'+' if sg > 0 else '-'}{ax}", pool="", joint="",
                         joint_name="", sign=sg, obs_source="ee_error_base_m", obs_index="xyz".index(ax))
    for k, ch in enumerate(L.ctx_ch):
        roles[ch] = dict(role="sensory", role_name=f"ctx:{k}", pool="", joint=k, joint_name=src.joints[k]["name"],
                         sign="", obs_source="joint_pos_norm01", obs_index=k)
    for k in range(src.n_joints):
        roles[L.pos_pool[k]] = dict(role="motor", role_name=f"motor+:{k}", pool="pos", joint=k,
                                    joint_name=src.joints[k]["name"], sign=+1, obs_source="", obs_index="")
        roles[L.neg_pool[k]] = dict(role="motor", role_name=f"motor-:{k}", pool="neg", joint=k,
                                    joint_name=src.joints[k]["name"], sign=-1, obs_source="", obs_index="")
    for ch in L.reward_ch:
        roles[ch] = dict(role="reward", role_name=f"reward:{ch}", pool="", joint="", joint_name="", sign="",
                         obs_source="", obs_index="")
    for ch in L.feedback_ch:
        roles[ch] = dict(role="feedback", role_name=f"feedback:{ch}", pool="", joint="", joint_name="", sign="",
                         obs_source="", obs_index="")
    for ch, r in roles.items():
        r["name"] = r["role_name"]
    return roles


def electrode_rows(target: Target, phys_roles: dict) -> list:
    rows = []
    for ch in range(GRID * GRID):
        r = phys_roles.get(ch)
        x, y = target.xy_um(ch)
        row = dict(electrode_id=target.electrode_id(ch), vendor_label=target.label(ch), sim_channel=ch,
                   trained_sim_channel=r["trained_sim_channel"] if r else "",
                   role=r["role"] if r else ("unavailable" if ch in target.unavailable else "unused"),
                   role_name=r["role_name"] if r else "", pool=r["pool"] if r else "",
                   joint=r["joint"] if r else "", joint_name=r["joint_name"] if r else "",
                   sign=r["sign"] if r else "", obs_source=r["obs_source"] if r else "",
                   obs_index=r["obs_index"] if r else "",
                   stimulated=bool(r and r["role"] in ("sensory", "motor", "reward", "feedback")),
                   grid_x=ch % GRID, grid_y=ch // GRID, x_um=round(x, 1), y_um=round(y, 1),
                   note=target.unavailable.get(ch, ""))
        if target.id == "maxwell":
            rec, stim = target.maxwell_block(ch)
            row["hw_record_electrodes"] = ";".join(map(str, rec))
            row["hw_stim_electrode"] = stim if row["stimulated"] else ""
        rows.append(row)
    return rows


def _write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _encoder(src: Source, target: Target, rows: list) -> dict:
    L = src.layout
    wf = target.waveform
    chans = []
    for r in rows:
        if r["role"] != "sensory":
            continue
        if r["obs_source"] == "ee_error_base_m":
            chans.append(dict(electrode=r["electrode_id"], role_name=r["role_name"], source="ee_error_base_m",
                              index=r["obs_index"], sign=r["sign"], transform="threshold_count",
                              thresholds=[round(t / 1000, 6) for t in L.thresh_mm], threshold_unit="m",
                              max_pulses=len(L.thresh_mm)))
        else:
            chans.append(dict(electrode=r["electrode_id"], role_name=r["role_name"], source="joint_pos_norm01",
                              index=r["obs_index"], transform="round_scale", scale=L.ctx_max_pulses,
                              max_pulses=L.ctx_max_pulses, joint_name=r["joint_name"]))
    chans.sort(key=lambda c: c["role_name"])
    n_max = len(L.thresh_mm)
    return {
        "schema_version": SCHEMA_VERSION,
        "description": "Observation -> number of stimulation pulses per sensory electrode, delivered once per "
                       "control step at the start of the trial (rate/count code, as trained in simulation).",
        "transforms": {
            "threshold_count": "electrode is active only when sign(value) matches 'sign' (value 0 counts as +); "
                               "pulses = number of thresholds with |value| >= threshold (0..max_pulses)",
            "round_scale": "pulses = clip(round(scale * value), 0, max_pulses)"},
        "channels": chans,
        "pulse_timing": {"spacing_ms": L.pulse_spacing_ms, "first_pulse_ms": 0, "stim_window_ms": L.stim_ms,
                         "instantaneous_rate_hz": 1000 / L.pulse_spacing_ms,
                         "max_pulses_per_step": n_max,
                         "frequency_equivalent": f"n pulses at {L.pulse_spacing_ms} ms spacing per "
                                                 f"{L.trial_ms} ms step = {1000 / L.trial_ms:.1f}*n Hz mean "
                                                 f"(inference only) -- e.g. 4 pulses = "
                                                 f"{4000 / L.trial_ms:.0f} Hz mean, 100 Hz instantaneous"},
        "waveform": {"shape": "biphasic, charge-balanced", "mode": wf.mode, "unit": wf.unit,
                     "amplitude": wf.amplitude, "teach_amplitude": wf.teach_amplitude, "polarity": wf.polarity,
                     "phase_width_us": wf.phase_width_us, "interphase_us": wf.interphase_us,
                     "charge_per_phase_nC": (round(wf.amplitude * wf.phase_width_us / 1000, 4)
                                             if wf.mode == "current" else None),
                     "status": MUST_VALIDATE, "note": wf.note},
        "simulation_equivalent": "Culture.stim(channel, 1.0): stim_gain*1.0 pA-equivalent current injected into "
                                 "the ~16 model neurons at the electrode for 1 ms. There is no physical unit "
                                 "correspondence; amplitude must be set by the calibration probe sweep.",
    }


def _decoder(src: Source, rows: list, io_map: dict) -> dict:
    L = src.layout
    inputs = [r["electrode_id"] for r in rows if r["role"] != "unavailable"]
    idx = {e: i for i, e in enumerate(inputs)}
    W = np.zeros((src.n_joints, len(inputs)))
    pools = []
    by_name = {r["role_name"]: r["electrode_id"] for r in rows if r["role_name"]}
    for k in range(src.n_joints):
        p, n = by_name[f"motor+:{k}"], by_name[f"motor-:{k}"]
        W[k, idx[p]], W[k, idx[n]] = 1.0, -1.0
        pools.append(dict(joint=k, joint_name=src.joints[k]["name"], pos_electrode=p, neg_electrode=n))
    return {
        "schema_version": SCHEMA_VERSION,
        "type": "linear_deadzone",
        "formula": "z = W @ (counts - input_offset) + bias; a = clip(sign(z) * max(|z| - deadzone, 0) / gain, -1, 1)",
        "input": {"electrodes": inputs, "count": "spikes per electrode", "window_ms": [0, L.resp_ms],
                  "artifact_blank_ms_after_each_pulse": 2,
                  "artifact_note": "the simulation has no stimulus artifacts; on hardware blank (or use the "
                                   "vendor's artifact suppression) for ~2 ms after every pulse on all channels"},
        "output": {"size": src.n_joints, "names": [j["name"] for j in src.joints], "range": [-1, 1],
                   "unit": "normalised joint-position delta (see manifest.action_space)"},
        "weights": W.tolist(), "bias": [0.0] * src.n_joints, "input_offset": [0.0] * len(inputs),
        "deadzone": L.deadzone, "gain": L.gain,
        "pools": pools,
        "provenance": "simulation: fixed pool-difference readout (+1 on the joint's + pool, -1 on its - pool), "
                      "exactly the decoder the simulated culture was trained with",
        "refit": {"when": "during calibration, before any closed-loop use, and at the start of every session",
                  "how": "ridge regression from spike counts (all non-stimulated usable electrodes) to the "
                         "teacher direction on guided trials, shrunk toward these weights; accepted only if it "
                         "beats this decoder on held-out trials (see calibration_protocol.json)",
                  "status": "these weights are an initial guess for a living culture, not a trained model"},
        "sim_reference_io": io_map,
    }


def sim_io_map(src: Source, reps: int = 10) -> dict:
    """Functional map of the trained SIM culture: mean extra spikes on each motor pool in the 50 ms
    after a 4-pulse sensory volley (minus no-stim baseline). This is what training shaped; a living
    culture's map is measured during calibration and can be compared with it, not copied into it."""
    L = src.layout
    c = copy.deepcopy(src.culture)
    c.rule = "frozen"

    def trial(chans):
        tot = np.zeros(64)
        for _ in range(reps):
            for t in range(L.trial_ms):
                if t % L.pulse_spacing_ms == 0 and t < 4 * L.pulse_spacing_ms:
                    for ch in chans:
                        c.stim(ch, 1.0)
                f = c.step()
                if t < L.resp_ms and f.size:
                    tot += np.bincount(c.ch[f], minlength=64)
            c.run(100)
        return tot / reps

    base = trial([])
    names = ["err+x", "err-x", "err+y", "err-y", "err+z", "err-z"]
    motor = [(f"motor+:{k}", L.pos_pool[k]) for k in range(src.n_joints)] + \
            [(f"motor-:{k}", L.neg_pool[k]) for k in range(src.n_joints)]
    M = []
    for ch in L.err_ch:
        r = trial([ch]) - base
        M.append([round(float(r[m]), 2) for _, m in motor])
    return {"description": "trained SIM culture only: extra spikes per motor pool in the response window after "
                           "4 pulses on one error electrode (mean of %d, baseline-subtracted). For comparison "
                           "with the living culture's evoked responses measured in calibration." % reps,
            "rows": names, "cols": [n for n, _ in motor], "matrix": M}


def _training(src: Source, target: Target, rows: list) -> tuple[dict, list]:
    L = src.layout
    by = {r["role_name"]: r["electrode_id"] for r in rows if r["role_name"]}
    proto = {
        "schema_version": SCHEMA_VERSION,
        "rule": "teach",
        "rule_trained_in_sim": src.rule,
        "summary": "After every control step, replay the error-encoding pattern and pair it with stimulation of "
                   "the motor pool that WOULD have moved each joint the right way (pro-pool 3 ms after the "
                   "sensory volley: pre->post, potentiation under STDP) and of the opposite pool 3 ms before it "
                   "(post->pre, depression). Sensory->motor STDP is the only plasticity the simulation models.",
        "biological_basis": "spike-timing-dependent plasticity in dissociated cortical cultures is induced by "
                            "paired/tetanic electrical stimulation (e.g. Jimbo et al. 1999; Bi & Poo 1998 for the "
                            "STDP window). Whether this exact 3 ms pairing potentiates the targeted pathway in a "
                            "given culture is an empirical question -- measure it (see controls).",
        "teacher": {"definition": "dq* = J(q)^T (target - EE) / step_scale, J = EE position Jacobian of the "
                                  "controlled joints (Jacobian-transpose direction, in action units)",
                    "real_robot": "computed from the robot's URDF and measured joint state; no culture involved"},
        "teach": {"enabled": src.rule == "teach", "after": "every control step (when teaching)",
                  "block_ms": L.teach_ms, "volleys": L.teach_volleys, "volley_start_ms": L.teach_volley_start_ms,
                  "volley_period_ms": L.teach_volley_period_ms, "pre_post_ms": L.teach_pre_post_ms,
                  "select_frac": L.teach_select_frac,
                  "sensory": "error electrodes whose step level > volley index",
                  "pro_pool": "joint k: motor+ if dq*_k > 0 else motor-, for |dq*_k| >= select_frac * max|dq*|",
                  "anti_pool": "the other pool of each selected joint",
                  "amplitude": "encoder.waveform.teach_amplitude"},
        "feedback": {"enabled": False,
                     "note": "not used by the simulated 'teach' rule. Optional DishBrain-style alternative if "
                             "paired stimulation does not shape the culture: predictable stimulation after "
                             "success, unpredictable (random electrodes/timing) after failure (Kagan et al. "
                             "2022). Electrodes: any unused, stimulation-responsive electrodes.",
                     "electrodes": [], "predictable": {"rate_hz": 100, "duration_ms": 100},
                     "unpredictable": {"rate_hz": 5, "duration_ms": 4000}},
        "session_plan": {
            "prerequisites": ["culture mature enough for network activity (typically >= 14-21 DIV for rodent "
                              "cortical cultures; per your lab's protocol)", "calibration_protocol.json completed "
                              "in the same session", "robot in a safe workspace (or use the simulated robot)"],
            "blocks": [{"name": "pre-test", "episodes": 10, "teach": False},
                       {"name": "training", "episodes": src.episodes or 30, "teach": True,
                        "interleave": "every 10 training episodes run 5 test episodes with teach off"},
                       {"name": "post-test", "episodes": 10, "teach": False},
                       {"name": "rest", "minutes": 30}],
            "targets": "held-out targets for tests, distinct from training targets"},
        "controls": ["sham: identical session with teach stimulation replaced by the same pulses at shuffled "
                     "times (breaks the pre->post order)",
                     "decoder-only: decoder refit on stimulation patterns alone (no culture) -- the culture must "
                     "beat it", "no-culture / dead-culture well with identical electrodes",
                     "spontaneous drift: pre/post baseline recordings without stimulation"],
        "success_metrics": ["test-episode success rate and final error vs pre-test and sham",
                            "change in evoked sensory->motor responses (compare calibration probe maps "
                            "before vs after; and with decoder.json sim_reference_io)"],
        "sim_training": {"episodes": src.episodes, "lr": src.lr, "rule": src.rule, "summary": src.summary},
    }
    sched = []
    for r in rows:
        if r["role"] != "sensory":
            continue
        n = len(L.thresh_mm) if r["obs_source"] == "ee_error_base_m" else L.ctx_max_pulses
        for k in range(n):
            sched.append(dict(phase="encode", t_ms=k * L.pulse_spacing_ms, electrode_id=r["electrode_id"],
                              role_name=r["role_name"], condition=f"level >= {k + 1}",
                              amplitude="amplitude", note="every control step"))
    for v in range(L.teach_volleys):
        tv = L.trial_ms + L.teach_volley_start_ms + v * L.teach_volley_period_ms
        for k in range(src.n_joints):
            p, n = by[f"motor+:{k}"], by[f"motor-:{k}"]
            sel = f"|dq*_{k}| >= {L.teach_select_frac} max|dq*|"
            sched.append(dict(phase="teach_anti", t_ms=tv - L.teach_pre_post_ms, electrode_id=p,
                              role_name=f"motor+:{k}", condition=f"{sel} and dq*_{k} < 0",
                              amplitude="teach_amplitude", note="LTD pairing (post before pre)"))
            sched.append(dict(phase="teach_anti", t_ms=tv - L.teach_pre_post_ms, electrode_id=n,
                              role_name=f"motor-:{k}", condition=f"{sel} and dq*_{k} > 0",
                              amplitude="teach_amplitude", note="LTD pairing (post before pre)"))
        for name in ["err+x", "err-x", "err+y", "err-y", "err+z", "err-z"]:
            sched.append(dict(phase="teach_sensory", t_ms=tv, electrode_id=by[name], role_name=name,
                              condition=f"level > {v}", amplitude="teach_amplitude", note="sensory replay"))
        for k in range(src.n_joints):
            p, n = by[f"motor+:{k}"], by[f"motor-:{k}"]
            sel = f"|dq*_{k}| >= {L.teach_select_frac} max|dq*|"
            sched.append(dict(phase="teach_pro", t_ms=tv + L.teach_pre_post_ms, electrode_id=p,
                              role_name=f"motor+:{k}", condition=f"{sel} and dq*_{k} > 0",
                              amplitude="teach_amplitude", note="LTP pairing (pre before post)"))
            sched.append(dict(phase="teach_pro", t_ms=tv + L.teach_pre_post_ms, electrode_id=n,
                              role_name=f"motor-:{k}", condition=f"{sel} and dq*_{k} < 0",
                              amplitude="teach_amplitude", note="LTP pairing (pre before post)"))
    sched.sort(key=lambda s: (s["t_ms"], s["phase"], str(s["electrode_id"])))
    return proto, sched


def _calibration(src: Source) -> dict:
    L = src.layout
    refit = dict(guided_episodes=10, steps_per_episode=60, guide_gain=0.5, ridge_lambda=10.0, min_holdout_r2=0.05)
    return {
        "schema_version": SCHEMA_VERSION,
        "purpose": "Find which electrodes of THIS culture can drive activity and which carry evoked activity, "
                   "move electrode roles onto them, and fit the spike->action decoder for this culture. "
                   "Required before any closed-loop use; the simulated weights do not transfer.",
        "steps": ["1 baseline: record spontaneous activity, no stimulation",
                  "2 probe sweep: single biphasic pulses on every usable electrode (random order, interleaved), "
                  "measure evoked network spikes; escalate amplitude up the ladder for non-responsive "
                  "electrodes without exceeding safety.json limits",
                  "3 remap: keep each sensory role on its electrode if stimulation there evokes a network "
                  "response, else move it to the nearest responsive free electrode; keep each motor role if its "
                  "electrode receives evoked spikes from sensory probes (or is spontaneously active), else move "
                  "it to the nearest receiving free electrode",
                  "4 guided trials: the robot is moved by the teacher (no culture control) while the encoder "
                  "stimulates the culture with the real observations; record spike counts",
                  "5 decoder refit: ridge regression counts -> teacher direction, shrunk toward the sim decoder; "
                  "accept only if it beats the sim decoder on held-out guided trials",
                  "6 save calibration report + calibrated decoder with the session recording"],
        "baseline": {"duration_s": 300, "active_rate_hz": 0.1},
        "probe": {"repetitions": 30, "isi_ms": 200, "order": "random permutation of electrodes per repetition",
                  "amplitude": "encoder.waveform.amplitude", "amplitude_ladder": [1.0, 1.5, 2.0],
                  "response_window_ms": [2, L.resp_ms], "responsive_z": 3.0, "min_evoked_spikes": 0.5},
        "remap": {"min_recipient_spikes_per_probe": 0.05,
                  "prefer": "original electrode, then nearest by distance"},
        "decoder_refit": dict(refit, features="all usable electrodes except sensory (stimulated) electrodes",
                              target="teacher direction dq*/max|dq*| per step",
                              prior="decoder.json weights / gain"),
        "mock_compressed": {"note": "shortened durations used by `run --backend mock` so a dry run takes seconds",
                            "baseline": {"duration_s": 5}, "probe": {"repetitions": 8, "isi_ms": 100},
                            "decoder_refit": dict(refit, guided_episodes=1, steps_per_episode=40)},
        "recalibrate": "every session; also when success drops > 20 % from the session's post-calibration level",
    }


def _safety(src: Source, target: Target, workspace: dict, n_stim: int) -> dict:
    L = src.layout
    lo = np.array([j["range"][0] for j in src.joints])
    hi = np.array([j["range"][1] for j in src.joints])
    sc = np.array([j["step_scale"] for j in src.joints])
    mg = 0.02 * (hi - lo)
    lim = target.limits
    return {
        "schema_version": SCHEMA_VERSION,
        "status": MUST_VALIDATE,
        "stimulation": {
            "max_amplitude": lim["max_amplitude"], "amplitude_unit": lim["amplitude_unit"],
            "max_charge_per_phase_nC": lim.get("max_charge_per_phase_nC"),
            "max_total_charge_nC": lim.get("max_total_charge_nC"),
            "charge_balanced_biphasic_only": True,
            "max_burst_hz": lim["max_burst_hz"], "max_mean_rate_hz_per_electrode": 100.0, "rate_window_ms": 1000,
            "max_stim_electrodes": target.max_stim_electrodes, "stim_electrodes_in_bundle": n_stim,
            "session": {"max_session_min": 60, "min_rest_between_sessions_min": 60, "max_sessions_per_day": 2},
            "on_violation": "drop the pulse and log it (never scale up, never retry immediately)",
            "status": MUST_VALIDATE},
        "robot": {
            "max_step_normalised": [1.0] * src.n_joints,
            "joint_velocity_limits": [round(float(s / src.control_dt_s), 4) for s in sc],
            "joint_velocity_unit": [("m/s" if j["type"] == "slide" else "rad/s") for j in src.joints],
            "velocity_note": "the sim moves at most step_scale per control step of %.0f ms; a real robot must "
                             "use min(these, manufacturer limits)" % (1000 * src.control_dt_s),
            "joint_soft_limits": {"lo": np.round(lo + mg, 5).tolist(), "hi": np.round(hi - mg, 5).tolist()},
            "workspace": workspace,
            "watchdog_timeout_ms": 2 * max(L.trial_ms, L.teach_ms) + 50,
            "robot_side_watchdog_ms": 500,
            "max_consecutive_faults": 3,
            "on_missing_data": "hold_position",
            "e_stop": {"triggers": ["no spike data within watchdog_timeout_ms", "MEA backend error",
                                    "non-finite decoded action", "EE outside workspace", "operator e-stop",
                                    "max_consecutive_faults reached -> stop session"],
                       "action": "hold current joint positions (servo to measured q), stop stimulation",
                       "hardware": "a physical, independent e-stop on the robot is mandatory; software holds "
                                   "are not a substitute"}},
    }


def _manifest(src: Source, target: Target, workspace: dict, changes: list, files: list, ref: dict) -> dict:
    L = src.layout
    tm = L.trial_ms + L.teach_ms
    culture = src.culture
    return {
        "schema_version": SCHEMA_VERSION,
        "biology_of_source": BIOLOGY_OF_SOURCE,
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "source_run": {"run_id": src.run_id, "kind": src.kind, "path": _rel(src.run_dir)},
        "robot": {"name": src.robot, "ee": src.ee, "sim": src.robot_sim, "n_joints": src.n_joints},
        "task": {"name": src.task, "success_radius_mm": src.success_radius_mm,
                 "steps_per_episode": src.steps_per_episode,
                 "definition": "move the end effector to a target point; success = within success radius "
                               "for consecutive steps"},
        "observation_space": {
            "ee_error_base_m": {"size": 3, "unit": "m", "order": ["x", "y", "z"],
                                "frame": f"robot base body ({src.robot_sim.get('base_body', 'chain base')})",
                                "definition": "target - EE position, rotated into the base frame",
                                "encoding": "electrodes err+x..err-z, threshold_count (encoder.json)"},
            "joint_pos_norm01": {"size": src.n_joints, "unit": "fraction of joint range",
                                 "definition": "(q - lo) / (hi - lo)", "joints": [j["name"] for j in src.joints],
                                 "encoding": "electrodes ctx:k, round_scale (encoder.json)"}},
        "action_space": {"type": "joint position delta", "range": [-1, 1],
                         "definition": "a_f = 0.5 a_f + 0.5 a; q_ref += a_f * step_scale (per control step); "
                                       "q_ref clipped to range; position servo tracks q_ref",
                         "joints": src.joints},
        "electrodes": {"target": target.id, "count_total": target.n_electrodes,
                       "count_logical": GRID * GRID, "layout": target.layout, "pitch_um": target.pitch_um,
                       "electrode_diameter_um": target.electrode_diameter_um,
                       "roles": {k: sum(1 for c in ref["roles"].values() if c["role"] == k)
                                 for k in ("sensory", "motor", "reward", "feedback")},
                       "remapped_for_target": changes, "map_file": "electrode_map.csv"},
        "timing": {"culture_resolution_ms": 1, "stim_window_ms": L.stim_ms, "pulse_spacing_ms": L.pulse_spacing_ms,
                   "response_window_ms": L.resp_ms, "trial_ms": L.trial_ms, "teach_ms": L.teach_ms,
                   "robot_control_dt_s": src.control_dt_s,
                   "loop_rate_hz": {"inference_only": round(1000 / L.trial_ms, 2),
                                    "with_teaching": round(1000 / tm, 2)},
                   "max_end_to_end_latency_ms": 20,
                   "notes": "One control step = one culture trial. In simulation the robot is paused while the "
                            "culture computes; on hardware use step-and-hold (robot holds q_ref during the trial "
                            "and executes one control_dt step per decoded action). Spike-to-action latency adds "
                            "to the loop period; keep it below max_end_to_end_latency_ms."},
        "target": target.as_dict(),
        "provenance": {
            "culture_model": "neurorover.culture.core.Culture: Izhikevich RS/FS (80/20), %d neurons, %d per "
                             "electrode, 1 ms steps, distance-dependent recurrent synapses + plastic "
                             "sensory->motor projection" % (culture.N, culture.npe),
            "culture_pickle": {"path": _rel(src.pkl_path), "sha256": sha256(src.pkl_path)},
            "rule": src.rule, "lr": src.lr, "seed": src.seed, "episodes": src.episodes,
            "training_summary": src.summary, "config": src.config,
            "exporter": "bnn_platform.transfer (%s)" % SCHEMA_VERSION},
        "sim_reference": ref.get("summary"),
        "transferable": ["electrode roles and layout", "observation->stimulation encoding", "timing",
                         "paired-stimulation training schedule", "safety limits",
                         "decoder STRUCTURE (weights only as an initial guess)",
                         "simulated reference activity for comparison"],
        "not_transferable": ["synaptic weights / connectivity of the simulated culture",
                             "the trained performance itself -- a living culture starts untrained",
                             "stimulation amplitudes in physical units (sim has none; set by calibration)"],
        "warnings": [NOT_TRANSFERABLE, "stimulation parameters are conservative defaults: " + MUST_VALIDATE],
        "files": files,
    }


def _readme(src: Source, target: Target, man: dict) -> str:
    T = man["timing"]
    return f"""# Sim-to-real bundle: {src.robot} / {src.task} -> {target.name}

**Source biology: SIMULATED.** This bundle was exported from a controller trained on a *simulated*
Izhikevich culture (run `{src.run_id}`). The simulated synaptic weights do **not** transfer to living
neurons. What this bundle gives you is the closed-loop *protocol*: which electrodes stimulate and record,
how robot observations become pulses, how spikes become joint commands, how to train (paired
stimulation), how to calibrate, and the safety envelope. A living culture must be calibrated and trained
in the dish.

Target: **{target.name}** -- status `{target.status}`.

## Files
| file | what |
|---|---|
| manifest.json | schema, source run, robot/task, observation & action spaces (units, scaling), timing, provenance |
| electrode_map.csv | every electrode: role (sensory/motor/reward/feedback/unused/unavailable), pool, joint, sign, grid x/y, um |
| encoder.json | observation -> pulse count per sensory electrode; biphasic waveform (**{MUST_VALIDATE}**) |
| decoder.json | spike counts -> joint deltas (initial weights from the sim; **refit during calibration**) |
| training_protocol.json + stim_schedule.csv | paired-stimulation 'teach' rule as a timed schedule; session plan; controls |
| calibration_protocol.json | baseline, probe sweep, role remap, guided trials, decoder refit |
| safety.json | stimulation limits, robot velocity/workspace limits, watchdog, e-stop = hold position |
| sim_reference.npz (+ .nwb if pynwb was available) | simulated spikes/stims/behaviour of a reference episode |

## Loop timing
- one control step = {T['trial_ms']} ms trial: pulses in 0-{T['stim_window_ms']} ms ({T['pulse_spacing_ms']} ms apart),
  spikes counted 0-{T['response_window_ms']} ms; + {T['teach_ms']} ms paired-stimulation block when teaching
- loop rate {T['loop_rate_hz']['inference_only']} Hz (inference) / {T['loop_rate_hz']['with_teaching']} Hz (teaching)
- robot: step-and-hold, one {int(1000 * T['robot_control_dt_s'])} ms joint-position step per decoded action

## Procedure (summary; see docs/SIM_TO_REAL.md in the repo)
1. Check every value in safety.json and encoder.json waveform against your hardware manual, electrode
   type and approved protocol. Then `python -m bnn_platform.transfer validate --bundle <this dir>`.
2. Dry run: `python -m bnn_platform.transfer run --bundle <this dir> --backend mock --episodes 2`.
3. Implement / configure the MEABackend for your rig (`bnn_platform/transfer/runtime.py`).
4. Session: calibration_protocol.json -> pre-test -> training blocks -> post-test, with the sham and
   decoder-only controls in training_protocol.json. Start with the **simulated robot**; move to the real
   robot only with a hardware e-stop and the workspace limits in safety.json.
5. Save the session recording (NWB recommended) and compare with sim_reference (rates, evoked
   responses, sensory->motor maps) -- differences are expected and are the point of the comparison.

## Target notes
""" + "\n".join(f"- {n}" for n in target.format_notes) + "\n"


def write_reference(bundle_dir: pathlib.Path, src: Source, spont_ms: int = 5000) -> dict:
    """Run one held-out episode with the trained sim culture through the bundle's own runtime (mock
    backend, teach off): doubles as an export round-trip test and gives the lab a reference recording."""
    from .runtime import Bundle, ClosedLoop, MockMEABackend, Recorder, make_robot
    from .nwb import write_reference_files
    b = Bundle(bundle_dir)
    cult = copy.deepcopy(src.culture)
    cult.rule = "frozen"
    be = MockMEABackend(cult, b, float(b.encoder["waveform"]["amplitude"]))
    robot = make_robot(b, "sim")
    rec = Recorder()
    sp = be.read_spikes(spont_ms)
    rec.add(sp, [], spont_ms)
    loop = ClosedLoop(b, be, robot, teach=False, frames_per_s=0, record=rec, robot_limits=False)
    res = loop.episode(0)
    files = write_reference_files(bundle_dir, b, rec, res, spont_ms)
    return dict(summary=dict(episode="held-out target 0 (seed 999)", teach=False, spontaneous_ms=spont_ms,
                             robot_safety_limits=False,
                             note="sim-faithful: identical actions to the original sim controller (verified); "
                                  "the runtime's robot soft limits are off for this recording only",
                             n_spikes=len(rec.spikes), n_stims=len(rec.stims), duration_ms=rec.t_ms, **res),
                files=files)


def export(run, target_id: str = "generic", out: str | None = None, reference: bool = True) -> dict:
    src = load_source(run)
    target = get_target(target_id)
    roles = trained_roles(src)
    phys, changes = remap_roles(roles, target)
    rows = electrode_rows(target, phys)
    n_stim = sum(1 for r in rows if r["stimulated"])
    warnings = [NOT_TRANSFERABLE]
    if target.max_stim_electrodes and n_stim > target.max_stim_electrodes:
        warnings.append(f"{n_stim} stimulated electrodes exceed {target.name}'s {target.max_stim_electrodes} "
                        f"stimulation units; reduce joints or time-multiplex stimulation units")
    if changes:
        warnings.append(f"{len(changes)} electrode role(s) moved off unavailable electrodes for {target.id}: "
                        + ", ".join(f"{c['name']} {c['from_electrode']}->{c['to_electrode']}" for c in changes))
    if target.status.startswith("stub"):
        warnings.append(f"target {target.id} is stub-needs-vendor-sdk: layout/format adapted, no backend")
    if out:
        bdir = pathlib.Path(out)
    elif src.kind == "studio":
        bdir = src.run_dir / "export" / target.id
    else:
        bdir = src.run_dir / "export" / target.id
    if bdir.exists():
        shutil.rmtree(bdir)
    bdir.mkdir(parents=True)

    from .runtime import _held_out_targets
    tg = np.asarray(_held_out_targets(src.robot_sim, 50))
    ext = np.ptp(tg, 0).max()
    m = max(0.05, 0.5 * ext)
    wmin, wmax = tg.min(0) - m, tg.max(0) + m
    if src.robot_sim["kind"] == "surgical_env":
        from surgical.arm import TABLE_Z
        wmin[2] = max(wmin[2], TABLE_Z)
        wmax[2] = max(wmax[2], 1.2)
    workspace = {"frame": "world (sim)", "min": np.round(wmin, 4).tolist(), "max": np.round(wmax, 4).tolist(),
                 "note": "box around the held-out target set + margin; on a real robot re-define in the robot "
                         "base frame around the physical task space"}

    _write_csv(bdir / "electrode_map.csv", rows)
    _dump(bdir / "encoder.json", _encoder(src, target, rows))
    _dump(bdir / "decoder.json", _decoder(src, rows, sim_io_map(src)))
    proto, sched = _training(src, target, rows)
    _dump(bdir / "training_protocol.json", proto)
    _write_csv(bdir / "stim_schedule.csv", sched)
    _dump(bdir / "calibration_protocol.json", _calibration(src))
    _dump(bdir / "safety.json", _safety(src, target, workspace, n_stim))
    ref = {"roles": roles}
    files = ["manifest.json", "electrode_map.csv", "encoder.json", "decoder.json", "training_protocol.json",
             "stim_schedule.csv", "calibration_protocol.json", "safety.json", "README.md"]
    man = _manifest(src, target, workspace, changes, files, ref)
    _dump(bdir / "manifest.json", man)
    if reference:
        r = write_reference(bdir, src)
        man["sim_reference"] = r["summary"]
        files += r["files"]
        man["files"] = files
        _dump(bdir / "manifest.json", man)
    (bdir / "README.md").write_text(_readme(src, target, man), encoding="utf-8")

    from .validate import validate
    v = validate(bdir)
    warnings += [w for w in v["warnings"] if w not in warnings]
    zpath = bdir.parent / f"{target.id}.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(x for x in bdir.iterdir() if x.is_file()):
            z.write(f, f"{src.run_id}_{target.id}/{f.name}")
    return {"ok": v["ok"], "bundle": _rel(bdir), "zip": _rel(zpath), "files": files, "target": target.id,
            "target_status": target.status, "source": _rel(src.run_dir), "biology_of_source": BIOLOGY_OF_SOURCE,
            "remapped": changes, "sim_reference": man.get("sim_reference"), "validation": {
                "ok": v["ok"], "errors": v["errors"], "n_checks": len(v["checks"])},
            "warnings": warnings}
