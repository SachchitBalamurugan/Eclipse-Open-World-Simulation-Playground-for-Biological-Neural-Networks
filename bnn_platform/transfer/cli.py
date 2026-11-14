"""python -m bnn_platform.transfer targets|export|validate|run|to-nwb  (JSON on stdout, logs on stderr)."""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import pathlib
import sys
import time

import numpy as np

from . import BIOLOGY_OF_SOURCE
from .source import ROOT, TransferError


def _default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, pathlib.Path):
        return o.as_posix()
    return str(o)


def out(obj, fh=None):
    s = json.dumps(obj, default=_default)
    sys.stdout.write(s + "\n")
    sys.stdout.flush()
    if fh is not None:
        fh.write(s + "\n")
        fh.flush()


def log(msg):
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


def cmd_targets(a):
    from .targets import TARGETS
    out({"targets": [dict(id=t.id, name=t.name, status=t.status, vendor=t.vendor, n_electrodes=t.n_electrodes,
                          sdk=t.sdk) for t in TARGETS.values()]})
    return 0


def cmd_export(a):
    from .export import export
    from .targets import TARGETS
    tids = list(TARGETS) if a.target == "all" else a.target.split(",")
    res = []
    for t in tids:
        log(f"[transfer] exporting {a.run} -> {t}")
        res.append(export(a.run, t, out=a.out if len(tids) == 1 else None, reference=not a.no_reference))
    out(res[0] if len(res) == 1 else {"ok": all(r["ok"] for r in res), "exports": res})
    return 0 if all(r["ok"] for r in res) else 1


def cmd_validate(a):
    from .validate import validate
    v = validate(a.bundle)
    if not a.verbose:
        v = {k: v[k] for k in ("ok", "bundle", "target", "errors", "warnings") if k in v} | \
            {"n_checks": len(v["checks"]), "n_passed": sum(c["ok"] for c in v["checks"])}
    out(v)
    return 0 if v["ok"] else 1


def cmd_to_nwb(a):
    from .nwb import to_nwb
    try:
        p = to_nwb(pathlib.Path(a.bundle))
    except ImportError:
        out({"ok": False, "error": "pynwb is not installed (pip install pynwb)"})
        return 1
    out({"ok": True, "nwb": str(p)})
    return 0


def cmd_run(a):
    from .runtime import (Bundle, CL1Backend, ClosedLoop, MaxWellBackend, MCSBackend, MockMEABackend, calibrate,
                          load_source_culture, make_robot)
    b = Bundle(a.bundle)
    run_id = f"{b.dir.name}_{a.backend}_{_dt.datetime.now():%Y%m%d-%H%M%S}"
    sdir = b.dir / "sessions" / run_id
    sdir.mkdir(parents=True, exist_ok=True)
    ev = open(sdir / "events.jsonl", "w", encoding="utf-8")

    def E(o):
        out(o, ev)

    backend = robot = None
    try:
        if b.target in ("maxwell", "mcs") and a.backend == "mock":
            log(f"[transfer] note: {b.target} bundle replayed on the mock (logical electrodes -> sim grid)")
        amp = float(b.encoder["waveform"]["amplitude"])
        lr = float(b.manifest["provenance"].get("lr") or 0.05)
        desc = ""
        if a.backend == "mock":
            cult, desc = load_source_culture(b, a.mock_culture, a.seed)
            cult.lr = lr if a.mock_culture == "naive" else cult.lr
            backend = MockMEABackend(cult, b, amp)
        elif a.backend in ("cl1-sim", "cl1"):
            if b.target not in ("cl1", "generic"):
                raise TransferError(f"cl1 backend needs a cl1 (or generic) bundle, got {b.target}")
            pkl = None
            if a.backend == "cl1-sim" and a.mock_culture == "trained":
                pkl = str(ROOT / b.manifest["provenance"]["culture_pickle"]["path"])
            backend = CL1Backend(b, simulate=(a.backend == "cl1-sim"), culture_pkl=pkl,
                                 rule="teach" if not a.no_teach else "frozen", seed=a.seed).open()
            desc = "cl API, " + ("cl-sdk simulator with " + (pkl or "naive culture") if a.backend == "cl1-sim"
                                 else "LIVING culture on CL1 hardware")
        elif a.backend == "maxwell":
            backend = MaxWellBackend()
        elif a.backend == "mcs":
            backend = MCSBackend()
        robot = make_robot(b, a.robot)
        loop = ClosedLoop(b, backend, robot, emit=E, teach=not a.no_teach, frames_per_s=a.fps)
        cfg = dict(bundle=b.dir.relative_to(ROOT).as_posix() if b.dir.is_relative_to(ROOT) else str(b.dir),
                   backend=a.backend, culture=desc, robot=a.robot, episodes=a.episodes, teach=not a.no_teach,
                   calibrate=not a.no_calibrate, target=b.target, session_dir=sdir.relative_to(ROOT).as_posix()
                   if sdir.is_relative_to(ROOT) else str(sdir))
        E({"type": "start", "run_id": run_id, "run_dir": cfg["session_dir"], "config": cfg,
           "biology": backend.biology, "biology_of_source": BIOLOGY_OF_SOURCE,
           "electrode_ids": b.usable})
        t0 = time.time()
        if not a.no_calibrate:
            rep = calibrate(backend, b, loop.roles, robot, loop.encoder, loop.decoder, loop.safety, E,
                            quick=(backend.biology == "simulated" and not a.full_calibration),
                            rng=np.random.default_rng(a.seed))
            (sdir / "calibration.json").write_text(json.dumps(rep, indent=1, default=_default), encoding="utf-8")
            (sdir / "decoder_calibrated.json").write_text(json.dumps(loop.decoder.to_spec(), indent=1,
                                                                     default=_default), encoding="utf-8")
            (sdir / "roles_calibrated.json").write_text(json.dumps(loop.roles.roles, indent=1), encoding="utf-8")
        results = []
        for i in range(a.episodes):
            r = loop.episode(i, label=i + 1)
            results.append(r)
            E({"type": "episode", "episode": i + 1, "episodes": a.episodes, "success": r["success"],
               "final_err_mm": r["final_err_mm"], "median_err_mm": r["median_err_mm"], "steps": r["steps"],
               "success_rate": round(float(np.mean([x["success"] for x in results])), 4),
               "held_steps": r["held_steps"]})
        s = loop.safety
        summary = dict(success_rate=round(float(np.mean([r["success"] for r in results])), 4) if results else None,
                       median_final_err_mm=round(float(np.median([r["final_err_mm"] for r in results])), 2)
                       if results else None, episodes=a.episodes, stims_dropped_by_safety=s.n_dropped,
                       wall_s=round(time.time() - t0, 1), biology=backend.biology)
        (sdir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
        E({"type": "done", "run_dir": cfg["session_dir"], "summary": summary})
        return 0
    except (TransferError, NotImplementedError) as e:
        E({"type": "error", "message": str(e)})
        return 1
    except KeyboardInterrupt:
        E({"type": "error", "message": "interrupted"})
        return 130
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        E({"type": "error", "message": f"{type(e).__name__}: {e}"})
        return 1
    finally:
        try:
            if robot is not None:
                robot.close()
            if backend is not None:
                backend.close()
        finally:
            ev.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m bnn_platform.transfer", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("targets", help="list export targets")
    p = sub.add_parser("export", help="build a deployment bundle from a trained run")
    p.add_argument("--run", required=True, help="Studio run dir (user_runs/<id>) or surgical/results[/culture_teach.pkl]")
    p.add_argument("--target", default="generic", help="generic|cl1|mcs|maxwell, comma list, or all")
    p.add_argument("--out", default=None, help="output dir (single target only)")
    p.add_argument("--no-reference", action="store_true", help="skip the simulated reference episode")
    p = sub.add_parser("validate", help="check a bundle against schema and safety bounds")
    p.add_argument("--bundle", required=True)
    p.add_argument("--verbose", action="store_true", help="include every check")
    p = sub.add_parser("run", help="calibrate + closed loop (JSON Lines)")
    p.add_argument("--bundle", required=True)
    p.add_argument("--backend", default="mock", choices=["mock", "cl1-sim", "cl1", "maxwell", "mcs"])
    p.add_argument("--mock-culture", default="trained", choices=["trained", "naive"],
                   help="trained = exported sim culture (round-trip test); naive = fresh culture (rehearses a dish)")
    p.add_argument("--robot", default="sim", choices=["sim", "udp", "ros2"])
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--no-teach", action="store_true")
    p.add_argument("--no-calibrate", action="store_true")
    p.add_argument("--full-calibration", action="store_true", help="use the full (living-culture) durations")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--fps", type=float, default=10.0, help="max frame events per second (0 = none)")
    p = sub.add_parser("to-nwb", help="convert sim_reference.npz to NWB (needs pynwb)")
    p.add_argument("--bundle", required=True)
    a = ap.parse_args(argv)
    try:
        return {"targets": cmd_targets, "export": cmd_export, "validate": cmd_validate, "run": cmd_run,
                "to-nwb": cmd_to_nwb}[a.cmd](a)
    except TransferError as e:
        out({"ok": False, "error": str(e)})
        return 1
