"""`studio runs` (all runs incl. legacy), `studio replay`, `studio pack`."""
from __future__ import annotations

import datetime as _dt
import json
import pathlib
import time
import zipfile

from .common import ROOT, USER_RUNS, StudioError, emit, rel

REPLAY_DT = 0.05  # ~20 fps


def _j(p: pathlib.Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def _mtime(p: pathlib.Path) -> str:
    return _dt.datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds")


def all_runs() -> list:
    runs = []
    if USER_RUNS.exists():
        for cfg in USER_RUNS.glob("*/config.json"):
            c = _j(cfg) or {}
            d = cfg.parent
            runs.append(dict(run_id=d.name, robot=c.get("robot"), kind="studio", rule=c.get("rule"),
                             platform=c.get("platform"), created=c.get("created") or _mtime(cfg),
                             summary=_j(d / "summary.json"), replayable=(d / "events.jsonl").exists(),
                             path=rel(d), parent_run=c.get("parent_run"), episodes=c.get("episodes"),
                             lr=c.get("lr"), seed=c.get("seed")))
    for d in sorted((ROOT / "runs").glob("*/")) if (ROOT / "runs").exists() else []:
        js = sorted(d.glob("*.json"))
        summ = _j(d / "summary.json") or (_j(js[0]) if js else None)
        if isinstance(summ, list):
            summ = {"rows": len(summ)}
        runs.append(dict(run_id=f"legacy_runs_{d.name}", robot="rover_culture", kind="legacy_culture",
                         rule=d.name.split("_")[1] if "_" in d.name else None, platform="local",
                         created=_mtime(d), summary=summ, replayable=bool(js), path=rel(d)))
    for d in sorted((ROOT / "sim" / "runs").glob("*/")) if (ROOT / "sim" / "runs").exists() else []:
        summ = _j(d / "season_summary.json") or _j(d / "eval.json")
        runs.append(dict(run_id=f"legacy_sim_{d.name}", robot="rover", kind="legacy_rover", rule=None,
                         platform="local", created=_mtime(d), summary=summ,
                         replayable=bool(list(d.glob("trajectory*.json"))), path=rel(d)))
    sr = ROOT / "surgical" / "results"
    for cv in sorted(sr.glob("curve_*.json")) if sr.exists() else []:
        c = _j(cv) or {}
        mode = cv.stem.replace("curve_", "")
        summ = {k: c.get(k) for k in ("mode", "lr", "episodes", "teaching_steps")}
        fin = c.get("final") or []
        if fin:
            summ["success_rate"] = sum(r["success"] for r in fin) / len(fin)
            errs = sorted(r["final_err_mm"] for r in fin)
            summ["median_final_err_mm"] = errs[len(errs) // 2]
        runs.append(dict(run_id=f"legacy_surgical_{mode}", robot="surgical_reach", kind="legacy_surgical",
                         rule=mode, platform="local", created=_mtime(cv), summary=summ, replayable=True,
                         path=rel(cv)))
    runs.sort(key=lambda r: r["created"] or "", reverse=True)
    return runs


def _find(run_id: str) -> dict:
    for r in all_runs():
        if r["run_id"] == run_id:
            return r
    raise StudioError(f"unknown run {run_id!r} (see `studio runs`)")


def replay(run_id: str) -> int:
    r = _find(run_id)
    p = ROOT / r["path"]
    last = [0.0]

    def frame(o):
        dt = REPLAY_DT - (time.time() - last[0])
        if dt > 0:
            time.sleep(dt)
        last[0] = time.time()
        emit(o)

    if r["kind"] == "studio":
        ev = p / "events.jsonl"
        if not ev.exists():
            raise StudioError(f"{run_id} has no events.jsonl")
        for line in ev.read_text(encoding="utf-8").splitlines():
            try:
                o = json.loads(line)
            except ValueError:
                continue
            o["replay"] = True
            frame(o) if o.get("type") == "frame" else emit(o)
        return 0
    emit({"type": "start", "run_id": run_id, "run_dir": r["path"], "config": {"robot": r["robot"], "kind": r["kind"]},
          "biology": "simulated", "replay": True})
    traj = None
    if r["kind"] == "legacy_surgical":
        traj = _j(p.parent / "trajectory.json") if r["rule"] == "teach" else None
        curve = (_j(p) or {}).get("curve", [])
    elif r["kind"] == "legacy_rover":
        ts = sorted(p.glob("trajectory*.json"))
        traj = _j(ts[0]) if ts else None
        curve = []
    else:
        js = sorted(p.glob("*.json"))
        c = _j(js[0]) if js else None
        curve = c if isinstance(c, list) else (c or {}).get("curve", []) if isinstance(c, dict) else []
    if isinstance(traj, dict):
        eps = traj.get("episodes") or [traj]
        for ei, ep in enumerate(eps if isinstance(eps, list) else [eps]):
            steps = ep.get("frames") or ep.get("steps") or ep.get("rows") or [] if isinstance(ep, dict) else []
            for si, st in enumerate(steps if isinstance(steps, list) else []):
                o = {"type": "frame", "episode": ei + 1, "step": si + 1, "replay": True}
                if isinstance(st, dict):
                    o.update({k: v for k, v in st.items() if k != "type"})
                frame(o)
    for i, row in enumerate(curve if isinstance(curve, list) else []):
        if isinstance(row, dict):
            emit({"type": "episode", "episode": row.get("episode", i + 1), "episodes": len(curve),
                  "success_rate": row.get("success"), "median_err_mm": row.get("median_err_mm"), "replay": True})
    emit({"type": "done", "run_dir": r["path"], "summary": r["summary"], "replay": True})
    return 0


def pack(run_id: str, out: str | None) -> dict:
    r = _find(run_id)
    p = ROOT / r["path"]
    dst = pathlib.Path(out) if out else (USER_RUNS / "_packs" / f"{run_id}.zip")
    dst.parent.mkdir(parents=True, exist_ok=True)
    files = [p] if p.is_file() else [f for f in p.rglob("*") if f.is_file() and f.suffix not in (".tgz",)
                                     and f.name != "payload.tar.gz"]
    if r["kind"] == "legacy_surgical":
        files += [f for f in (p.parent / f"culture_{r['rule']}.pkl", p.parent / "trajectory.json") if f.exists()]
    base = p.parent if p.is_file() else p.parent
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, f.relative_to(base).as_posix())
    return {"ok": True, "zip": rel(dst), "files": len(files), "bytes": dst.stat().st_size}
