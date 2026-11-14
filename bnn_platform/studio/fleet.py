"""`studio train --platform fleet` and `studio fleet-status`: one Freestyle VM per seed.

Same mechanics as sweep/run_jobs.sh (payload tarball -> `vm scp`, a per-job `job.sh`, launched with
nohup via `vm exec`, JOB_DONE marker, collect by tarring results on the VM and `vm scp` back), but
self-contained in Python with its own state file (user_runs/fleet_jobs.json), so it never touches
sweep/jobs_state.txt or the VMs listed there.

The VM runs the exact same local trainer:
  python -m bnn_platform studio train --robot R ... --seed S --run-id RID --no-frames > events.jsonl
and its run directory user_runs/RID/ is pulled back by fleet-status once the job is done.
"""
from __future__ import annotations

import datetime as _dt
import json
import re
import os
import pathlib
import shlex
import subprocess
import tarfile
import threading

from .common import ROOT, USER_ROBOTS, USER_RUNS, StudioError, emit, log, rel
from .robots import BUILTINS, read_manifest
from .train import check_args, new_run_id

STATE = USER_RUNS / "fleet_jobs.json"
PIP = "python3 -m venv .v && . .v/bin/activate && pip install -q mujoco==4.0.1 numpy"
REMOTE_HOME = "/home/ubuntu"
FREE_TIER_MAX = 10


def fs_cmd() -> list:
    return ["npx.cmd" if os.name == "nt" else "npx", "-y", "freestyle@latest"]


def _fs(args: list, timeout: int = 300) -> str:
    p = subprocess.run(fs_cmd() + args, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError(f"freestyle {' '.join(args[:2])} failed: {(p.stderr or p.stdout).strip()[-400:]}")
    return p.stdout


# ---------------------------------------------------------------- state
def load_state() -> list:
    if STATE.exists():
        return json.loads(STATE.read_text(encoding="utf-8")).get("jobs", [])
    return []


_lock = threading.Lock()


def save_state(jobs: list) -> None:
    with _lock:
        cur = {j["slug"]: j for j in load_state()}
        for j in jobs:
            cur[j["slug"]] = j
        USER_RUNS.mkdir(exist_ok=True)
        STATE.write_text(json.dumps({"jobs": list(cur.values())}, indent=1), encoding="utf-8")


# ---------------------------------------------------------------- payload
def _payload_files(robot: str) -> list:
    files = []

    def add_tree(d: pathlib.Path, pattern="**/*"):
        for p in d.glob(pattern):
            if p.is_file() and "__pycache__" not in p.parts and p.suffix not in (".pyc", ".pkl", ".tgz"):
                files.append(p)

    add_tree(ROOT / "bnn_platform")
    files.append(ROOT / "neurorover" / "__init__.py")
    add_tree(ROOT / "neurorover" / "culture", "*.py")
    if robot == "surgical_reach":
        add_tree(ROOT / "surgical", "*.py")
        add_tree(ROOT / "surgical" / "model")
    else:
        add_tree(USER_ROBOTS / robot)
    return files


def build_payload(robot: str, out: pathlib.Path) -> dict:
    files = _payload_files(robot)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out, "w:gz") as tar:
        for p in files:
            tar.add(p, arcname=p.relative_to(ROOT).as_posix())
    return {"path": rel(out), "files": len(files), "bytes": out.stat().st_size}


H = f"H={REMOTE_HOME}"


def job_script(train_cmd: str, run_id: str) -> str:
    return "\n".join([
        H,
        "mkdir -p $H/nr && cd $H/nr && tar -xzf $H/payload.tar.gz",
        f"if ( {PIP} && {train_cmd} > $H/nr/events.jsonl 2> $H/nr/train.log ); then ok=1; else ok=0; fi",
        f"cd $H/nr && tar -czf $H/out.tgz user_runs/{run_id} events.jsonl train.log 2>/dev/null; true",
        "if [ $ok = 1 ]; then touch $H/nr/JOB_DONE; else touch $H/nr/JOB_FAILED; fi", ""])


START_SH = f"{H}\nnohup bash $H/job.sh > $H/job.log 2>&1 < /dev/null &\necho started\n"
STATUS_SH = "\n".join([
    H,
    "if [ -f $H/nr/JOB_DONE ]; then echo STATE=done; elif [ -f $H/nr/JOB_FAILED ]; then echo STATE=failed;"
    " else echo STATE=running; fi",
    "grep '\"type\":\"episode\"' $H/nr/events.jsonl 2>/dev/null | tail -n 400",  # history; last line = latest
    "grep '\"type\":\"error\"' $H/nr/events.jsonl 2>/dev/null | tail -n 1",
    "echo LOG=$(tail -n 1 $H/nr/train.log 2>/dev/null)", ""])


# ---------------------------------------------------------------- train --platform fleet
def train_fleet(a) -> int:
    check_args(a)
    if a.robot in BUILTINS and not BUILTINS[a.robot]["trainable"]:
        raise StudioError(f"{a.robot!r} is not trainable in Studio")
    if a.robot not in BUILTINS:
        read_manifest(a.robot)  # validates
    if not 1 <= a.vms <= FREE_TIER_MAX:
        raise StudioError(f"--vms must be 1..{FREE_TIER_MAX} (Freestyle free tier: {FREE_TIER_MAX} running VMs)")
    parent = new_run_id(a.robot, f"{a.rule}_fleet", a.seed)
    run_dir = USER_RUNS / parent
    slug_robot = a.robot.replace("_", "-")
    jobs = []
    for s in range(a.seed, a.seed + a.vms):
        rid = f"{parent}_s{s}"
        cmd = (f"python -m bnn_platform studio train --robot {a.robot} --task {a.task} --platform local "
               f"--episodes {a.episodes} --lr {a.lr:g} --neurons {a.neurons} --electrodes {a.electrodes} "
               f"--rule {a.rule} --seed {s} --run-id {rid} --parent-run {parent} --no-frames"
               + (f" --steps {a.steps}" if a.steps else ""))
        jobs.append(dict(slug=f"bnn-{slug_robot}-s{s}", vm=None, state="planned" if a.dry_run else "starting",
                         seed=s, run_id=rid, parent_run=parent, robot=a.robot, command=cmd, last=None,
                         collected=False, created=_dt.datetime.now().isoformat(timespec="seconds")))
    config = dict(run_id=parent, robot=a.robot, task=a.task, platform="fleet", episodes=a.episodes, lr=a.lr,
                  neurons=a.neurons, electrodes=a.electrodes, rule=a.rule, seed=a.seed, vms=a.vms,
                  steps_per_episode=a.steps, jobs=[j["slug"] for j in jobs], biology="simulated",
                  created=_dt.datetime.now().isoformat(timespec="seconds"))

    def public(j):
        return {k: j[k] for k in ("slug", "vm", "state", "seed", "run_id")}

    if a.dry_run:
        payload = {"path": rel(run_dir / "payload.tar.gz"), "files": len(_payload_files(a.robot)),
                   "built": False}
        plan = [dict(public(j), command=j["command"],
                     freestyle=[shlex.join(fs_cmd()[1:] + ["vm", "create", "--slug", j["slug"], "--replace-slug",
                                                           "--no-ssh", "--idle-timeout-seconds", "10800",
                                                           "--metadata", "project=bnn-studio", "--output", "json"]),
                                f"freestyle vm scp {payload['path']} <vm>:{REMOTE_HOME}/payload.tar.gz",
                                f"freestyle vm exec <vm> -- bash {REMOTE_HOME}/start.sh"],
                     job_sh=job_script(j["command"], j["run_id"])) for j in jobs]
        emit({"type": "fleet", "dry_run": True, "run_id": parent, "payload": payload, "jobs": plan,
              "biology": "simulated"})
        emit({"type": "done", "run_dir": None, "dry_run": True,
              "summary": {"platform": "fleet", "jobs": len(jobs), "launched": 0}})
        return 0

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config, indent=1), encoding="utf-8")
    ev = open(run_dir / "events.jsonl", "w", encoding="utf-8")
    try:
        emit({"type": "start", "run_id": parent, "run_dir": rel(run_dir), "config": config,
              "biology": "simulated"}, ev)
        info = build_payload(a.robot, run_dir / "payload.tar.gz")
        emit({"type": "log", "message": f"payload {info['bytes'] / 1e6:.1f} MB, {info['files']} files"}, ev)
        emit({"type": "fleet", "run_id": parent, "jobs": [public(j) for j in jobs]}, ev)
        save_state(jobs)
        jsdir = run_dir / "jobscripts"
        jsdir.mkdir(exist_ok=True)
        (jsdir / "start.sh").write_bytes(START_SH.encode())
        (jsdir / "status.sh").write_bytes(STATUS_SH.encode())

        def launch(j):
            try:
                out = _fs(["vm", "create", "--slug", j["slug"], "--replace-slug", "--no-ssh",
                           "--idle-timeout-seconds", "10800", "--metadata", "project=bnn-studio",
                           "--output", "json"], timeout=300)
                d = json.loads(out[out.find("{"):])
                j["vm"] = d.get("id") or d.get("vmId")
                save_state([j])
                pay = rel(run_dir / "payload.tar.gz")
                for t in range(3):
                    try:
                        _fs(["vm", "scp", pay, f"{j['vm']}:{REMOTE_HOME}/payload.tar.gz"], timeout=600)
                        break
                    except RuntimeError:
                        if t == 2:
                            raise
                js = jsdir / f"job_{j['slug']}.sh"
                js.write_bytes(job_script(j["command"], j["run_id"]).encode())
                _fs(["vm", "scp", rel(js), f"{j['vm']}:{REMOTE_HOME}/job.sh"])
                for nm in ("start.sh", "status.sh"):
                    _fs(["vm", "scp", rel(jsdir / nm), f"{j['vm']}:{REMOTE_HOME}/{nm}"])
                _fs(["vm", "exec", j["vm"], "--timeout-ms", "60000", "--", "bash", f"{REMOTE_HOME}/start.sh"],
                    timeout=180)
                j["state"] = "running"
            except Exception as e:  # noqa: BLE001
                j["state"] = "failed"
                j["error"] = str(e)[-400:]
                log(f"{j['slug']}: {e}")
            save_state([j])

        th = [threading.Thread(target=launch, args=(j,)) for j in jobs]
        for t in th:
            t.start()
        for t in th:
            t.join()
        emit({"type": "fleet", "run_id": parent, "jobs": [dict(public(j), **({"error": j["error"]} if "error" in j
                                                                             else {})) for j in jobs]}, ev)
        ok = sum(j["state"] == "running" for j in jobs)
        summary = {"platform": "fleet", "jobs": len(jobs), "launched": ok,
                   "note": "poll `studio fleet-status`; results are collected into user_runs/<run_id>_s<seed>/"}
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
        emit({"type": "done", "run_dir": rel(run_dir), "summary": summary}, ev)
        return 0 if ok else 1
    finally:
        ev.close()


# ---------------------------------------------------------------- fleet-status
def _safe_extract(tgz: pathlib.Path, run_id: str) -> None:
    with tarfile.open(tgz) as tar:
        for mbr in tar.getmembers():
            p = pathlib.PurePosixPath(mbr.name)
            if p.is_absolute() or ".." in p.parts or not (mbr.isfile() or mbr.isdir()):
                continue
            if p.parts[0] == "user_runs" and len(p.parts) > 1 and p.parts[1] == run_id:
                tar.extract(mbr, USER_RUNS.parent)
            elif mbr.isfile() and p.name in ("train.log",):
                tar.extract(mbr, USER_RUNS / run_id / "_vm")


def _poll(j: dict) -> None:
    try:
        out = _fs(["vm", "exec", j["vm"], "--timeout-ms", "60000", "--", "bash", f"{REMOTE_HOME}/status.sh"],
                  timeout=120)
    except Exception as e:  # noqa: BLE001
        j["poll_error"] = str(e)[-300:]
        return
    j.pop("poll_error", None)
    hist = []
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("STATE="):
            j["state"] = line[6:]
        elif line.startswith("{"):
            try:
                o = json.loads(line)
            except ValueError:
                continue
            if o.get("type") == "episode":
                j["last"] = o
                hist.append({"episode": o.get("episode"), "final_err_mm": o.get("final_err_mm")})
            elif o.get("type") == "error":
                j["error"] = o.get("message")
        elif line.startswith("LOG=") and len(line) > 4:
            j["log"] = line[4:][-300:]
    if hist:
        j["history"] = hist
    if j["state"] in ("done", "failed") and not j.get("collected"):
        dst = USER_RUNS / j["parent_run"] / f"{j['slug']}.tgz"
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            _fs(["vm", "scp", f"{j['vm']}:{REMOTE_HOME}/out.tgz", rel(dst)], timeout=600)
            _safe_extract(dst, j["run_id"])
            j["collected"] = True
            sp = USER_RUNS / j["run_id"] / "summary.json"
            if sp.exists():
                j["summary"] = json.loads(sp.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            j["collect_error"] = str(e)[-300:]


def fleet_status(a) -> dict:
    jobs = load_state()
    todo = [j for j in jobs if j.get("vm") and (j["state"] in ("starting", "running")
                                               or (j["state"] in ("done", "failed") and not j.get("collected")))]
    if not a.no_remote:
        th = [threading.Thread(target=_poll, args=(j,)) for j in todo]
        for t in th:
            t.start()
        for t in th:
            t.join()
        if todo:
            save_state(todo)
    keys = ("slug", "vm", "state", "last", "run_id", "parent_run", "robot", "seed", "collected", "summary",
            "error", "history", "created")
    out = []
    for j in jobs:
        o = {k: j.get(k) for k in keys}
        m = re.search(r"--lr\s+(\S+)", j.get("command") or "")
        o["lr"] = float(m.group(1)) if m else None
        m = re.search(r"--episodes\s+(\d+)", j.get("command") or "")
        o["episodes"] = int(m.group(1)) if m else None
        if not o["history"]:
            o["history"] = _local_history(j.get("run_id"))
        out.append(o)
    return {"jobs": out}


def _local_history(run_id) -> list:
    """Per-episode error from a collected run's events.jsonl (for jobs polled before history existed)."""
    if not run_id or not re.fullmatch(r"[A-Za-z0-9_.-]{1,160}", str(run_id)):
        return []
    p = USER_RUNS / run_id / "events.jsonl"
    hist = []
    try:
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            if '"episode"' not in line:
                continue
            try:
                o = json.loads(line)
            except ValueError:
                continue
            if o.get("type") == "episode":
                hist.append({"episode": o.get("episode"), "final_err_mm": o.get("final_err_mm")})
    except OSError:
        return []
    return hist[-400:]
