"""python -m bnn_platform <command>

  robots                                   list registered robots/tasks
  train --robot X [--lr ..] [--episodes ..] run the robot's existing training entry point
  eval  --robot X                          run the robot's existing evaluation
  fleet launch|status|collect|teardown [--jobs FILE] [--dry-run]
                                           wrap sweep/run_jobs.sh (one Freestyle VM per job)
  fleet sweep --robot X --lrs 0.02 0.05 [--episodes N] [--dry-run]
                                           write a jobs file (sweep/jobs_surgical.txt format) and launch
  studio list|import|scene|train|fleet-status|runs
                                           BNN Robot Studio backend, JSON on stdout
                                           (docs/STUDIO_CONTRACT.md; `studio -h` for details)
"""
from __future__ import annotations

import argparse
import os
import pathlib
import shlex
import shutil
import subprocess
import sys

from .registry import get_robot, list_robots

ROOT = pathlib.Path(__file__).resolve().parents[1]
GIT_BASH = r"C:\Program Files\Git\bin\bash.exe"


def _run(cmd: list, dry: bool) -> int:
    print("$", " ".join(shlex.quote(str(c)) for c in cmd), flush=True)
    if dry:
        return 0
    return subprocess.call(cmd, cwd=ROOT)


def _bash() -> str:
    if os.name == "nt" and pathlib.Path(GIT_BASH).exists():
        return GIT_BASH
    return shutil.which("bash") or "bash"


def _rel(path: str) -> str:
    p = pathlib.Path(path).resolve()
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def cmd_robots(a) -> int:
    for r in list_robots():
        print(f"{r.name}\n  {r.description}\n  env:        {r.env_module}\n  controller: {r.controller}")
        print(f"  fleet sweep: {'yes' if r.fleet_job else 'no'}")
        if r.notes:
            print(f"  notes:      {r.notes}")
    return 0


def cmd_train(a) -> int:
    r = get_robot(a.robot)
    return _run(r.train_cmd(a) + a.extra, a.dry_run)


def cmd_eval(a) -> int:
    r = get_robot(a.robot)
    return _run(r.eval_cmd(a) + a.extra, a.dry_run)


def cmd_fleet(a) -> int:
    if a.action == "sweep":
        if not a.robot or not a.lrs:
            raise SystemExit("fleet sweep needs --robot and --lrs")
        r = get_robot(a.robot)
        if r.fleet_job is None:
            raise SystemExit(f"robot {r.name!r} has no fleet_job defined yet")
        episodes = a.episodes if a.episodes is not None else r.defaults.get("episodes", 60)
        prefix = {"surgical_reach": "nr-surg"}.get(r.name, f"nr-{r.name.replace('_', '-')}")
        lines = [f"{prefix}-lr{lr:g}".replace(".", "p") + "|" + r.fleet_job(lr, episodes) for lr in a.lrs]
        if len(lines) > 10:
            print("warning: Freestyle free tier allows at most 10 running VMs", file=sys.stderr)
        out = pathlib.Path(a.jobs or ROOT / "sweep" / f"jobs_{r.name}_sweep.txt")
        out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
        print(f"wrote {len(lines)} jobs -> {_rel(str(out))}")
        a.jobs, action = str(out), "launch"
    else:
        action = a.action
    cmd = [_bash(), "sweep/run_jobs.sh", action]
    if action == "launch":
        if not a.jobs:
            raise SystemExit("fleet launch needs --jobs FILE")
        cmd.append(_rel(a.jobs))
    return _run(cmd, a.dry_run)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m bnn_platform", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    sub.add_parser("robots", help="list registered robots/tasks").set_defaults(fn=cmd_robots)

    for name, fn, hlp in (("train", cmd_train, "train a robot's BNN controller locally"),
                          ("eval", cmd_eval, "evaluate a robot locally")):
        p = sub.add_parser(name, help=hlp)
        p.add_argument("--robot", required=True)
        p.add_argument("--lr", type=float, default=None, help="culture plasticity learning rate")
        p.add_argument("--episodes", type=int, default=None)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--mode", default=None, help="surgical_reach: teach|frozen")
        p.add_argument("--run", default=None, help="rover: sim/runs/<run> (PPO body policy)")
        p.add_argument("--dry-run", action="store_true", help="print the command, don't run it")
        p.add_argument("extra", nargs=argparse.REMAINDER, help="extra args passed through after --")
        p.set_defaults(fn=fn)

    sub.add_parser("studio", help="BNN Robot Studio backend (JSON; see docs/STUDIO_CONTRACT.md)")

    p = sub.add_parser("fleet", help="run jobs in parallel on Freestyle VMs (sweep/run_jobs.sh)")
    p.add_argument("action", choices=["launch", "status", "collect", "teardown", "sweep"])
    p.add_argument("--jobs", default=None, help="jobs file: <slug>|<shell command> per line")
    p.add_argument("--robot", default=None)
    p.add_argument("--lrs", type=float, nargs="+", default=None)
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--dry-run", action="store_true", help="print, don't launch")
    p.set_defaults(fn=cmd_fleet)
    return ap


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] == "studio":
        from .studio.cli import main as studio_main
        return studio_main(argv[1:])
    a = build_parser().parse_args(argv)
    if getattr(a, "extra", None) and a.extra[0] == "--":
        a.extra = a.extra[1:]
    return a.fn(a) or 0


if __name__ == "__main__":
    raise SystemExit(main())
