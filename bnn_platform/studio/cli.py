"""python -m bnn_platform studio <command>   (contract: docs/STUDIO_CONTRACT.md)

  list                                   builtin + imported robots
  import --file PATH [--name N] [--ee X] import an MJCF / URDF / zip robot into user_robots/<name>/
  scene --robot N                        viewport geometry (JSON)
  train --robot N [--platform local|fleet] [--episodes 20] [--lr 0.02] [--neurons 1024]
        [--electrodes 64] [--rule teach|frozen] [--seed 0] [--vms 4] [--dry-run]
                                         JSON Lines event stream
  fleet-status                           poll / collect Freestyle jobs
  runs                                   list user_runs/

stdout carries only JSON; logs go to stderr. Biology is simulated (neurorover.culture).
"""
from __future__ import annotations

import argparse
import json
import sys

from . import common
from .common import USER_RUNS, StudioError, emit


def cmd_list(a):
    from .robots import list_robots
    emit({"robots": list_robots()})
    return 0


def cmd_import(a):
    from .robots import import_robot
    try:
        man = import_robot(a.file, a.name, a.ee)
    except StudioError as e:
        emit({"ok": False, "error": str(e)})
        return 1
    except Exception as e:  # noqa: BLE001
        emit({"ok": False, "error": f"{type(e).__name__}: {e}"})
        return 1
    emit({"ok": True, "robot": man})
    return 0


def cmd_scene(a):
    from .scene import scene
    emit(scene(a.robot))
    return 0


def cmd_train(a):
    try:
        if a.platform == "fleet":
            from .fleet import train_fleet
            return train_fleet(a)
        from .train import train_local
        return train_local(a)
    except StudioError as e:
        emit({"type": "error", "message": str(e)})
        return 1


def cmd_fleet_status(a):
    from .fleet import fleet_status
    emit(fleet_status(a))
    return 0


def cmd_runs(a):
    from .history import all_runs
    emit({"runs": all_runs()})
    return 0


def cmd_replay(a):
    from .history import replay
    return replay(a.run)


def cmd_pack(a):
    from .history import pack
    emit(pack(a.run, a.out))
    return 0


def build_parser():
    ap = argparse.ArgumentParser(prog="python -m bnn_platform studio", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list").set_defaults(fn=cmd_list)
    p = sub.add_parser("import")
    p.add_argument("--file", required=True)
    p.add_argument("--name")
    p.add_argument("--ee", help="EE site or body name (default: deepest body of the longest chain)")
    p.set_defaults(fn=cmd_import)
    p = sub.add_parser("scene")
    p.add_argument("--robot", required=True)
    p.set_defaults(fn=cmd_scene)
    p = sub.add_parser("train")
    p.add_argument("--robot", required=True)
    p.add_argument("--task", default="reach")
    p.add_argument("--platform", choices=["local", "fleet"], default="local")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--lr", type=float, default=0.02,
                   help="culture plasticity rate (0.02 tuned for the generic reach task; 0.05 overshoots)")
    p.add_argument("--neurons", type=int, default=1024)
    p.add_argument("--electrodes", type=int, default=64)
    p.add_argument("--rule", choices=["teach", "frozen"], default="teach")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--vms", type=int, default=4)
    p.add_argument("--dry-run", action="store_true")
    # not in the UI contract (used by fleet jobs / power users)
    p.add_argument("--steps", type=int, default=None, help="control steps per episode (default 60 = 3 s)")
    p.add_argument("--run-id", default=None, help=argparse.SUPPRESS)
    p.add_argument("--parent-run", default=None, help=argparse.SUPPRESS)
    p.add_argument("--no-frames", action="store_true", help="don't emit frame events")
    p.set_defaults(fn=cmd_train)
    p = sub.add_parser("fleet-status")
    p.add_argument("--no-remote", action="store_true", help="report the cached state, don't contact VMs")
    p.set_defaults(fn=cmd_fleet_status)
    sub.add_parser("runs").set_defaults(fn=cmd_runs)
    p = sub.add_parser("replay")
    p.add_argument("--run", required=True)
    p.set_defaults(fn=cmd_replay)
    p = sub.add_parser("pack")
    p.add_argument("--run", required=True)
    p.add_argument("--out", default=None)
    p.set_defaults(fn=cmd_pack)
    return ap


def main(argv=None) -> int:
    real_out = sys.stdout
    common.set_stdout(real_out)
    a = build_parser().parse_args(argv)
    sys.stdout = sys.stderr  # stray prints must not corrupt the JSON stream
    try:
        return a.fn(a) or 0
    except StudioError as e:
        emit({"ok": False, "error": str(e)} if a.cmd not in ("train", "replay") else {"type": "error", "message": str(e)})
        return 1
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        msg = f"{type(e).__name__}: {e}"
        emit({"ok": False, "error": msg} if a.cmd not in ("train", "replay") else {"type": "error", "message": msg})
        return 1
    finally:
        sys.stdout = real_out
