"""Load a trained source run (Studio run dir, or the legacy surgical results) into one description.

Supported sources
  * Studio run:   user_runs/<id>/ with config.json + culture.pkl (bnn_platform.studio.bnn.BNNReachController)
                  [+ curve.json, summary.json]
  * Legacy surgical: surgical/results/culture_teach.pkl (surgical.culture_ctrl.CultureController),
                  or the surgical/results directory (uses culture_teach.pkl + curve_teach.json)
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import pickle
import sys
from dataclasses import dataclass, field
from typing import Any

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class TransferError(Exception):
    pass


@dataclass
class Layout:
    """The electrode protocol the culture was trained with (sim channel numbers, 0..63)."""
    err_ch: list            # 6 channels: +x,-x,+y,-y,+z,-z of the EE->target error
    ctx_ch: list            # one per controlled joint: normalised joint position
    pos_pool: list          # motor pool driving joint k in + direction
    neg_pool: list          # motor pool driving joint k in - direction
    thresh_mm: list         # error magnitude thresholds -> 0..4 pulses
    ctx_max_pulses: int = 2
    pulse_spacing_ms: int = 10
    stim_ms: int = 40       # encoder pulses fall in [0, stim_ms)
    resp_ms: int = 50       # spikes counted in [0, resp_ms)
    trial_ms: int = 60      # one control step of culture time
    teach_ms: int = 85      # paired-stimulation block after each control step
    teach_volleys: int = 4
    teach_volley_start_ms: int = 10
    teach_volley_period_ms: int = 20
    teach_pre_post_ms: int = 3
    teach_select_frac: float = 0.15
    deadzone: float = 2.0
    gain: float = 8.0
    reward_ch: list = field(default_factory=list)
    feedback_ch: list = field(default_factory=list)


@dataclass
class Source:
    kind: str                       # "studio" | "surgical_legacy"
    run_id: str
    run_dir: pathlib.Path
    pkl_path: pathlib.Path
    robot: str
    task: str
    rule: str
    seed: int
    lr: float
    episodes: int | None
    joints: list                    # [{name,type,range,scale,unit}]
    layout: Layout
    controller: Any                 # unpickled controller (has .culture)
    culture: Any                    # neurorover.culture.core.Culture
    config: dict
    curve: dict | None
    summary: dict | None
    robot_sim: dict                 # how to rebuild the robot in simulation
    control_dt_s: float = 0.05
    steps_per_episode: int = 80
    success_radius_mm: float = 10.0
    ee: str = "tip"
    action_lowpass: float = 0.5
    extra: dict = field(default_factory=dict)

    @property
    def n_joints(self) -> int:
        return len(self.joints)


def sha256(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_pickle(p: pathlib.Path):
    try:
        with open(p, "rb") as fh:
            return pickle.load(fh)
    except Exception as e:  # noqa: BLE001
        raise TransferError(f"could not unpickle {p}: {type(e).__name__}: {e}")


def _culture_of(obj):
    if hasattr(obj, "culture"):
        return obj.culture
    if isinstance(obj, dict) and "culture" in obj:
        return obj["culture"]
    if hasattr(obj, "W") and hasattr(obj, "step"):
        return obj
    raise TransferError("culture.pkl does not contain a culture (expected .culture or a Culture)")


def _read_json(p: pathlib.Path):
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return None
    return None


def _joint_info(names, lo, hi, is_slide, scale):
    out = []
    for i, n in enumerate(names):
        slide = bool(is_slide[i])
        out.append(dict(index=i, name=n, type="slide" if slide else "hinge", unit="m" if slide else "rad",
                        range=[float(lo[i]), float(hi[i])], step_scale=float(scale[i])))
    return out


def load_source(run: str | pathlib.Path) -> Source:
    p = pathlib.Path(run)
    if not p.is_absolute():
        p = (ROOT / p) if not p.exists() else p.resolve()
    if not p.exists():
        raise TransferError(f"source not found: {run}")
    if p.is_file() and p.suffix == ".pkl":
        if (p.parent / "config.json").exists():
            return _load_studio(p.parent)
        return _load_surgical(p)
    if p.is_dir():
        if (p / "config.json").exists() and (p / "culture.pkl").exists():
            return _load_studio(p)
        if (p / "culture_teach.pkl").exists():
            return _load_surgical(p / "culture_teach.pkl")
        if (p / "config.json").exists():
            raise TransferError(f"{p} has config.json but no culture.pkl (training not finished?)")
    raise TransferError(f"{run}: expected a Studio run dir (config.json + culture.pkl) "
                        f"or surgical/results/culture_teach.pkl")


def _load_surgical(pkl: pathlib.Path) -> Source:
    import surgical.culture_ctrl as CC
    from surgical.arm import CONTROLLED, load_model, ArmIndex
    ctrl = _load_pickle(pkl)
    cult = _culture_of(ctrl)
    mode = pkl.stem.replace("culture_", "") or "teach"
    curve = _read_json(pkl.parent / f"curve_{mode}.json")
    m = load_model()
    ix = ArmIndex(m)
    scale = np.where(ix.is_slide, 0.006, 0.03)
    joints = _joint_info(CONTROLLED, ix.lo, ix.hi, ix.is_slide, scale)
    layout = Layout(err_ch=list(CC.ERR_CH), ctx_ch=list(CC.CTX_CH), pos_pool=list(CC.POS_POOL),
                    neg_pool=list(CC.NEG_POOL), thresh_mm=[float(x) for x in CC.THRESH_MM],
                    stim_ms=CC.STIM_MS, resp_ms=CC.RESP_MS, trial_ms=CC.TRIAL_MS,
                    deadzone=float(CC.DEADZONE), gain=float(CC.GAIN))
    summary = None
    if curve and curve.get("final"):
        f = curve["final"]
        summary = dict(success_rate=float(np.mean([r["success"] for r in f])),
                       median_final_err_mm=float(np.median([r["final_err_mm"] for r in f])),
                       episodes=curve.get("episodes"), eval_targets=len(f))
    rel = pkl.relative_to(ROOT).as_posix() if pkl.is_relative_to(ROOT) else str(pkl)
    return Source(kind="surgical_legacy", run_id=f"surgical_reach_{mode}_legacy", run_dir=pkl.parent,
                  pkl_path=pkl, robot="surgical_reach", task="reach", rule=mode,
                  seed=0, lr=float(curve.get("lr", 0.05)) if curve else 0.05,
                  episodes=curve.get("episodes") if curve else None, joints=joints, layout=layout,
                  controller=ctrl, culture=cult, config=dict(source=rel, trainer="surgical/train_culture.py"),
                  curve=curve, summary=summary,
                  robot_sim=dict(kind="surgical_env", env="surgical.reach_env:SurgicalReachEnv",
                                 model="surgical/model/cart.xml", base_body="handBase_01", ee="tip",
                                 dynamics="MuJoCo position servos, 25 x 2 ms physics substeps per action"),
                  control_dt_s=0.05, steps_per_episode=80, success_radius_mm=10.0, ee="tip")


def _load_studio(run_dir: pathlib.Path) -> Source:
    cfg = _read_json(run_dir / "config.json") or {}
    ctrl = _load_pickle(run_dir / "culture.pkl")
    cult = _culture_of(ctrl)
    nj = int(getattr(ctrl, "nj", len(cfg.get("controlled_joints", [])) or 0))
    if nj <= 0:
        raise TransferError("cannot determine number of controlled joints")
    ctx = list(getattr(ctrl, "ctx_ch", range(6, 6 + nj)))
    pos = list(getattr(ctrl, "pos_pool", [32 + 2 * k for k in range(nj)]))
    neg = list(getattr(ctrl, "neg_pool", [33 + 2 * k for k in range(nj)]))
    thr = getattr(ctrl, "thresh", None)
    thr = [float(x) for x in (thr if thr is not None else cfg.get("encode_thresh_mm", [2, 8, 25, 60]))]
    try:
        from bnn_platform.studio import bnn as B
        timing = dict(stim_ms=B.STIM_MS, resp_ms=B.RESP_MS, trial_ms=B.TRIAL_MS,
                      deadzone=float(B.DEADZONE), gain=float(B.GAIN))
        err = list(B.ERR_CH)
    except Exception:  # noqa: BLE001
        timing, err = {}, list(range(6))
    layout = Layout(err_ch=err, ctx_ch=ctx, pos_pool=pos, neg_pool=neg, thresh_mm=thr, **timing)
    robot = cfg.get("robot", "unknown")
    names = cfg.get("controlled_joints") or [f"j{k}" for k in range(nj)]
    scale = cfg.get("step_scale") or [0.04] * nj
    lo = hi = slide = None
    try:  # joint ranges from the robot model (studio loader)
        from bnn_platform.studio.robots import load_robot
        rm = load_robot(robot)
        lo, hi, slide, scale = rm.lo, rm.hi, rm.is_slide, rm.scale
        names = rm.joint_names
    except Exception:  # noqa: BLE001
        lo, hi = [-np.pi] * nj, [np.pi] * nj
        slide = [False] * nj
    joints = _joint_info(names, lo, hi, slide, scale)
    curve = _read_json(run_dir / "curve.json")
    summary = _read_json(run_dir / "summary.json")
    rel = run_dir.relative_to(ROOT).as_posix() if run_dir.is_relative_to(ROOT) else str(run_dir)
    return Source(kind="studio", run_id=cfg.get("run_id", run_dir.name), run_dir=run_dir,
                  pkl_path=run_dir / "culture.pkl", robot=robot, task=cfg.get("task", "reach"),
                  rule=cfg.get("rule", "teach"), seed=int(cfg.get("seed", 0)), lr=float(cfg.get("lr", 0.05)),
                  episodes=cfg.get("episodes"), joints=joints, layout=layout, controller=ctrl, culture=cult,
                  config=dict(cfg, source=rel), curve=curve, summary=summary,
                  robot_sim=dict(kind="studio_kinematic", env="bnn_platform.studio.bnn:KinematicReach",
                                 robot=robot, ee=cfg.get("ee"),
                                 dynamics=cfg.get("sim", "kinematic (ideal joint-position servo)")),
                  control_dt_s=float(cfg.get("control_dt_s", 0.05)),
                  steps_per_episode=int(cfg.get("steps_per_episode", 60)),
                  success_radius_mm=float(cfg.get("success_radius_mm", 10.0)), ee=cfg.get("ee") or "ee")
