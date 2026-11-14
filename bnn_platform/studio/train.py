"""`studio train` (local): BNN reach training on any Studio robot, streamed as JSON Lines."""
from __future__ import annotations

import datetime as _dt
import json
import pickle
import time

import numpy as np

from .bnn import MAX_J, BNNReachController, KinematicReach
from .common import USER_RUNS, StudioError, emit, log, r6, rel
from .robots import load_robot

FRAME_DT = 0.1          # <= 10 frames / s
DEFAULT_STEPS = 60      # 3 s of 50 ms control steps per episode
N_EVAL = 10             # held-out targets evaluated (plasticity off) before and after training


def new_run_id(robot: str, tag: str, seed: int) -> str:
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M")
    base = f"{robot}_{tag}_s{seed}_{stamp}"
    rid, i = base, 2
    while (USER_RUNS / rid).exists():
        rid, i = f"{base}_{i}", i + 1
    return rid


def check_args(a):
    if a.task != "reach":
        raise StudioError(f"unknown task {a.task!r} (only 'reach')")
    if a.rule not in ("teach", "frozen"):
        raise StudioError("--rule must be teach or frozen")
    if a.electrodes != 64:
        raise StudioError("only the 64-electrode MEA layout is supported (--electrodes 64)")
    if not 64 <= a.neurons <= 4096:
        raise StudioError("--neurons must be in 64..4096 (rounded to a multiple of 64)")
    if a.episodes < 1:
        raise StudioError("--episodes must be >= 1")


def train_local(a) -> int:
    check_args(a)
    rm = load_robot(a.robot)
    if len(rm.jid) > MAX_J:
        raise StudioError(f"robot has {len(rm.jid)} controlled joints (max {MAX_J})")
    run_id = a.run_id or new_run_id(a.robot, a.rule, a.seed)
    run_dir = USER_RUNS / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    npe = max(1, a.neurons // 64)
    steps = a.steps or DEFAULT_STEPS
    config = dict(run_id=run_id, robot=a.robot, task=a.task, platform="local", episodes=a.episodes, lr=a.lr,
                  neurons=64 * npe, electrodes=64, rule=a.rule, seed=a.seed, steps_per_episode=steps,
                  control_dt_s=0.05, n_eval=N_EVAL, controlled_joints=rm.joint_names, ee=rm.ee_label,
                  success_radius_mm=round(1000 * rm.success_r, 1), encode_thresh_mm=r6(rm.thresh_mm, 1),
                  step_scale=r6(rm.scale), sim="kinematic (ideal joint-position servo)",
                  created=_dt.datetime.now().isoformat(timespec="seconds"), biology="simulated",
                  culture="neurorover.culture.core.Culture (Izhikevich, simulated)")
    if a.parent_run:
        config["parent_run"] = a.parent_run
    (run_dir / "config.json").write_text(json.dumps(config, indent=1), encoding="utf-8")
    ev = open(run_dir / "events.jsonl", "w", encoding="utf-8")

    def E(o):
        emit(o, ev)

    try:
        E({"type": "start", "run_id": run_id, "run_dir": rel(run_dir), "config": config, "biology": "simulated"})
        t0 = time.time()
        ctrl = BNNReachController(len(rm.jid), seed=a.seed, rule=a.rule, lr=a.lr, npe=npe, thresh_mm=rm.thresh_mm)
        task = KinematicReach(rm, steps=steps)
        train_targets = rm.sample_targets(a.episodes, np.random.default_rng(a.seed + 1))
        eval_targets = rm.sample_targets(N_EVAL, np.random.default_rng(999))
        last_frame = [0.0]

        def episode(target, learn, ep=None):
            ctrl.learning = learn and ctrl.teach
            task.reset(target)
            errs = []
            while True:
                dq = task.dq_star() if ctrl.learning else None
                act, counts = ctrl.act(task.err_base(), task.qn01(), dq)
                done = task.step(act)
                errs.append(task.dist)
                now = time.time()
                if ep is not None and not a.no_frames and now - last_frame[0] >= FRAME_DT:
                    last_frame[0] = now
                    E({"type": "frame", "episode": ep, "step": task.t, "bodies": task.body_poses(),
                       "target": r6(task.target), "tip": r6(task.tip()), "err_mm": round(1000 * task.dist, 1),
                       "spikes": counts.tolist()})
                if done:
                    break
            return dict(success=bool(task.success), final_err_mm=round(1000 * task.dist, 2),
                        median_err_mm=round(1000 * float(np.median(errs)), 2), steps=task.t)

        def evaluate(tag):
            R = [episode(t, learn=False) for t in eval_targets]
            s = dict(success_rate=float(np.mean([r["success"] for r in R])),
                     median_final_err_mm=round(float(np.median([r["final_err_mm"] for r in R])), 2), episodes=R)
            E({"type": "log", "message": f"eval {tag} ({N_EVAL} held-out targets, plasticity off): "
                                         f"{int(round(s['success_rate'] * N_EVAL))}/{N_EVAL} reached, "
                                         f"median final error {s['median_final_err_mm']:.1f} mm"})
            return s

        E({"type": "log", "message": f"{rm.name}: {len(rm.jid)} joints {rm.joint_names}, EE {rm.ee_label}, "
                                     f"{64 * npe} simulated neurons, rule {a.rule}, lr {a.lr}"})
        eval_before = evaluate("before training")
        curve, finals, succ = [], [], []
        for i, tgt in enumerate(train_targets):
            r = episode(tgt, learn=True, ep=i + 1)
            finals.append(r["final_err_mm"])
            succ.append(r["success"])
            row = dict(episode=i + 1, success=r["success"], final_err_mm=r["final_err_mm"],
                       episode_median_err_mm=r["median_err_mm"], steps=r["steps"],
                       teaching_steps=ctrl.n_teach, wall_s=round(time.time() - t0, 2))
            curve.append(row)
            E({"type": "episode", "episode": i + 1, "episodes": a.episodes, "success": r["success"],
               "final_err_mm": r["final_err_mm"], "median_err_mm": round(float(np.median(finals)), 2),
               "success_rate": round(float(np.mean(succ)), 4), "steps": r["steps"],
               "teaching_steps": ctrl.n_teach})
        eval_after = evaluate("after training")
        c = ctrl.culture
        summary = dict(success_rate=round(float(np.mean(succ)), 4),
                       median_final_err_mm=round(float(np.median(finals)), 2), episodes=a.episodes,
                       eval_before={k: v for k, v in eval_before.items() if k != "episodes"},
                       eval_after={k: v for k, v in eval_after.items() if k != "episodes"},
                       teaching_steps=ctrl.n_teach,
                       plastic_w_mean=round(float(c.W[c.plastic].mean()), 4) if c.plastic.any() else None,
                       success_radius_mm=round(1000 * rm.success_r, 1), wall_s=round(time.time() - t0, 1),
                       rule=a.rule, lr=a.lr, robot=a.robot, biology="simulated")
        (run_dir / "curve.json").write_text(json.dumps(dict(curve=curve, eval_before=eval_before,
                                                            eval_after=eval_after), indent=1), encoding="utf-8")
        ctrl.learning = False
        with open(run_dir / "culture.pkl", "wb") as fh:
            pickle.dump(ctrl, fh)
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
        E({"type": "done", "run_dir": rel(run_dir), "summary": summary})
        log(f"done in {time.time() - t0:.1f}s -> {rel(run_dir)}")
        return 0
    except KeyboardInterrupt:
        E({"type": "error", "message": "interrupted"})
        return 130
    except StudioError as e:
        E({"type": "error", "message": str(e)})
        return 1
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        E({"type": "error", "message": f"{type(e).__name__}: {e}"})
        return 1
    finally:
        ev.close()
