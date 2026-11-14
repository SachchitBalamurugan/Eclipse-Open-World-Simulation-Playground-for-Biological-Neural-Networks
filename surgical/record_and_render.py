"""Replay the trained culture on held-out targets: trajectory.json, spikes/stims CSV, MP4.

usage: python surgical/record_and_render.py [teach|frozen] [n_targets]
"""
import csv
import json
import pathlib
import pickle
import shutil
import subprocess
import sys

import mujoco
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from surgical import controllers as C  # noqa: E402
from surgical.arm import CONTROLLED, make_targets  # noqa: E402
from surgical.culture_ctrl import MOTOR, NEG_POOL, POS_POOL  # noqa: E402
from surgical.reach_env import SurgicalReachEnv  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent / "results"
mode = sys.argv[1] if len(sys.argv) > 1 else "teach"
NT = int(sys.argv[2]) if len(sys.argv) > 2 else 3

with open(OUT / f"culture_{mode}.pkl", "rb") as fh:
    cc = pickle.load(fh)
cc.learning = False
env = SurgicalReachEnv(seed=0)
held = make_targets(env.m, 50, seed=999)
culture = cc.culture

# ---- per-spike logging (like neurorover/record_run.py SignalWriter) ----
log = dict(active=False, step=0, t0=0, spikes=[], stims=[])
orig_step, orig_stim = culture.step, culture.stim


def logged_step():
    f = orig_step()
    if log["active"] and f.size:
        t = culture.t_ms - log["t0"]
        for ch in culture.ch[f]:
            log["spikes"].append((log["ep"], log["step"], "control", int(t), int(ch)))
    return f


def logged_stim(ch, cur=1.0):
    if log["active"]:
        t = culture.t_ms - log["t0"]
        kind = "sensory_error" if ch < 6 else "sensory_context"
        log["stims"].append((log["ep"], log["step"], "control", int(t), int(ch), kind))
    return orig_stim(ch, cur)


culture.step, culture.stim = logged_step, logged_stim

renderer = mujoco.Renderer(env.m, 480, 640)
cam = mujoco.MjvCamera()
cam.lookat[:] = [1.1, 0.2, 0.85]
cam.distance, cam.azimuth, cam.elevation = 1.6, 205, -22
frames_dir = OUT / "frames"
shutil.rmtree(frames_dir, ignore_errors=True)
frames_dir.mkdir()
fi = 0
episodes = []
pick = [0, 3, 7, 11, 19][:NT]
for ep, ti in enumerate(pick):
    steps = []
    log["ep"] = ep

    def rec(env_, a, info):
        global fi
        steps.append(dict(t=env_.t * 0.05, q=env_.q().tolist(), qref=env_.qref.tolist(),
                          tip=env_.ix.tip(env_.d).tolist(), target=env_.target.tolist(),
                          dist_mm=1000 * info["dist"], action=np.asarray(a).tolist(),
                          levels=cc.last["levels"].tolist(), spike_counts=cc.last["counts"].tolist()))
        renderer.update_scene(env_.d, cam)
        from PIL import Image
        Image.fromarray(renderer.render()).save(frames_dir / f"f{fi:05d}.png")
        fi += 1
        log["step"] = env_.t
        log["t0"] = culture.t_ms

    log["active"] = ep == 0
    log["t0"] = culture.t_ms
    log["step"] = 0
    r = C.run_episode(env, cc, held[ti], seed=ti, record=rec)
    episodes.append(dict(target_index=int(ti), result=r, steps=steps))
    print(ep, ti, r)
log["active"] = False

traj = dict(
    description="Simulated-culture (rule=%s) control of surgical cart arm 1, held-out targets" % mode,
    joints=CONTROLLED, control_dt_s=0.05, frame="world, Z-up, meters (USD stage frame)",
    electrodes=dict(sensory_error=list(range(6)), sensory_context=list(range(6, 14)),
                    motor_pos_pool=POS_POOL, motor_neg_pool=NEG_POOL,
                    spike_window="0-50 ms of each 60 ms control trial"),
    episodes=episodes)
(OUT / "trajectory.json").write_text(json.dumps(traj))
with open(OUT / "culture_episode0_spikes.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["episode", "step", "phase", "t_ms", "channel"])
    w.writerows(log["spikes"])
with open(OUT / "culture_episode0_stims.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["episode", "step", "phase", "t_ms", "channel", "kind"])
    w.writerows(log["stims"])
mp4 = OUT / f"culture_{mode}_reach.mp4"
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "20", "-i", str(frames_dir / "f%05d.png"),
                "-pix_fmt", "yuv420p", "-vcodec", "libx264", str(mp4)], check=True)
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "20", "-i", str(frames_dir / "f%05d.png"),
                "-vf", "fps=10,scale=480:-1", str(OUT / f"culture_{mode}_reach.gif")], check=True)
print("frames", fi, "->", mp4)
