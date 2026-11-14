"""FrostRoverEnv: Gymnasium env for the frost-guard rover in MuJoCo.

Task per episode: drive from a random headland start into the lane beside a randomly chosen crop row, stop at
the station marker aligned with the row, and if tonight's decision is DEPLOY, swing the boom out so the frost
cloth drapes over the row. Decision comes from reset(options={"deploy": bool}) or is random (p=0.5).

Recipe borrowed from the Open Duck Mini / Open Duck Playground training setup (our own code, not a copy):
per-episode dynamics randomisation, observation noise, observation and action latency, a low-pass action
filter, and an action-rate (smoothness) penalty.

Smoke test:  .venv/Scripts/python -m sim.rover_env
"""
from __future__ import annotations

import collections
import math
import pathlib

import gymnasium as gym
import mujoco
import numpy as np

from . import params as P
from .build_model import (HF_NCOL, HF_NROW, HF_POS, HF_SIZE, ROW_HALF_W, ROW_H, ROW_X0, ROW_X1, ROW_YS)

XML = pathlib.Path(__file__).resolve().parent / "assets" / "rover.xml"
LANE_OFFSET = 0.65          # m, rover centre to row centre so the boom tip sits over the row
N_SUB = int(round(P.CTRL_DT / P.SIM_DT))
MAX_STEPS = int(P.EPISODE_S / P.CTRL_DT)
BOOM_MAX = math.pi / 2
BOOM_RATE = P.ACT_SPEED / P.LEVER   # rad/s the boom can swing at actuator no-load speed (~0.5)
KT = P.STALL_TORQUE / P.STALL_CURRENT  # N*m/A at the gearbox output
OBS_DIM = 22


def yaw_of(q):
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class FrostRoverEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": int(1 / P.CTRL_DT)}

    def __init__(self, randomize: bool = True, render_mode: str | None = None):
        self.m = mujoco.MjModel.from_xml_path(str(XML))
        self.d = mujoco.MjData(self.m)
        self.randomize = randomize
        self.render_mode = render_mode
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (3,), np.float32)
        m = self.m
        self.cid = m.body("chassis").id
        self.bat_id = m.body("chassis").id
        self.base = dict(mass=m.body_mass.copy(), inertia=m.body_inertia.copy(), ipos=m.body_ipos.copy(),
                         fric=m.geom_friction.copy(), gain=m.actuator_gainprm.copy(), bias=m.actuator_biasprm.copy())
        self.wheel_dof = [m.joint(f"wheel_{n}").dofadr[0] for n in ("fl", "fr", "rl", "rr")]
        self.wheel_body = [m.body(f"wheel_{n}").id for n in ("fl", "fr", "rl", "rr")]
        self.boom_q = m.joint("boom").qposadr[0]
        self.panel_body = [m.body(f"cover_{i}").id for i in range(P.N_PANELS)]
        self.panel_geom = [m.geom(f"panel_{i}").id for i in range(P.N_PANELS)]
        self.ground_geom = m.geom("ground").id
        self.tyre_geom = [m.geom(f"tyre_{n}").id for n in ("fl", "fr", "rl", "rr")]
        self.station_mocap = m.body_mocapid[m.body("station").id]
        self.sens = {n: (m.sensor(n).adr[0], m.sensor(n).dim[0]) for n in
                     ("accel", "gyro", "gnss", "quat", "boom_pot", "enc_fl", "enc_fr", "enc_rl", "enc_rr")}
        a = 2 * math.pi * P.FILTER_CUTOFF_HZ * P.CTRL_DT
        self.alpha = a / (1 + a)    # first-order low-pass coefficient
        self.renderer = None
        self.np_random_local = np.random.default_rng(0)

    # ------------------------------------------------------------------ helpers
    def _s(self, name):
        a, n = self.sens[name]
        return self.d.sensordata[a:a + n].copy()

    def _terrain(self, rng, amp):
        nr, nc = HF_NROW, HF_NCOL
        yy, xx = np.meshgrid(np.linspace(-1, 1, nr), np.linspace(-1, 1, nc), indexing="ij")
        h = np.zeros((nr, nc))
        for _ in range(8):  # gentle random sinusoid bumps (wavelength 1-4 m)
            kx, ky = rng.uniform(2, 8, 2) * rng.choice([-1, 1], 2)
            h += np.sin(kx * xx + ky * yy + rng.uniform(0, 2 * np.pi))
        h = (h - h.min()) / (np.ptp(h) + 1e-9) * amp
        self.m.hfield_data[:] = (h / HF_SIZE[2]).ravel().astype(np.float32)
        self._hf = h

    def ground_z(self, x, y):
        i = int(np.clip((y - HF_POS[1] + HF_SIZE[1]) / (2 * HF_SIZE[1]) * (HF_NROW - 1), 0, HF_NROW - 1))
        j = int(np.clip((x - HF_POS[0] + HF_SIZE[0]) / (2 * HF_SIZE[0]) * (HF_NCOL - 1), 0, HF_NCOL - 1))
        return float(self._hf[i, j])

    def _apply_dr(self, rng):
        m, b = self.m, self.base
        m.body_mass[:] = b["mass"]; m.body_inertia[:] = b["inertia"]; m.body_ipos[:] = b["ipos"]
        m.geom_friction[:] = b["fric"]; m.actuator_gainprm[:] = b["gain"]; m.actuator_biasprm[:] = b["bias"]
        R = P.DR
        if not self.randomize:
            self.dr = dict(friction=0.8, mass_scale=1.0, payload=0.0, motor=1.0, noise=1.0, obs_delay=0,
                           act_delay=0, terrain=0.01, wind=0.0, gnss_sigma=0.03)
        else:
            u = lambda k: float(rng.uniform(*R[k]))
            self.dr = dict(friction=u("friction"), mass_scale=u("mass_scale"), payload=u("payload_delta"),
                           motor=u("motor_strength"), noise=u("noise_level"),
                           obs_delay=int(rng.integers(R["obs_delay"][0], R["obs_delay"][1] + 1)),
                           act_delay=int(rng.integers(R["act_delay"][0], R["act_delay"][1] + 1)),
                           terrain=u("terrain"), wind=u("wind"), gnss_sigma=float(rng.uniform(*P.GNSS_NOISE)),
                           com=rng.uniform(*R["com_shift"], 3) * np.array([1, 1, 0.3]))
        dr = self.dr
        for gid in [self.ground_geom] + self.tyre_geom:
            m.geom_friction[gid, 0] = dr["friction"]
        ratio = dr["mass_scale"] + dr["payload"] / b["mass"][self.cid]
        m.body_mass[self.cid] = b["mass"][self.cid] * ratio
        m.body_inertia[self.cid] = b["inertia"][self.cid] * ratio
        if "com" in dr:
            m.body_ipos[self.cid] = b["ipos"][self.cid] + dr["com"]
        for wb in self.wheel_body:
            m.body_mass[wb] = b["mass"][wb] * dr["mass_scale"]
        m.actuator_gainprm[:4, 0] = b["gain"][:4, 0] * dr["motor"]
        m.actuator_biasprm[:4, 2] = b["bias"][:4, 2] * dr["motor"]
        self._terrain(rng, dr["terrain"])
        if self.renderer is not None:
            mujoco.mjr_uploadHField(self.m, self.renderer._mjr_context, 0)

    # ------------------------------------------------------------------ gym API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        options = options or {}
        self._apply_dr(rng)
        mujoco.mj_resetData(self.m, self.d)
        self.deploy = bool(options["deploy"]) if "deploy" in options else bool(rng.random() < 0.5)
        self.row = int(options.get("row", rng.integers(len(ROW_YS))))
        self.row_y = ROW_YS[self.row]
        gx = float(options.get("goal_x", rng.uniform(2.0, 5.5)))
        self.goal = np.array([gx, self.row_y - LANE_OFFSET])
        self.goal_yaw = 0.0
        sx, sy = rng.uniform(-3.0, -0.5), rng.uniform(-3.6, 3.6)
        syaw = rng.uniform(-math.pi, math.pi)
        if "start" in options:
            sx, sy, syaw = options["start"]
        z = self.ground_z(sx, sy) + P.WHEEL_RADIUS + 0.03
        self.d.qpos[:3] = [sx, sy, z]
        self.d.qpos[3:7] = [math.cos(syaw / 2), 0, 0, math.sin(syaw / 2)]
        self.d.mocap_pos[self.station_mocap] = [gx, self.goal[1], self.ground_z(gx, self.goal[1]) + 0.005]
        mujoco.mj_forward(self.m, self.d)
        for _ in range(20):  # settle on the terrain
            mujoco.mj_step(self.m, self.d)
        self.t = 0
        self.a_filt = np.zeros(3); self.a_prev = np.zeros(3)
        self.boom_set = 0.0
        self.act_q = collections.deque([np.zeros(3)] * (self.dr["act_delay"] + 1), maxlen=self.dr["act_delay"] + 1)
        self.gust = np.zeros(3); self.gust_t = 0
        self.energy_motor = self.energy_act = 0.0
        self.crop_steps = 0
        self.hold = 0
        self.success = False
        self.t_goal = None
        self.dist_prev = self._true_state()["dist"]
        o = self._obs_now()
        self.obs_q = collections.deque([o] * (self.dr["obs_delay"] + 1), maxlen=self.dr["obs_delay"] + 1)
        return self.obs_q[0].astype(np.float32), {"deploy": self.deploy, "dr": {k: v for k, v in self.dr.items() if k != "com"}}

    def _true_state(self):
        x, y = self.d.qpos[0], self.d.qpos[1]
        yaw = yaw_of(self.d.qpos[3:7])
        dist = float(np.hypot(self.goal[0] - x, self.goal[1] - y))
        return dict(x=x, y=y, yaw=yaw, dist=dist, herr=wrap(self.goal_yaw - yaw),
                    speed=float(np.hypot(*self.d.qvel[:2])))

    def sensors(self):
        """Noisy sensor readings (what the Pi would see). Shared by the PPO obs and the baseline controller."""
        rng, k = self.np_random, self.dr["noise"]
        gnss = self._s("gnss")[:2] + rng.normal(0, self.dr["gnss_sigma"], 2)
        yaw = yaw_of(self._s("quat")) + rng.normal(0, P.HEADING_NOISE * k)
        gyro = self._s("gyro") + rng.normal(0, P.GYRO_NOISE * k, 3)
        acc = self._s("accel") / 9.81 + rng.normal(0, P.ACCEL_NOISE * k, 3)
        enc = np.array([self._s(f"enc_{n}")[0] for n in ("fl", "fr", "rl", "rr")]) + rng.normal(0, P.ENC_NOISE * k, 4)
        boom = float(self._s("boom_pot")[0]) + rng.normal(0, 0.02 * k)
        # mast GNSS antenna is offset from the chassis centre; correct with the noisy yaw
        c, s = math.cos(yaw), math.sin(yaw)
        off = np.array([-0.10, P.DECK_W / 2 - 0.02])
        pos = gnss - np.array([c * off[0] - s * off[1], s * off[0] + c * off[1]])
        return dict(pos=pos, yaw=yaw, gyro=gyro, acc=acc, enc=enc, boom=boom)

    def _obs_now(self):
        s = self.sensors()
        self.last_sensors = s
        dxw, dyw = self.goal - s["pos"]
        c, sn = math.cos(s["yaw"]), math.sin(s["yaw"])
        dxb, dyb = c * dxw + sn * dyw, -sn * dxw + c * dyw
        herr = wrap(self.goal_yaw - s["yaw"])
        wl = (s["enc"][0] + s["enc"][2]) / 2 / P.NO_LOAD_SPEED
        wr = (s["enc"][1] + s["enc"][3]) / 2 / P.NO_LOAD_SPEED
        dist = math.hypot(dxw, dyw)
        o = np.array([np.clip(dxw / 4, -2, 2), np.clip(dyw / 4, -2, 2), np.clip(dxb / 4, -2, 2), np.clip(dyb / 4, -2, 2),
                      min(dist / 4, 2), math.sin(herr), math.cos(herr),
                      (self.goal[0] - ROW_X0) / 4, *(s["gyro"] * 0.5), *s["acc"], wl, wr,
                      s["boom"] / BOOM_MAX, *self.a_prev, 1.0 if self.deploy else -1.0,
                      1 - self.t / MAX_STEPS], dtype=np.float64)
        return o

    def coverage(self):
        """Fraction of cloth panels lying over the target row footprint (low enough to be on the crop)."""
        n = 0
        for b in self.panel_body:
            g = self.d.xpos[b]  # panel top hinge; use geom centre
        for gid in self.panel_geom:
            x, y, z = self.d.geom_xpos[gid]
            if ROW_X0 <= x <= ROW_X1 and abs(y - self.row_y) < ROW_HALF_W + 0.12 and z < ROW_H + 0.25:
                n += 1
        return n / len(self.panel_geom)

    def _wheels_on_crops(self):
        for wb in self.wheel_body:
            x, y = self.d.xpos[wb][:2]
            if ROW_X0 - 0.03 <= x <= ROW_X1 + 0.03:
                for ry in ROW_YS:
                    if abs(y - ry) < ROW_HALF_W + P.WHEEL_WIDTH / 2:
                        return True
        return False

    def step(self, action):
        action = np.clip(np.asarray(action, float), -1, 1)
        self.act_q.append(action)
        a_in = self.act_q[0]                                  # action latency
        self.a_filt = self.alpha * a_in + (1 - self.alpha) * self.a_filt   # low-pass filter
        l, r, b = self.a_filt
        self.d.ctrl[0] = self.d.ctrl[2] = l
        self.d.ctrl[1] = self.d.ctrl[3] = r
        # actuator: command -> boom target, rate-limited by the L16 stroke speed
        target = (b + 1) / 2 * BOOM_MAX
        step_max = BOOM_RATE * P.CTRL_DT
        moved = float(np.clip(target - self.boom_set, -step_max, step_max))
        self.boom_set += moved
        self.d.ctrl[4] = self.boom_set
        # wind gusts on the cloth panels
        if self.gust_t <= 0:
            self.gust = self.np_random.normal(0, 1, 3) * np.array([1, 1, 0.2]) * self.dr["wind"]
            self.gust_t = int(self.np_random.integers(10, 40))
        self.gust_t -= 1
        self.d.xfrc_applied[:] = 0
        ext_now = self.d.qpos[self.boom_q] / BOOM_MAX   # cloth is rolled up when stowed: wind acts as it unrolls
        for pb in self.panel_body:
            self.d.xfrc_applied[pb, :3] = self.gust * float(np.clip(ext_now, 0, 1))
        mujoco.mj_step(self.m, self.d, nstep=N_SUB)
        self.t += 1

        # energy from a DC-motor current estimate
        tau = self.d.actuator_force[:4]
        duty = np.abs(self.d.ctrl[:4])
        i_mot = np.abs(tau) / KT + P.NO_LOAD_CURRENT * (duty > 0.05)
        p_mot = float(np.sum(P.BATT_VOLTAGE * duty * i_mot)) / P.DRIVER_EFF
        e_mot = p_mot * P.CTRL_DT
        e_act = P.BATT_VOLTAGE * P.ACT_CURRENT * P.CTRL_DT * (abs(moved) > 1e-6)
        self.energy_motor += e_mot; self.energy_act += e_act

        st = self._true_state()
        up_z = self.d.xmat[self.cid][8]
        tipped = up_z < 0.5
        oob = not (-5.5 < st["x"] < 9.5 and -5.5 < st["y"] < 5.5)
        on_crops = self._wheels_on_crops()
        self.crop_steps += int(on_crops)
        boom = self.d.qpos[self.boom_q]
        ext = boom / BOOM_MAX

        dist = st["dist"]
        rew = 4.0 * (self.dist_prev - dist)
        self.dist_prev = dist
        near = math.exp(-dist / 0.5)
        rew += 0.3 * math.cos(st["herr"]) * near
        rew -= 0.002 * e_mot
        rew -= 0.05 * float(np.sum((action - self.a_prev) ** 2))
        rew -= 1.0 * on_crops
        rew -= 0.01
        rew -= 0.2 * st["speed"] * near * (dist < 0.3)
        if self.deploy:
            rew += 0.3 * ext * (dist < 0.3) - 0.3 * ext * st["speed"]
        else:
            rew -= 0.3 * ext
        self.a_prev = action.copy()

        aligned = dist < 0.2 and abs(st["herr"]) < 0.2 and st["speed"] < 0.1
        cov = self.coverage() if self.deploy else 0.0
        if aligned and self.t_goal is None:
            self.t_goal = self.t * P.CTRL_DT
        ok = aligned and ((not self.deploy and ext < 0.1) or (self.deploy and ext > 0.85 and cov >= 0.5))
        self.hold = self.hold + 1 if ok else 0
        terminated = False
        if self.hold >= 10:
            self.success = True
            terminated = True
            rew += 20.0 + (10.0 * cov if self.deploy else 0.0)
        if tipped:
            terminated = True; rew -= 20.0
        if oob:
            terminated = True; rew -= 10.0
        truncated = self.t >= MAX_STEPS and not terminated

        self.obs_q.append(self._obs_now())
        obs = self.obs_q[0].astype(np.float32)
        info = {}
        if terminated or truncated:
            info["episode_metrics"] = dict(
                success=float(self.success), deploy=float(self.deploy), time=self.t * P.CTRL_DT,
                time_to_goal=self.t_goal if self.success else float("nan"),
                energy_J=self.energy_motor + self.energy_act + P.BATT_VOLTAGE * P.IDLE_CURRENT * self.t * P.CTRL_DT,
                motor_energy_J=self.energy_motor, crop_steps=self.crop_steps, crop_hit=float(self.crop_steps > 0),
                tipped=float(tipped), oob=float(oob), coverage=self.coverage(), final_dist=dist,
                final_herr=abs(st["herr"]))
        return obs, float(rew), terminated, truncated, info

    # ------------------------------------------------------------------ rendering / export
    def render(self, camera=None):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.m, 540, 960)
            mujoco.mjr_uploadHField(self.m, self.renderer._mjr_context, 0)
        cam = camera or self._chase_cam()
        self.renderer.update_scene(self.d, camera=cam)
        return self.renderer.render()

    def _chase_cam(self):
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        x, y = self.d.qpos[0], self.d.qpos[1]
        cam.lookat[:] = [(x + self.goal[0]) / 2 * 0.6 + x * 0.4, (y + self.goal[1]) / 2, 0.3]
        cam.distance = 5.5
        cam.azimuth = -60
        cam.elevation = -28
        return cam

    def poses(self):
        """Per-step body poses for trajectory export (pos xyz + quat wxyz)."""
        names = ["chassis"] + [f"wheel_{n}" for n in ("fl", "fr", "rl", "rr")] + ["boom", "roll"] + \
                [f"cover_{i}" for i in range(P.N_PANELS)]
        out = {}
        for n in names:
            bid = self.m.body(n).id
            out[n] = [round(float(v), 4) for v in (*self.d.xpos[bid], *self.d.xquat[bid])]
        return out


def make_env(rank: int = 0, randomize: bool = True):
    def _f():
        e = FrostRoverEnv(randomize=randomize)
        e.reset(seed=rank * 1000 + 7)
        return e
    return _f


if __name__ == "__main__":
    import time
    env = FrostRoverEnv()
    o, info = env.reset(seed=0)
    print("obs", o.shape, info)
    t = time.time(); n = 0
    for ep in range(3):
        env.reset(seed=ep)
        done = False
        while not done:
            o, r, te, tr, inf = env.step(env.action_space.sample())
            n += 1; done = te or tr
        print(inf["episode_metrics"])
    print("env steps/s", n / (time.time() - t))
