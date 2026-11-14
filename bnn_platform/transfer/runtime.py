"""Runtime bridge: run an exported bundle in closed loop against an MEA backend and a robot.

    MEABackend  -- stimulate(pattern) / read_spikes(window_ms) / close()
        MockMEABackend     simulated Izhikevich culture (trained source culture, or a naive one)
        CL1Backend         Cortical Labs `cl` API (tested against the cl-sdk simulator only)
        MaxWellBackend     stub -> NotImplementedError with integration pointers
        MCSBackend         stub -> NotImplementedError with integration pointers
    RobotIO     -- reset / observe / apply / teacher_action / hold
        SurgicalEnvRobotIO     MuJoCo surgical arm (surgical/reach_env.py), as in legacy training
        StudioKinematicRobotIO MuJoCo Studio robot (bnn_platform/studio), as in Studio training
        UDPJSONRobotIO         real robot over UDP JSON (implemented, NOT tested on hardware)
        ROS2RobotIO            stub -> NotImplementedError with the topic contract

One closed-loop step (times from the bundle's manifest.timing):
    observe robot -> encoder pulses (0..stim_ms) -> count spikes (0..resp_ms) -> decoder -> safety
    -> robot joint-position delta -> [teach: paired stimulation block, teach_ms]
"""
from __future__ import annotations

import abc
import copy
import csv
import json
import pathlib
import socket
import time
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from .source import ROOT, TransferError

Pulse = tuple  # (t_ms: int, electrode: int, amplitude: float, kind: str)


# =============================================================================================
# Bundle
# =============================================================================================
class Bundle:
    def __init__(self, path: str | pathlib.Path):
        p = pathlib.Path(path)
        if not p.is_absolute() and not p.exists():
            p = ROOT / p
        if not (p / "manifest.json").exists():
            raise TransferError(f"{path}: not a bundle (no manifest.json)")
        self.dir = p.resolve()

        def J(name):
            return json.loads((p / name).read_text(encoding="utf-8"))
        self.manifest = J("manifest.json")
        self.encoder = J("encoder.json")
        self.decoder = J("decoder.json")
        self.training = J("training_protocol.json")
        self.calibration = J("calibration_protocol.json")
        self.safety = J("safety.json")
        with open(p / "electrode_map.csv", newline="", encoding="utf-8") as fh:
            self.electrodes = list(csv.DictReader(fh))
        self.timing = self.manifest["timing"]
        self.target = self.manifest["target"]["id"]
        self.n_joints = len(self.manifest["action_space"]["joints"])
        self.usable = [int(r["electrode_id"]) for r in self.electrodes if r["role"] != "unavailable"]
        self.sim_channel = {int(r["electrode_id"]): int(r["sim_channel"]) for r in self.electrodes}
        self.roles = {r["role_name"]: int(r["electrode_id"]) for r in self.electrodes if r["role_name"]}


@dataclass
class RoleMap:
    """role name -> electrode id. Role names: err+x err-x err+y err-y err+z err-z, ctx:<k>,
    motor+:<k>, motor-:<k>."""
    roles: dict
    n_joints: int

    def err(self):
        return [self.roles[f"err{s}{a}"] for a in "xyz" for s in "+-"]

    def ctx(self, k):
        return self.roles.get(f"ctx:{k}")

    def pos(self, k):
        return self.roles[f"motor+:{k}"]

    def neg(self, k):
        return self.roles[f"motor-:{k}"]

    def sensory(self):
        return [e for n, e in self.roles.items() if n.startswith(("err", "ctx"))]

    def motor(self):
        return [e for n, e in self.roles.items() if n.startswith("motor")]


# =============================================================================================
# Encoder / decoder / teacher (pure functions of the bundle)
# =============================================================================================
class Encoder:
    def __init__(self, spec: dict, roles: RoleMap):
        self.spec = spec
        self.roles = roles
        self.spacing = int(spec["pulse_timing"]["spacing_ms"])
        self.amp = float(spec["waveform"]["amplitude"])

    def levels(self, obs: dict) -> dict:
        lv = {}
        for ch in self.spec["channels"]:
            e = self.roles.roles[ch["role_name"]]
            v = float(np.asarray(obs[ch["source"]])[ch["index"]])
            if ch["transform"] == "threshold_count":
                active = (v >= 0) if ch["sign"] > 0 else (v < 0)
                n = int(np.sum(abs(v) >= np.asarray(ch["thresholds"]))) if active else 0
            elif ch["transform"] == "round_scale":
                n = int(np.clip(np.round(ch["scale"] * v), 0, ch["max_pulses"]))
            else:
                raise TransferError(f"unknown encoder transform {ch['transform']}")
            lv[e] = min(n, int(ch["max_pulses"]))
        return lv

    def pattern(self, levels: dict) -> list:
        return [(k * self.spacing, e, self.amp, "encode") for e, n in levels.items() for k in range(n)]


class Decoder:
    """z = W @ (counts - offset) + bias ; a = clip(sign(z) * max(|z| - deadzone, 0) / gain, -1, 1)"""

    def __init__(self, spec: dict):
        self.spec = spec
        self.inputs = [int(e) for e in spec["input"]["electrodes"]]
        self.W = np.asarray(spec["weights"], float)
        self.b = np.asarray(spec["bias"], float)
        self.off = np.asarray(spec["input_offset"], float)
        self.deadzone = float(spec["deadzone"])
        self.gain = float(spec["gain"])
        self.window = spec["input"]["window_ms"]

    def counts_vector(self, counts: dict) -> np.ndarray:
        return np.array([counts.get(e, 0) for e in self.inputs], float)

    def __call__(self, counts: dict) -> np.ndarray:
        z = self.W @ (self.counts_vector(counts) - self.off) + self.b
        z = np.sign(z) * np.maximum(np.abs(z) - self.deadzone, 0)
        return np.clip(z / self.gain, -1, 1)

    @staticmethod
    def pool_weights(inputs: list, roles: RoleMap) -> np.ndarray:
        idx = {e: i for i, e in enumerate(inputs)}
        W = np.zeros((roles.n_joints, len(inputs)))
        for k in range(roles.n_joints):
            W[k, idx[roles.pos(k)]] = 1.0
            W[k, idx[roles.neg(k)]] = -1.0
        return W

    def to_spec(self) -> dict:
        s = copy.deepcopy(self.spec)
        s.update(weights=np.round(self.W, 6).tolist(), bias=np.round(self.b, 6).tolist(),
                 input_offset=np.round(self.off, 6).tolist(), deadzone=self.deadzone, gain=self.gain)
        s["input"]["electrodes"] = list(self.inputs)
        return s


def teach_pattern(tp: dict, roles: RoleMap, levels: dict, dq: np.ndarray, amp: float) -> list:
    """Paired stimulation (training_protocol.json 'teach'): for joints with |dq_k| >= frac*max|dq|,
    per volley v (t = start + v*period): anti-pool at t-d (post->pre: LTD), error electrodes with
    level > v at t, pro-pool at t+d (pre->post: LTP)."""
    t = tp["teach"]
    dq = np.asarray(dq, float)
    mx = float(np.abs(dq).max()) if dq.size else 0.0
    if mx < 1e-9:
        return []
    good, bad = [], []
    for k in range(roles.n_joints):
        if abs(dq[k]) < t["select_frac"] * mx:
            continue
        good.append(roles.pos(k) if dq[k] > 0 else roles.neg(k))
        bad.append(roles.neg(k) if dq[k] > 0 else roles.pos(k))
    out = []
    d = int(t["pre_post_ms"])
    for v in range(int(t["volleys"])):
        tv = int(t["volley_start_ms"]) + v * int(t["volley_period_ms"])
        sens = [e for e in roles.err() if levels.get(e, 0) > v]
        if not sens:
            continue
        out += [(tv - d, e, amp, "teach_anti") for e in bad]
        out += [(tv, e, amp, "teach_sensory") for e in sens]
        out += [(tv + d, e, amp, "teach_pro") for e in good]
    return out


# =============================================================================================
# MEA backends
# =============================================================================================
class MEABackend(abc.ABC):
    """Contract every MEA rig implements.

    Time model: stimulate() queues pulses with times relative to the start of the NEXT read window;
    read_spikes(window_ms) advances the device by window_ms, delivers the queued pulses on the device
    clock and returns the detected spikes [(t_ms_from_window_start, electrode_id)].
    Electrode ids are the bundle's logical electrode ids (electrode_map.csv electrode_id)."""

    name = "abstract"
    biology = "unknown"
    has_stim_artifacts = True

    def open(self):
        return self

    @abc.abstractmethod
    def stimulate(self, pattern: Iterable[Pulse]) -> None: ...

    @abc.abstractmethod
    def read_spikes(self, window_ms: int) -> list: ...

    def set_plasticity(self, on: bool) -> None:
        """Only meaningful for simulated cultures; a living culture is always plastic."""

    def close(self) -> None:
        pass

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()


class MockMEABackend(MEABackend):
    """Simulated culture (neurorover.culture.core.Culture). The bundle's electrode -> physical sim
    channel map (electrode_map.csv sim_channel) is applied, so remapped roles really stimulate a
    different place in the simulated dish, and unavailable electrodes neither stimulate nor record."""

    name = "mock"
    biology = "simulated"
    has_stim_artifacts = False

    def __init__(self, culture, bundle: Bundle, nominal_amp: float):
        self.c = culture
        self.map = {e: bundle.sim_channel[e] for e in bundle.usable}
        self.rev = {s: e for e, s in self.map.items()}
        self.nominal = nominal_amp
        self.queue: list = []
        self.dropped = 0

    def stimulate(self, pattern):
        for t, e, amp, *_ in pattern:
            if e in self.map:
                self.queue.append((int(t), self.map[e], float(amp)))
            else:
                self.dropped += 1

    def read_spikes(self, window_ms):
        sched: dict = {}
        for t, ch, amp in self.queue:
            sched.setdefault(t, []).append((ch, amp))
        self.queue = [(t - window_ms, ch, a) for t, ch, a in self.queue if t >= window_ms]
        out = []
        c, rev = self.c, self.rev
        for t in range(window_ms):
            for ch, amp in sched.get(t, []):
                c.stim(ch, amp / self.nominal)
            f = c.step()
            if f.size:
                for ch in c.ch[f]:
                    e = rev.get(int(ch))
                    if e is not None:
                        out.append((t, e))
        return out

    def set_plasticity(self, on):
        self.c.rule = "teach" if on else "frozen"


class CL1Backend(MEABackend):
    """Cortical Labs `cl` API. With simulate=True the trained/naive Izhikevich culture is registered
    as the cl-sdk simulator data source (bnn_platform/transfer/cl_source.py); with simulate=False
    nothing is registered and the same code drives a CL1 / Cortical Cloud device.
    Pulses are one-shot StimPlans on the 25 kHz device clock (as neurorover/run_cl.py)."""

    name = "cl1"
    FPS_MS = 25

    def __init__(self, bundle: Bundle, simulate: bool = True, culture_pkl: str | None = None,
                 rule: str = "frozen", tick_ms: int = 5, seed: int = 0):
        self.bundle = bundle
        self.simulate = simulate
        self.biology = "simulated" if simulate else "living"
        self.has_stim_artifacts = not simulate
        self.culture_pkl = culture_pkl
        self.rule = rule
        self.tick_ms = tick_ms
        self.seed = seed
        wf = bundle.encoder["waveform"]
        self.phase_us = int(wf["phase_width_us"])
        self.queue: list = []
        self._ctx = None

    def open(self):
        import os
        os.environ.setdefault("CL_SDK_VISUALISATION", "0")
        if self.simulate:
            os.environ["CL_SDK_ACCELERATED_TIME"] = "1"
        import cl
        self.cl = cl
        if self.simulate:
            from .cl_source import register
            register(culture_pkl=self.culture_pkl, rule=self.rule, seed=self.seed)
        self._ctx = cl.open()
        self.n = self._ctx.__enter__()
        self.ticks = iter(self.n.loop(ticks_per_second=1000 / self.tick_ms, ignore_jitter=True))
        self.tick_frames = self.tick_ms * self.FPS_MS
        return self

    def stimulate(self, pattern):
        self.queue.extend(pattern)

    def _design(self, amp):
        a = float(abs(amp))
        return self.cl.StimDesign(self.phase_us, -a, self.phase_us, a)

    def read_spikes(self, window_ms):
        tick = next(self.ticks)
        T = tick.iteration_timestamp
        m0 = -(-(T + 2 * self.tick_frames) // self.FPS_MS)
        f0 = m0 * self.FPS_MS
        for t, e, amp, *_ in sorted(self.queue, key=lambda p: p[0]):
            if t >= window_ms:
                continue
            plan = self.n.create_stim_plan()
            plan.stim(self.cl.ChannelSet(int(e)), self._design(amp))
            plan.run(at_timestamp=int(f0 + t * self.FPS_MS))
        self.queue = [(t - window_ms, e, a, *k) for t, e, a, *k in self.queue if t >= window_ms]
        end = f0 + window_ms * self.FPS_MS
        out = []
        while True:
            tick = next(self.ticks)
            for s in tick.analysis.spikes:
                t = (s.timestamp - f0) // self.FPS_MS
                if 0 <= t < window_ms:
                    out.append((int(t), int(s.channel)))
            if tick.iteration_next_timestamp >= end + self.tick_frames:
                return out

    def close(self):
        if self._ctx is not None:
            self._ctx.__exit__(None, None, None)
            self._ctx = None


class MaxWellBackend(MEABackend):
    name = "maxwell"
    biology = "living"

    def __init__(self, *a, **k):
        raise NotImplementedError(
            "MaxWell backend is a stub (stub-needs-vendor-sdk). Implement with the MaxLab Live `maxlab` "
            "Python API: route electrode_map.csv hw_record_electrodes (<=1024) with maxlab.chip.Array "
            "select_electrodes()/route(), connect each hw_stim_electrode to a StimulationUnit "
            "(<=32), build pulses as maxlab.Sequence DAC commands (biphasic, encoder.json waveform), and "
            "read spikes from the MaxLab Live real-time stream (C++ closed-loop API for <10 ms latency). "
            "Sum spikes over each region's 16 electrodes after deduplication. See docs/SIM_TO_REAL.md.")

    def stimulate(self, pattern): ...

    def read_spikes(self, window_ms): ...


class MCSBackend(MEABackend):
    name = "mcs"
    biology = "living"

    def __init__(self, *a, **k):
        raise NotImplementedError(
            "MCS MEA2100 backend is a stub (stub-needs-vendor-sdk). Implement with McsUsbNet (.NET, via "
            "pythonnet): CMeaDeviceNet for the data stream (spike detection on the MEA2100 DSP or in "
            "software), CStg200xDownloadNet / the MEA2100 stimulus generator for biphasic pulses on the "
            "electrode labels in electrode_map.csv, or use Multi Channel Experimenter's real-time "
            "feedback. Map labels -> HW channels with your channel layout. See docs/SIM_TO_REAL.md.")

    def stimulate(self, pattern): ...

    def read_spikes(self, window_ms): ...


# =============================================================================================
# Robot IO
# =============================================================================================
class RobotIO(abc.ABC):
    """observe() returns {'ee_error_base_m': (3,), 'joint_pos_norm01': (n,), 'tip': (3,), 'target': (3,),
    'err_m': float}. apply(a) takes normalised joint-position deltas in [-1, 1] (scaled by
    manifest.action_space.joints[k].step_scale) and returns done. teacher_action() is the direction
    that reduces the error (J^T e, in action units) -- on a real robot compute it from the robot's own
    kinematics (URDF + current joint state)."""

    success = False

    @abc.abstractmethod
    def reset(self, episode: int) -> None: ...

    @abc.abstractmethod
    def observe(self) -> dict: ...

    @abc.abstractmethod
    def apply(self, action: np.ndarray) -> bool: ...

    @abc.abstractmethod
    def teacher_action(self) -> np.ndarray: ...

    @abc.abstractmethod
    def hold(self) -> None: ...

    def body_poses(self):
        return None

    def close(self):
        pass


def _held_out_targets(robot_sim: dict, n: int):
    if robot_sim["kind"] == "surgical_env":
        from surgical.arm import load_model, make_targets
        return make_targets(load_model(), n, seed=999)
    from bnn_platform.studio.robots import load_robot
    rm = load_robot(robot_sim["robot"])
    return rm.sample_targets(n, np.random.default_rng(999))


class SurgicalEnvRobotIO(RobotIO):
    def __init__(self, robot_sim: dict, n_targets: int = 50):
        from surgical.reach_env import SurgicalReachEnv
        self.targets = _held_out_targets(robot_sim, n_targets)
        self.env = SurgicalReachEnv(seed=0, target_pool=self.targets)   # skip the 2000-target training pool
        self.done = False

    def reset(self, episode):
        self.env.reset(seed=episode, options={"target": self.targets[episode % len(self.targets)]})
        self.success = False
        self.dist = float(np.linalg.norm(self.env.err()))

    def observe(self):
        e = self.env
        return dict(ee_error_base_m=e.err_arm_frame(), joint_pos_norm01=(e.q() - e.ix.lo) / (e.ix.hi - e.ix.lo),
                    tip=e.ix.tip(e.d), target=e.target.copy(), err_m=float(np.linalg.norm(e.err())),
                    q=e.q())

    def apply(self, a):
        _, _, term, trunc, info = self.env.step(a)
        self.success = bool(info["success"])
        self.dist = info["dist"]
        return bool(term or trunc)

    def teacher_action(self):
        e = self.env
        return (e.ix.jac(e.m, e.d).T @ e.err()) / e.scale

    def hold(self):
        e = self.env
        e.qref = np.clip(e.q(), e.ix.lo, e.ix.hi)
        e.a_f[:] = 0

    def body_poses(self):
        d = self.env.d
        return np.round(np.concatenate([d.xpos, d.xquat], 1), 5).tolist()


class StudioKinematicRobotIO(RobotIO):
    def __init__(self, robot_sim: dict, steps: int, n_targets: int = 50):
        from bnn_platform.studio.bnn import KinematicReach
        from bnn_platform.studio.robots import load_robot
        self.rm = load_robot(robot_sim["robot"])
        self.task = KinematicReach(self.rm, steps=steps)
        self.targets = self.rm.sample_targets(n_targets, np.random.default_rng(999))

    def reset(self, episode):
        self.task.reset(self.targets[episode % len(self.targets)])
        self.success = False

    def observe(self):
        t = self.task
        return dict(ee_error_base_m=t.err_base(), joint_pos_norm01=t.qn01(), tip=t.tip(), target=t.target.copy(),
                    err_m=float(t.dist), q=t.qref.copy())

    def apply(self, a):
        done = self.task.step(a)
        self.success = bool(self.task.success)
        return done

    def teacher_action(self):
        return self.task.dq_star()

    def hold(self):
        self.task.a_f[:] = 0

    def body_poses(self):
        return self.task.body_poses()


class UDPJSONRobotIO(RobotIO):
    """Real robot over UDP JSON (one datagram per message). NOT tested against hardware.

    robot -> bridge (state, sent every control period and in reply to commands):
        {"seq": int, "q": [n joint positions, rad|m], "ee": [x,y,z] (m, robot base frame),
         "target": [x,y,z] (m, base frame), "jacobian": [[3 x n]] (m per joint unit, optional),
         "ok": true}
    bridge -> robot:
        {"seq": int, "cmd": "reset", "episode": k}
        {"seq": int, "cmd": "move", "dq": [n joint position deltas, rad|m]}   (already safety-limited)
        {"seq": int, "cmd": "hold"}
    The robot controller must itself enforce joint limits, velocity limits and its own e-stop; the
    robot must hold position if no command arrives within safety.json robot.watchdog_timeout_ms."""

    def __init__(self, manifest: dict, host="127.0.0.1", port=17800, listen_port=17801, timeout_s=0.5):
        self.joints = manifest["action_space"]["joints"]
        self.scale = np.array([j["step_scale"] for j in self.joints])
        self.lo = np.array([j["range"][0] for j in self.joints])
        self.hi = np.array([j["range"][1] for j in self.joints])
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", listen_port))
        self.sock.settimeout(timeout_s)
        self.seq = 0
        self.state = None
        self.success_r = manifest["task"].get("success_radius_mm", 10.0) / 1000

    def _send(self, msg):
        self.seq += 1
        msg["seq"] = self.seq
        self.sock.sendto(json.dumps(msg).encode(), self.addr)
        data, _ = self.sock.recvfrom(65536)   # socket.timeout -> caught by the watchdog
        self.state = json.loads(data)
        return self.state

    def reset(self, episode):
        self._send({"cmd": "reset", "episode": int(episode)})

    def observe(self):
        s = self.state
        q, ee, tg = np.array(s["q"]), np.array(s["ee"]), np.array(s["target"])
        e = tg - ee
        return dict(ee_error_base_m=e, joint_pos_norm01=(q - self.lo) / (self.hi - self.lo), tip=ee, target=tg,
                    err_m=float(np.linalg.norm(e)), q=q)

    def apply(self, a):
        self._send({"cmd": "move", "dq": (np.clip(a, -1, 1) * self.scale).tolist()})
        self.success = self.observe()["err_m"] < self.success_r
        return False

    def teacher_action(self):
        J = np.array(self.state.get("jacobian") or np.zeros((3, len(self.scale))))
        return (J.T @ self.observe()["ee_error_base_m"]) / self.scale

    def hold(self):
        self._send({"cmd": "hold"})

    def close(self):
        self.sock.close()


class ROS2RobotIO(RobotIO):
    def __init__(self, *a, **k):
        raise NotImplementedError(
            "ROS 2 robot bridge is a stub. Contract: subscribe sensor_msgs/JointState on /joint_states "
            "(controlled joints from manifest.action_space.joints), geometry_msgs/PointStamped on "
            "/bnn/target and the EE pose from tf2 (base frame = manifest.observation_space frame); publish "
            "std_msgs/Float64MultiArray joint position targets on /bnn/joint_position_targets (or "
            "trajectory_msgs/JointTrajectory to your controller) after safety limiting; publish "
            "std_msgs/Bool /bnn/hold on any watchdog/e-stop. Teacher direction J^T e from the URDF via "
            "your kinematics library (e.g. KDL / Pinocchio). See docs/SIM_TO_REAL.md.")

    def reset(self, episode): ...

    def observe(self): ...

    def apply(self, action): ...

    def teacher_action(self): ...

    def hold(self): ...


def make_robot(bundle: Bundle, kind: str = "sim", **kw) -> RobotIO:
    rs = bundle.manifest["robot"]["sim"]
    if kind == "sim":
        if rs["kind"] == "surgical_env":
            return SurgicalEnvRobotIO(rs)
        return StudioKinematicRobotIO(rs, steps=int(bundle.manifest["task"]["steps_per_episode"]))
    if kind == "udp":
        return UDPJSONRobotIO(bundle.manifest, **kw)
    if kind == "ros2":
        return ROS2RobotIO()
    raise TransferError(f"unknown robot io {kind!r} (sim|udp|ros2)")


# =============================================================================================
# Safety
# =============================================================================================
class SafetyMonitor:
    def __init__(self, safety: dict, bundle: Bundle):
        s = safety["stimulation"]
        self.min_isi_ms = 1000.0 / float(s["max_burst_hz"])
        self.max_rate = float(s["max_mean_rate_hz_per_electrode"])
        self.rate_window_ms = float(s["rate_window_ms"])
        self.max_amp = float(s["max_amplitude"])
        self.allowed = set(bundle.usable)
        r = safety["robot"]
        self.max_step = np.array(r["max_step_normalised"], float)
        self.lo = np.array(r["joint_soft_limits"]["lo"], float)
        self.hi = np.array(r["joint_soft_limits"]["hi"], float)
        self.ws = r.get("workspace")
        self.watchdog_ms = float(r["watchdog_timeout_ms"])
        self.max_trips = int(r["max_consecutive_faults"])
        self.history: dict = {}
        self.t_ms = 0.0
        self.events: list = []
        self.n_dropped = 0
        self.faults = 0

    def advance(self, ms):
        self.t_ms += ms

    def filter_stims(self, pattern):
        """Drop pulses that would violate amplitude / per-electrode rate limits (never raises)."""
        out = []
        for p in sorted(pattern, key=lambda p: p[0]):
            t, e, amp = p[0], p[1], p[2]
            ta = self.t_ms + t
            h = self.history.setdefault(e, [])
            while h and h[0] < ta - self.rate_window_ms:
                h.pop(0)
            reason = None
            if e not in self.allowed:
                reason = "electrode_not_allowed"
            elif abs(amp) > self.max_amp:
                reason = "amplitude"
            elif h and ta - h[-1] < self.min_isi_ms:
                reason = "burst_rate"
            elif len(h) + 1 > self.max_rate * self.rate_window_ms / 1000:
                reason = "mean_rate"
            if reason:
                self.n_dropped += 1
                self.events.append(dict(event="stim_dropped", reason=reason, electrode=int(e)))
                continue
            h.append(ta)
            out.append(p)
        return out

    def limit_action(self, a, obs):
        """-> (action, reason or None). Non-finite -> hold. Clip per-joint step. Hold if the EE has
        left the workspace or a joint is outside its soft limits and the action pushes further out."""
        a = np.asarray(a, float)
        if not np.all(np.isfinite(a)):
            return np.zeros_like(a), "non_finite_action"
        a = np.clip(a, -self.max_step, self.max_step)
        q = obs.get("q")
        if q is not None:
            out_lo = (q <= self.lo) & (a < 0)
            out_hi = (q >= self.hi) & (a > 0)
            a = np.where(out_lo | out_hi, 0.0, a)
        if self.ws is not None and obs.get("tip") is not None:
            tip = np.asarray(obs["tip"])
            if np.any(tip < np.asarray(self.ws["min"])) or np.any(tip > np.asarray(self.ws["max"])):
                return np.zeros_like(a), "workspace"
        return a, None


# =============================================================================================
# Calibration (living-culture protocol, run identically on the mock)
# =============================================================================================
def _counts(spikes, t0, t1, exclude=()):
    c: dict = {}
    for t, e in spikes:
        if t0 <= t < t1 and e not in exclude:
            c[e] = c.get(e, 0) + 1
    return c


def calibrate(backend: MEABackend, bundle: Bundle, roles: RoleMap, robot: RobotIO, encoder: Encoder,
              decoder: Decoder, safety: SafetyMonitor, emit, quick: bool = True, rng=None) -> dict:
    cp = bundle.calibration
    stg = cp["mock_compressed"] if quick else cp
    rng = rng or np.random.default_rng(0)
    report: dict = {"quick": quick}
    usable = bundle.usable
    backend.set_plasticity(False)

    # 1. baseline
    base_ms = int(stg["baseline"]["duration_s"] * 1000)
    win = int(cp["probe"]["response_window_ms"][1])
    blank = int(cp["probe"]["response_window_ms"][0])
    sp = backend.read_spikes(base_ms)
    safety.advance(base_ms)
    rate = {e: 0.0 for e in usable}
    for _, e in sp:
        rate[e] = rate.get(e, 0) + 1000.0 / base_ms
    thr_hz = cp["baseline"]["active_rate_hz"]
    active = [e for e in usable if rate.get(e, 0) >= thr_hz]
    nbins = max(1, base_ms // win)
    totals = np.zeros(nbins)
    for t, _ in sp:
        if t // win < nbins:
            totals[t // win] += 1
    base_mean, base_sd = float(totals.mean()), float(totals.std() + 1e-9)
    report["baseline"] = dict(duration_ms=base_ms, active=active, rate_hz={str(e): round(r, 3) for e, r in rate.items()},
                              network_spikes_per_window_mean=round(base_mean, 3))
    emit({"type": "calibration", "stage": "baseline", "duration_s": base_ms / 1000, "n_active": len(active),
          "n_electrodes": len(usable), "mean_rate_hz": round(float(np.mean(list(rate.values()))), 3)})

    # 2. probe sweep: single pulses, electrodes in random order, isi_ms apart
    reps, isi = int(stg["probe"]["repetitions"]), int(stg["probe"]["isi_ms"])
    amp = float(bundle.encoder["waveform"]["amplitude"])
    evoked_total = {e: [] for e in usable}
    recv = {e: 0.0 for e in usable}
    sens_set = set(roles.sensory())
    for r in range(reps):
        for e in rng.permutation(usable):
            e = int(e)
            pat = safety.filter_stims([(0, e, amp, "probe")])
            backend.stimulate(pat)
            s = backend.read_spikes(isi)
            safety.advance(isi)
            c = _counts(s, blank, win, exclude=(e,))
            evoked_total[e].append(sum(c.values()))
            if e in sens_set:
                for k, v in c.items():
                    recv[k] = recv.get(k, 0) + v / (reps * max(1, len(sens_set)))
    z_thr = cp["probe"]["responsive_z"]
    base_win = base_mean * (win - blank) / win
    responsive, zs = [], {}
    for e in usable:
        x = np.array(evoked_total[e], float)
        z = (x.mean() - base_win) / (base_sd / np.sqrt(len(x)) + 1e-9)
        zs[e] = float(z)
        if z >= z_thr and x.mean() - base_win >= cp["probe"]["min_evoked_spikes"]:
            responsive.append(e)
    report["probe"] = dict(repetitions=reps, isi_ms=isi, responsive=responsive,
                           z={str(e): round(v, 2) for e, v in zs.items()},
                           recipient_rate={str(e): round(v, 3) for e, v in recv.items()})
    emit({"type": "calibration", "stage": "probe", "n_responsive": len(responsive), "n_electrodes": len(usable),
          "repetitions": reps})

    # 3. remap roles onto responsive (sensory) / receiving (motor) electrodes
    xy = {int(r["electrode_id"]): (float(r["x_um"]), float(r["y_um"])) for r in bundle.electrodes}
    recv_thr = cp["remap"]["min_recipient_spikes_per_probe"]
    changes = []
    taken = set(roles.roles.values())
    for name in sorted(roles.roles):
        e = roles.roles[name]
        ok = (e in responsive) if name.startswith(("err", "ctx")) else (recv.get(e, 0) >= recv_thr or e in active)
        if ok:
            continue
        cands = [c for c in usable if c not in taken and
                 ((c in responsive) if name.startswith(("err", "ctx")) else recv.get(c, 0) >= recv_thr)]
        if not cands:
            changes.append(dict(role=name, electrode=e, action="kept (no responsive free electrode)"))
            continue
        x0, y0 = xy[e]
        best = min(cands, key=lambda c: (xy[c][0] - x0) ** 2 + (xy[c][1] - y0) ** 2)
        roles.roles[name] = best
        taken.discard(e)
        taken.add(best)
        changes.append(dict(role=name, from_electrode=e, to_electrode=best))
    report["remap"] = changes
    emit({"type": "calibration", "stage": "remap", "changes": changes, "n_changed": sum("to_electrode" in c for c in changes)})

    # 4. guided trials -> ridge refit of the decoder toward the teacher direction
    gt = stg["decoder_refit"]
    X, Y = [], []
    trial, resp = int(bundle.timing["trial_ms"]), int(bundle.timing["response_window_ms"])
    for ep in range(int(gt["guided_episodes"])):
        robot.reset(25 + ep)   # guided targets 25.. (closed-loop tests use 0..)
        for _ in range(int(gt["steps_per_episode"])):
            obs = robot.observe()
            lv = encoder.levels(obs)
            backend.stimulate(safety.filter_stims(encoder.pattern(lv)))
            s = backend.read_spikes(trial)
            safety.advance(trial)
            X.append(decoder.counts_vector(_counts(s, 0, resp)))
            dq = robot.teacher_action()
            y = dq / (np.abs(dq).max() + 1e-9)
            Y.append(y)
            a, _ = safety.limit_action(gt["guide_gain"] * y, obs)
            if robot.apply(a):
                break
    X, Y = np.array(X), np.array(Y)
    feat = [i for i, e in enumerate(decoder.inputs) if e not in set(roles.sensory())]
    W0 = Decoder.pool_weights(decoder.inputs, roles) / decoder.gain
    n = len(X)
    ntr = max(2, int(0.75 * n))
    mu = X[:ntr].mean(0)
    Xc = X - mu
    lam = float(gt["ridge_lambda"])
    A = Xc[:ntr][:, feat]
    W = np.zeros_like(W0)
    W[:, feat] = np.linalg.solve(A.T @ A + lam * np.eye(len(feat)), A.T @ Y[:ntr] + lam * W0[:, feat].T).T

    def r2(Wm, rows, off, dz=0.0, g=1.0):
        z = (X[rows] - off) @ Wm.T
        P = np.clip(np.sign(z) * np.maximum(np.abs(z) - dz, 0) / g, -1, 1)
        ss = ((Y[rows] - Y[rows].mean(0)) ** 2).sum()
        return float(1 - ((Y[rows] - P) ** 2).sum() / (ss + 1e-9))

    tr, ho = slice(0, ntr), slice(ntr, n)
    prior_ho = r2(Decoder.pool_weights(decoder.inputs, roles), ho, 0.0, decoder.deadzone, decoder.gain)
    fit_tr, fit_ho = r2(W, tr, mu), r2(W, ho, mu)
    accept = n - ntr >= 2 and fit_ho > max(prior_ho, gt["min_holdout_r2"])
    if accept:
        decoder.W, decoder.b, decoder.off, decoder.deadzone, decoder.gain = W, np.zeros(len(W)), mu, 0.0, 1.0
        decoder.spec["provenance"] = "ridge refit on guided trials (calibration)"
    else:
        decoder.W = Decoder.pool_weights(decoder.inputs, roles)  # remapped sim pool-difference prior
    report["decoder"] = dict(n_trials=n, r2_train=round(fit_tr, 3), r2_holdout=round(fit_ho, 3),
                             r2_holdout_sim_decoder=round(prior_ho, 3), accepted=bool(accept))
    emit({"type": "calibration", "stage": "decoder", **report["decoder"]})
    return report


# =============================================================================================
# Closed loop
# =============================================================================================
@dataclass
class Recorder:
    spikes: list = field(default_factory=list)   # (t_ms, electrode)
    stims: list = field(default_factory=list)    # (t_ms, electrode, amplitude, kind)
    steps: list = field(default_factory=list)    # dicts per control step
    t_ms: int = 0

    def add(self, spikes, stims, window_ms):
        self.spikes += [(self.t_ms + t, e) for t, e in spikes]
        self.stims += [(self.t_ms + p[0], p[1], p[2], p[3]) for p in stims]
        self.t_ms += window_ms


class ClosedLoop:
    def __init__(self, bundle: Bundle, backend: MEABackend, robot: RobotIO, emit=None,
                 teach: bool = True, frames_per_s: float = 10.0, record: Recorder | None = None,
                 robot_limits: bool = True):
        self.b = bundle
        self.backend = backend
        self.robot = robot
        self.emit = emit or (lambda o: None)
        self.teach = teach
        self.frame_dt = 1.0 / frames_per_s if frames_per_s > 0 else None
        self.rec = record
        self.robot_limits = robot_limits   # False only for the sim-faithful reference recording
        self.roles = RoleMap(dict(bundle.roles), bundle.n_joints)
        self.encoder = Encoder(bundle.encoder, self.roles)
        self.decoder = Decoder(copy.deepcopy(bundle.decoder))
        self.safety = SafetyMonitor(bundle.safety, bundle)
        self.teach_amp = float(bundle.encoder["waveform"]["teach_amplitude"])
        self._last_frame = 0.0
        self.grid_index = {e: bundle.sim_channel[e] for e in bundle.usable}

    def _read(self, pattern, window_ms):
        """Stimulate + read with watchdog; returns spikes or None on fault (robot is held)."""
        pattern = self.safety.filter_stims(pattern)
        t0 = time.perf_counter()
        try:
            self.backend.stimulate(pattern)
            sp = self.backend.read_spikes(window_ms)
        except Exception as e:  # noqa: BLE001
            self.safety.faults += 1
            self.robot.hold()
            self.emit({"type": "safety", "event": "backend_fault", "message": f"{type(e).__name__}: {e}",
                       "action": "hold"})
            return None
        wall_ms = 1000 * (time.perf_counter() - t0)
        self.safety.advance(window_ms)
        if self.backend.biology == "living" and wall_ms > self.safety.watchdog_ms:
            self.safety.faults += 1
            self.robot.hold()
            self.emit({"type": "safety", "event": "watchdog", "wall_ms": round(wall_ms, 1), "action": "hold"})
            return None
        self.safety.faults = 0
        if self.rec is not None:
            self.rec.add(sp, pattern, window_ms)
        return sp

    def episode(self, ep: int, label: int | None = None) -> dict:
        b, T = self.b, self.b.timing
        trial, resp, teach_ms = int(T["trial_ms"]), int(T["response_window_ms"]), int(T["teach_ms"])
        self.robot.reset(ep)
        errs, step, held = [], 0, 0
        while True:
            obs = self.robot.observe()
            lv = self.encoder.levels(obs)
            self.backend.set_plasticity(self.teach)
            sp = self._read(self.encoder.pattern(lv), trial)
            if sp is None:
                if self.safety.faults >= self.safety.max_trips:
                    self.emit({"type": "safety", "event": "estop", "reason": "consecutive faults"})
                    raise TransferError("e-stop: consecutive backend faults")
                continue
            counts = _counts(sp, 0, resp)
            a_raw = self.decoder(counts)
            if self.robot_limits:
                a, why = self.safety.limit_action(a_raw, obs)
            else:
                a, why = np.clip(a_raw, -1, 1), None
            if why:
                held += 1
                self.robot.hold()
                self.emit({"type": "safety", "event": why, "action": "hold", "episode": (ep if label is None else label), "step": step})
            done = self.robot.apply(a)
            step += 1
            o2 = self.robot.observe()
            errs.append(o2["err_m"])
            if self.rec is not None:
                self.rec.steps.append(dict(episode=(ep if label is None else label), step=step, t_ms=self.rec.t_ms - trial,
                                           levels={int(k): int(v) for k, v in lv.items()},
                                           counts=[int(counts.get(e, 0)) for e in self.decoder.inputs],
                                           action=np.round(a, 5).tolist(), err_mm=1000 * o2["err_m"],
                                           tip=np.round(o2["tip"], 5).tolist(),
                                           target=np.round(o2["target"], 5).tolist()))
            if self.teach:
                dq = self.robot.teacher_action()
                self._read(teach_pattern(b.training, self.roles, lv, dq, self.teach_amp), teach_ms)
            now = time.time()
            if self.frame_dt is not None and now - self._last_frame >= self.frame_dt:
                self._last_frame = now
                grid = [0] * 64
                for e, c in counts.items():
                    g = self.grid_index.get(e)
                    if g is not None:
                        grid[g] = int(c)
                self.emit({"type": "frame", "episode": (ep if label is None else label), "step": step, "bodies": self.robot.body_poses(),
                           "target": np.round(o2["target"], 5).tolist(), "tip": np.round(o2["tip"], 5).tolist(),
                           "err_mm": round(1000 * o2["err_m"], 1), "spikes": grid,
                           "action": np.round(a, 3).tolist()})
            if done:
                break
        return dict(success=bool(self.robot.success), final_err_mm=round(1000 * errs[-1], 2),
                    median_err_mm=round(1000 * float(np.median(errs)), 2), steps=step, held_steps=held)


# =============================================================================================
# Building a backend from a bundle
# =============================================================================================
def load_source_culture(bundle: Bundle, mock_culture: str, seed: int):
    """'trained' -> the exported source culture (replay / export round-trip test);
    'naive'   -> a fresh culture with a different seed and NO trained sensory->motor projection
                 (rehearses the real situation: a living culture knows nothing about the robot)."""
    from neurorover.culture.core import Culture
    if mock_culture == "naive":
        c = Culture(seed=seed, rule="frozen")
        c.W[c.plastic] = 0.0          # no special pathway: generic local connectivity only
        c.mask[c.plastic] = False
        c.plastic[:] = False
        # give the paired-stimulation rule something to shape: weak random sensory->motor synapses
        rm = RoleMap(dict(bundle.roles), bundle.n_joints)
        sens = np.isin(c.ch, [bundle.sim_channel[e] for e in rm.sensory()]) & c.exc
        mot = np.isin(c.ch, [bundle.sim_channel[e] for e in rm.motor()])
        m = sens[:, None] & mot[None, :] & (c.rng.random((c.N, c.N)) < 0.15)
        c.mask |= m
        c.plastic[m] = True
        c.W[m] = c.rng.uniform(0.5, 1.5, m.sum())
        c.run(1000)
        return c, "naive simulated culture (seed %d)" % seed
    src = bundle.manifest["provenance"]["culture_pickle"]
    p = ROOT / src["path"]
    if not p.exists():
        raise TransferError(f"trained source culture not found at {src['path']} (use --mock-culture naive)")
    import pickle
    with open(p, "rb") as fh:
        obj = pickle.load(fh)
    c = obj.culture if hasattr(obj, "culture") else obj
    return c, f"trained source culture {src['path']}"
