"""Shared model helpers for arm 1 of the surgical cart (MuJoCo, from usd_to_mjcf.py)."""
from __future__ import annotations

import pathlib

import mujoco
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
XML = HERE / "model" / "cart.xml"

# Arm 1 chain, proximal -> distal (parent body base_05 = the cart column head)
ARM1_JOINTS = [
    "joint_handBase_01_joint_00",    # setup yaw (held fixed during the task)
    "joint_slider_01_joint_00",      # prismatic setup slide
    "joint_junction_01_joint_00",
    "joint_elbowBig_01_joint_00",
    "joint_elbowSmall_01_joint_00",
    "joint_elbowLong_01_joint_00",
    "joint_elbowMiddle_01_joint_00",
    "joint_elbowShort_01_joint_00",
    "joint_mechanism_01_joint_00",
]
CONTROLLED = ARM1_JOINTS[1:]          # 8 joints driven by every controller
ARM1_BODIES = ["handBase_01", "slider_01", "junction_01", "elbowBig_01", "elbowSmall_01",
               "elbowLong_01", "elbowMiddle_01", "elbowShort_01", "mechanism_01"]
TABLE_Z = 0.70                        # virtual patient-table top (m, world)
# Home ("port") pose for the controlled joints: authored USD pose moved off the
# elbowLong/elbowMiddle lower limits and the slider stop. Tip ~0.96 m above floor.
HOME_Q = np.array([-0.08, 0.0, 0.0, 0.0, 0.25, 0.25, -0.1, 0.0])


def tip_offset(m: mujoco.MjModel) -> np.ndarray:
    """Instrument tip = distal end of mechanism_01: the centroid of its mesh vertices
    within 1 cm of the lowest point (the instrument holder points down at the patient)."""
    bid = m.body("mechanism_01").id
    g = [i for i in range(m.ngeom) if m.geom_bodyid[i] == bid][0]
    mi = m.geom_dataid[g]
    v = m.mesh_vert[m.mesh_vertadr[mi]: m.mesh_vertadr[mi] + m.mesh_vertnum[mi]].astype(float)
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, m.geom_quat[g])
    vb = v @ R.reshape(3, 3).T + m.geom_pos[g]            # body frame
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    Rb = d.xmat[bid].reshape(3, 3)
    zw = vb @ Rb.T[:, 2] + d.xpos[bid][2]
    sel = zw < zw.min() + 0.01
    return vb[sel].mean(0)


def load_model(xml: pathlib.Path = XML) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(xml))
    m0 = spec.compile()
    off = tip_offset(m0)
    b = spec.body("mechanism_01")
    b.add_site(name="tip", pos=off, size=[0.008] * 3, rgba=[1, 0.2, 0.2, 1])
    # gravity compensation on every link (real surgical arms are counterbalanced /
    # gravity compensated; without it the stated position gains sag ~5 deg)
    for bb in spec.bodies:
        if bb.name != "world":
            bb.gravcomp = 1.0
    wb = spec.worldbody
    # virtual patient table (visual only) and a mocap-free target marker body
    wb.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.35, 0.9, 0.02],
                pos=[0.95, 0.15, TABLE_Z - 0.02], rgba=[0.3, 0.45, 0.6, 1], contype=0, conaffinity=0)
    t = wb.add_body(name="target", mocap=True, pos=[0.9, 0.3, 0.7])
    t.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.012, 0, 0], rgba=[0.1, 1, 0.3, 0.8],
               contype=0, conaffinity=0)
    return spec.compile()


class ArmIndex:
    def __init__(self, m: mujoco.MjModel):
        self.jid = np.array([m.joint(n).id for n in CONTROLLED])
        self.qadr = m.jnt_qposadr[self.jid]
        self.dadr = m.jnt_dofadr[self.jid]
        self.act = np.array([m.actuator("a_" + n).id for n in CONTROLLED])
        self.lo = m.jnt_range[self.jid, 0].copy()
        self.hi = m.jnt_range[self.jid, 1].copy()
        self.site = m.site("tip").id
        self.arm_bodies = set(m.body(b).id for b in ARM1_BODIES)
        self.is_slide = m.jnt_type[self.jid] == mujoco.mjtJoint.mjJNT_SLIDE

    def tip(self, d) -> np.ndarray:
        return d.site_xpos[self.site].copy()

    def jac(self, m, d) -> np.ndarray:
        jp = np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, d, jp, None, self.site)
        return jp[:, self.dadr]

    def collisions(self, m, d) -> int:
        """Contacts between an arm-1 body and anything that is not arm 1 (other arms, cart)."""
        n = 0
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if (b1 in self.arm_bodies) != (b2 in self.arm_bodies) and c.dist < 0:
                n += 1
        return n


def sample_workspace(m, n=4000, seed=0):
    """Forward kinematics on random arm-1 configs (uniform in joint limits)."""
    rng = np.random.default_rng(seed)
    d = mujoco.MjData(m)
    ix = ArmIndex(m)
    pts, qs = [], []
    for _ in range(n):
        q = rng.uniform(ix.lo, ix.hi)
        d.qpos[:] = m.qpos0
        d.qpos[ix.qadr] = q
        mujoco.mj_kinematics(m, d)
        pts.append(d.site_xpos[ix.site].copy())
        qs.append(q)
    return np.array(pts), np.array(qs)


def make_targets(m, n, seed, home_q=None):
    """Targets = tip positions of random joint configs drawn from a 'port' region:
    joints within +-45% of their range around the home pose, kept only if the tip is
    above the table (>= 2 cm) and within 5-25 cm of the home tip and below it (port-to-tissue reach). Every target
    is reachable by construction (it is the FK of a valid config)."""
    rng = np.random.default_rng(seed)
    d = mujoco.MjData(m)
    ix = ArmIndex(m)
    home = HOME_Q if home_q is None else home_q
    d.qpos[:] = m.qpos0
    d.qpos[ix.qadr] = home
    mujoco.mj_kinematics(m, d)
    tip0 = d.site_xpos[ix.site].copy()
    out = []
    span = 0.45 * (ix.hi - ix.lo)
    while len(out) < n:
        q = np.clip(home + rng.uniform(-span, span), ix.lo, ix.hi)
        d.qpos[:] = m.qpos0
        d.qpos[ix.qadr] = q
        mujoco.mj_forward(m, d)
        p = d.site_xpos[ix.site].copy()
        if ix.collisions(m, d):       # target config must be collision-free (hulls)
            continue
        # below the port, above the tissue/table, 5-25 cm reach
        if TABLE_Z + 0.02 <= p[2] <= tip0[2] - 0.03 and 0.05 < np.linalg.norm(p - tip0) < 0.25:
            out.append(p)
    return np.array(out)
