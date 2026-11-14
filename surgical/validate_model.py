"""Load surgical/model/cart.xml, hold zero command for 2 s under gravity, print arm-1 tree."""
import pathlib
import sys

import mujoco
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from surgical.arm import ARM1_JOINTS, load_model  # noqa: E402

m = load_model()
d = mujoco.MjData(m)
print(f"nbody={m.nbody} njnt={m.njnt} nu={m.nu} ngeom={m.ngeom} total mass={m.body_subtreemass[0]:.2f} kg")
mujoco.mj_forward(m, d)
q0 = d.qpos.copy()
tip0 = d.site("tip").xpos.copy()
for _ in range(1000):
    mujoco.mj_step(m, d)
dq = np.abs(d.qpos - q0)
print("after 2 s: NaN:", bool(np.isnan(d.qpos).any()), " max |dq| =", f"{dq.max():.4g}",
      " tip drift (mm) =", f"{1000*np.linalg.norm(d.site('tip').xpos - tip0):.2f}",
      " ncon =", d.ncon, " max |qvel| =", f"{np.abs(d.qvel).max():.3g}")
print("worst joint:", mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, int(np.argmax(dq[:m.njnt]))) if m.nq == m.njnt else "")
print("\nArm 1 kinematic tree:")
for jn in ARM1_JOINTS:
    j = m.joint(jn)
    b = m.body(j.bodyid[0])
    par = m.body(m.body_parentid[j.bodyid[0]]).name
    lo, hi = m.jnt_range[j.id]
    unit = "deg" if m.jnt_type[j.id] == mujoco.mjtJoint.mjJNT_HINGE else "m"
    sc = 180 / np.pi if unit == "deg" else 1
    print(f"  {par:>14s} -> {b.name:<14s} {['free','ball','slide','hinge'][m.jnt_type[j.id]]:5s} "
          f"axis={np.round(d.xaxis[j.id],3)} range=[{lo*sc:.1f},{hi*sc:.1f}] {unit} mass={m.body_mass[j.bodyid[0]]:.3f}kg")
print("tip site (world):", np.round(tip0, 4))
