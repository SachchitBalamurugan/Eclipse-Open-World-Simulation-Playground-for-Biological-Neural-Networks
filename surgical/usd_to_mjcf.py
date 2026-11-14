"""Convert the NVIDIA SimReady surgical cart USD into an MJCF model (pxr only).

Why this exists: the pip `mujoco` 4.0.1 Windows wheel ships no OpenUSD decoder
(`MjSpec.from_file(*.usd)` -> "could not find decoder for resource"), and
`pip install "mujoco[usd]"` timed out / only adds the Python USD *exporter*
deps. So we walk the UsdPhysics articulation ourselves.

Route:
  bodies  = prims with UsdPhysics.RigidBodyAPI (47)
  tree    = joint body0 -> body1 (joints reference the mesh child; mapped to its body)
  root    = base_01 (never a body1) -> welded to world
  joints  = revolute -> hinge, prismatic -> slide; axis = joint-frame X/Y/Z
            expressed in the child body via localRot1; position = localPos1;
            limits deg -> rad; the authored pose's joint value is computed from
            the body0/body1 frames and written as `ref` so q=ref is the USD pose.
  inertia = UsdPhysics MassAPI on the body prim or its mesh (mass, centerOfMass,
            diagonalInertia, principalAxes). Bodies without one fall back to a
            stated 0.2 kg box estimate (reported on stdout).
  geoms   = each body's meshes, triangulated, transformed mesh->body, written
            as OBJ. MuJoCo collides meshes via their convex hulls. Body pairs
            already in contact in the authored pose are excluded (standard
            practice for CAD-derived robots whose hulls interpenetrate).
  actuators = position actuators on all limited joints. The USD drives are
            force-type with stiffness 0 (damping 0.1), i.e. no usable gain, so
            kp is a stated value: 400 N*m/rad (hinge), 4000 N/m (slide), joint
            damping 25 N*m*s/rad / 300 N*s/m, armature 0.05 (servo-like, near critical).

Usage: .venv/Scripts/python -I surgical/usd_to_mjcf.py   (writes surgical/model/)
"""
from __future__ import annotations

import math
import pathlib
import sys

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
USD = ROOT / "assets/surgical/robotic_system_a01/simready_usd/sm_robotic_system_a01_01.usd"
OUT = HERE / "model"
KP_HINGE, KP_SLIDE = 400.0, 4000.0
DAMP_HINGE, DAMP_SLIDE = 25.0, 300.0


def mat(m: Gf.Matrix4d) -> np.ndarray:
    """pxr row-vector matrix -> column-vector 4x4 numpy (p_world = T @ p)."""
    return np.array(m, dtype=float).T


def quat_wxyz(q) -> np.ndarray:
    return np.array([q.GetReal(), *q.GetImaginary()], float)


def q2R(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def R2q(R):
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(R)))
        if i == 0:
            s = math.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
        elif i == 1:
            s = math.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
        else:
            s = math.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    return q / np.linalg.norm(q)


def f(v):
    return " ".join(f"{x:.7g}" for x in v)


def short(name: str) -> str:
    return name.replace("_obj_00", "")


def main():
    stage = Usd.Stage.Open(str(USD))
    xc = UsdGeom.XformCache()
    bodies = {}
    for p in stage.Traverse():
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            bodies[str(p.GetPath())] = p
    T = {k: mat(xc.GetLocalToWorldTransform(p)) for k, p in bodies.items()}

    def owner(path) -> str:
        s = str(path)
        while s and s not in bodies:
            s = s.rsplit("/", 1)[0]
        return s

    joints = []
    for p in stage.Traverse():
        if not p.IsA(UsdPhysics.Joint):
            continue
        t = p.GetTypeName()
        if t not in ("PhysicsRevoluteJoint", "PhysicsPrismaticJoint"):
            print("skip joint type", t, p.GetPath())
            continue
        J = UsdPhysics.RevoluteJoint(p) if t == "PhysicsRevoluteJoint" else UsdPhysics.PrismaticJoint(p)
        b0 = owner(J.GetBody0Rel().GetTargets()[0])
        b1 = owner(J.GetBody1Rel().GetTargets()[0])
        joints.append(dict(
            name=p.GetName(), kind="hinge" if t == "PhysicsRevoluteJoint" else "slide",
            b0=b0, b1=b1, axis=J.GetAxisAttr().Get(),
            lo=J.GetLowerLimitAttr().Get(), hi=J.GetUpperLimitAttr().Get(),
            p0=np.array(J.GetLocalPos0Attr().Get(), float), r0=quat_wxyz(J.GetLocalRot0Attr().Get()),
            p1=np.array(J.GetLocalPos1Attr().Get(), float), r1=quat_wxyz(J.GetLocalRot1Attr().Get()),
        ))
    parent = {j["b1"]: j for j in joints}
    roots = [b for b in bodies if b not in parent]
    print("roots:", [short(pathlib.PurePosixPath(r).name) for r in roots])
    children = {b: [] for b in bodies}
    for j in joints:
        children[j["b0"]].append(j["b1"])

    # ---- meshes -> OBJ (body frame) ----
    (OUT / "meshes").mkdir(parents=True, exist_ok=True)
    meshfiles, inertials, estimated = {}, {}, []
    for k, bp in bodies.items():
        nm = short(bp.GetName())
        Tb_inv = np.linalg.inv(T[k])
        files = []
        mass_src = bp if bp.HasAPI(UsdPhysics.MassAPI) and UsdPhysics.MassAPI(bp).GetMassAttr().Get() else None
        for i, mp in enumerate(Usd.PrimRange(bp, Usd.TraverseInstanceProxies())):
            if not mp.IsA(UsdGeom.Mesh):
                continue
            m = UsdGeom.Mesh(mp)
            pts = np.array(m.GetPointsAttr().Get(), float)
            cnt = np.array(m.GetFaceVertexCountsAttr().Get(), int)
            idx = np.array(m.GetFaceVertexIndicesAttr().Get(), int)
            Tm = Tb_inv @ mat(xc.GetLocalToWorldTransform(mp))
            P = (Tm[:3, :3] @ pts.T).T + Tm[:3, 3]
            tris, o = [], 0
            for c in cnt:
                for a in range(1, c - 1):
                    tris.append((idx[o], idx[o + a], idx[o + a + 1]))
                o += c
            fn = f"{nm}_{i}.obj"
            with open(OUT / "meshes" / fn, "w") as fh:
                fh.write("\n".join(f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in P))
                fh.write("\n")
                fh.write("\n".join(f"f {a+1} {b+1} {c+1}" for a, b, c in tris))
                fh.write("\n")
            files.append((fn, Tm))
            if mass_src is None and mp.HasAPI(UsdPhysics.MassAPI) and UsdPhysics.MassAPI(mp).GetMassAttr().Get():
                mass_src = mp
        meshfiles[k] = files
        if mass_src is not None:
            M = UsdPhysics.MassAPI(mass_src)
            Tm = Tb_inv @ mat(xc.GetLocalToWorldTransform(mass_src))
            com = Tm[:3, :3] @ np.array(M.GetCenterOfMassAttr().Get(), float) + Tm[:3, 3]
            Rpa = Tm[:3, :3] @ q2R(quat_wxyz(M.GetPrincipalAxesAttr().Get()))
            di = np.maximum(np.array(M.GetDiagonalInertiaAttr().Get(), float), 1e-6)
            mass = float(M.GetMassAttr().Get())
            inertials[k] = (mass, com, R2q(Rpa), di)
        else:
            estimated.append(nm)
            inertials[k] = (0.2, np.zeros(3), np.array([1, 0, 0, 0.]), np.full(3, 1e-3))
    if estimated:
        print("mass estimated (0.2 kg) for:", estimated)

    # ---- joint ref angle from authored frames ----
    ax_idx = {"X": 0, "Y": 1, "Z": 2}
    for j in joints:
        e = np.eye(3)[ax_idx[j["axis"]]]
        R0 = T[j["b0"]][:3, :3] @ q2R(j["r0"])
        R1 = T[j["b1"]][:3, :3] @ q2R(j["r1"])
        P0 = T[j["b0"]][:3, :3] @ j["p0"] + T[j["b0"]][:3, 3]
        P1 = T[j["b1"]][:3, :3] @ j["p1"] + T[j["b1"]][:3, 3]
        if j["kind"] == "hinge":
            Rrel = R0.T @ R1  # rotation of frame1 wrt frame0, about e
            # angle about e
            a = 0.5 * np.array([Rrel[2, 1] - Rrel[1, 2], Rrel[0, 2] - Rrel[2, 0], Rrel[1, 0] - Rrel[0, 1]])
            ang = math.atan2(float(a @ e), (np.trace(Rrel) - 1) / 2)
            j["ref"] = ang
            j["anchor_err"] = float(np.linalg.norm(P1 - P0))
        else:
            j["ref"] = float((R0.T @ (P1 - P0)) @ e)
            j["anchor_err"] = float(np.linalg.norm((P1 - P0) - (R0 @ e) * j["ref"]))
        j["axis_body1"] = q2R(j["r1"]) @ e
        j["pos_body1"] = j["p1"]

    # ---- emit MJCF ----
    L = ['<mujoco model="surgical_cart_a01">',
         '  <compiler angle="radian" meshdir="meshes" autolimits="true"/>',
         '  <option timestep="0.002" integrator="implicitfast" gravity="0 0 -9.81"/>',
         '  <default>',
         '    <geom type="mesh" rgba="0.85 0.87 0.9 1" contype="1" conaffinity="1" condim="3" density="0"/>',
         '    <joint armature="0.05"/>',
         '  </default>',
         '  <visual><global offwidth="960" offheight="720"/><quality shadowsize="2048"/></visual>',
         '  <asset>']
    for k in bodies:
        for fn, _ in meshfiles[k]:
            L.append(f'    <mesh name="{fn[:-4]}" file="{fn}"/>')
    L += ['    <texture name="grid" type="2d" builtin="checker" rgb1=".25 .27 .3" rgb2=".2 .22 .25" width="300" height="300"/>',
          '    <material name="grid" texture="grid" texrepeat="8 8" reflectance=".1"/>',
          '  </asset>', '  <worldbody>',
          '    <light pos="0 0 4" dir="0 0 -1" directional="true"/>',
          '    <light pos="2 -2 3" dir="-1 1 -1"/>',
          '    <geom name="floor" type="plane" size="5 5 0.1" material="grid" contype="0" conaffinity="0"/>']

    jmap = {j["b1"]: j for j in joints}

    def emit(k, parentT, ind):
        nm = short(bodies[k].GetName())
        rel = np.linalg.inv(parentT) @ T[k]
        L.append(f'{ind}<body name="{nm}" pos="{f(rel[:3,3])}" quat="{f(R2q(rel[:3,:3]))}">')
        mass, com, iq, di = inertials[k]
        L.append(f'{ind}  <inertial mass="{mass:.6g}" pos="{f(com)}" quat="{f(iq)}" diaginertia="{f(di)}"/>')
        if k in jmap:
            j = jmap[k]
            lim = ""
            if np.isfinite(j["lo"]) and np.isfinite(j["hi"]):
                lo, hi = (math.radians(j["lo"]), math.radians(j["hi"])) if j["kind"] == "hinge" else (j["lo"], j["hi"])
                lim = f' range="{lo:.6g} {hi:.6g}"'
            damp = DAMP_HINGE if j["kind"] == "hinge" else DAMP_SLIDE
            if not lim:
                damp = 0.05
            L.append(f'{ind}  <joint name="{j["name"]}" type="{j["kind"]}" pos="{f(j["pos_body1"])}" '
                     f'axis="{f(j["axis_body1"])}" ref="{j["ref"]:.6g}" damping="{damp}"{lim}/>')
        for fn, _ in meshfiles[k]:
            L.append(f'{ind}  <geom name="g_{fn[:-4]}" mesh="{fn[:-4]}"/>')
        if nm == "mechanism_01":
            L.append(f'{ind}  <site name="tip_site_placeholder" pos="0 0 0" size="0.01"/>')
        for c in children[k]:
            emit(c, T[k], ind + "  ")
        L.append(f"{ind}</body>")

    for r in roots:
        emit(r, np.eye(4), "    ")
    L.append("  </worldbody>")
    L.append("  <actuator>")
    for j in joints:
        if np.isfinite(j["lo"]) and np.isfinite(j["hi"]):
            kp = KP_HINGE if j["kind"] == "hinge" else KP_SLIDE
            lo, hi = (math.radians(j["lo"]), math.radians(j["hi"])) if j["kind"] == "hinge" else (j["lo"], j["hi"])
            L.append(f'    <position name="a_{j["name"]}" joint="{j["name"]}" kp="{kp}" ctrlrange="{lo:.6g} {hi:.6g}"/>')
    L.append("  </actuator>")
    L.append("</mujoco>")
    (OUT / "cart_raw.xml").write_text("\n".join(L))
    print("joints:", len(joints), "max anchor mismatch (m):", max(j["anchor_err"] for j in joints))
    for j in joints:
        if abs(j["ref"]) > 1e-3:
            print(f"  ref {j['name']}: {j['ref']:.4f}")
    finalize()


def finalize():
    """Compile, find body pairs touching in the authored pose, exclude them, save cart.xml."""
    import mujoco
    raw = (OUT / "cart_raw.xml").read_text()
    m = mujoco.MjModel.from_xml_path(str(OUT / "cart_raw.xml"))
    d = mujoco.MjData(m)
    d.qpos[:] = m.qpos0
    mujoco.mj_forward(m, d)
    pairs = set()
    for i in range(d.ncon):
        c = d.contact[i]
        b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
        if b1 != b2 and b1 > 0 and b2 > 0:
            pairs.add(tuple(sorted((mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b1),
                                    mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b2)))))
    # also exclude grandparent pairs within the same chain (hulls of CAD links overlap)
    for b in range(1, m.nbody):
        p = m.body_parentid[b]
        gp = m.body_parentid[p]
        if gp > 0:
            pairs.add(tuple(sorted((mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b),
                                    mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, gp)))))
    ex = "  <contact>\n" + "\n".join(f'    <exclude body1="{a}" body2="{b}"/>' for a, b in sorted(pairs)) + "\n  </contact>\n"
    xml = raw.replace("  <actuator>", ex + "  <actuator>")
    (OUT / "cart.xml").write_text(xml)
    print("excluded initial-contact/grandparent pairs:", len(pairs))


if __name__ == "__main__":
    main()
