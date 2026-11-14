"""Build app/assets/surgical/ from surgical/model (converted NVIDIA SimReady robotic_system_a01, CC-BY-4.0).

Reads surgical/model/cart.xml (MJCF, Z-up metres) and the per-body OBJ meshes, writes
  app/assets/surgical/<body>.glb   one GLB per body, vertices in that body's local frame (Z-up, untouched)
  app/assets/surgical/tree.json    body tree: parent, local pos/quat (wxyz), joint name/type/axis/pos/ref/range,
                                   plus the arm-1 joint list and the instrument-tip offset in mechanism_01.
Meshes are decimated (quadric, needs fast_simplification) only if the GLB total would exceed 15 MB.
Run:  .venv/Scripts/python app/tools/build_surgical_glb.py
"""
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "surgical" / "model"
OUT = ROOT / "app" / "assets" / "surgical"
LIMIT = 15 * 1024 * 1024
ARM1 = ["joint_handBase_01_joint_00", "joint_slider_01_joint_00", "joint_junction_01_joint_00",
        "joint_elbowBig_01_joint_00", "joint_elbowSmall_01_joint_00", "joint_elbowLong_01_joint_00",
        "joint_elbowMiddle_01_joint_00", "joint_elbowShort_01_joint_00", "joint_mechanism_01_joint_00"]


def fl(s, d):
    return [float(x) for x in s.split()] if s else list(d)


def qmat(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def main():
    root = ET.parse(SRC / "cart.xml").getroot()
    meshfile = {m.get("name"): m.get("file") for m in root.iter("mesh")}
    bodies = []

    def walk(el, parent):
        for b in el.findall("body"):
            j = b.find("joint")
            g = b.find("geom")
            rec = dict(name=b.get("name"), parent=parent, pos=fl(b.get("pos"), [0, 0, 0]), quat=fl(b.get("quat"), [1, 0, 0, 0]),
                       mesh=g.get("mesh") if g is not None and g.get("mesh") else None)
            if g is not None and (g.get("pos") or g.get("quat")):
                rec["geom_pos"] = fl(g.get("pos"), [0, 0, 0]); rec["geom_quat"] = fl(g.get("quat"), [1, 0, 0, 0])
            if j is not None:
                rec["joint"] = dict(name=j.get("name"), type=j.get("type") or "hinge", axis=fl(j.get("axis"), [0, 0, 1]),
                                    pos=fl(j.get("pos"), [0, 0, 0]), ref=float(j.get("ref") or 0),
                                    range=fl(j.get("range"), []) or None)
            bodies.append(rec)
            walk(b, rec["name"])

    walk(root.find("worldbody"), None)
    OUT.mkdir(parents=True, exist_ok=True)
    meshes = {}
    for b in bodies:
        if b["mesh"]:
            meshes[b["name"]] = trimesh.load(SRC / "meshes" / Path(meshfile[b["mesh"]]).name, force="mesh", process=False)
    nface = sum(len(m.faces) for m in meshes.values())
    est = sum(len(m.vertices) * 24 + len(m.faces) * 12 for m in meshes.values())
    ratio = 1.0
    if est > LIMIT:
        ratio = LIMIT / est * 0.9
        try:
            import fast_simplification  # noqa: F401
        except ImportError:
            sys.exit("GLB would be %.1f MB > 15 MB: pip install fast_simplification" % (est / 1e6))
    total = 0
    for name, m in meshes.items():
        if ratio < 1 and len(m.faces) > 400:
            m = m.simplify_quadric_decimation(percent=1 - ratio)
        m.visual = trimesh.visual.ColorVisuals(m)
        p = OUT / (name + ".glb")
        p.write_bytes(trimesh.exchange.gltf.export_glb(trimesh.Scene(m), include_normals=True))
        total += p.stat().st_size

    # instrument tip: centroid of mechanism_01 vertices within 1 cm of its lowest world point at the authored pose
    T = {}
    by = {b["name"]: b for b in bodies}
    for b in bodies:
        M = np.eye(4); M[:3, :3] = qmat(np.array(b["quat"])); M[:3, 3] = b["pos"]
        T[b["name"]] = (T[b["parent"]] @ M) if b["parent"] else M
    v = np.asarray(meshes["mechanism_01"].vertices)
    w = v @ T["mechanism_01"][:3, :3].T + T["mechanism_01"][:3, 3]
    sel = w[:, 2] < w[:, 2].min() + 0.01
    tip = v[sel].mean(0).tolist()
    tree = dict(source="NVIDIA SimReady robotic_system_a01 (CC-BY-4.0) via surgical/usd_to_mjcf.py",
                frame="MuJoCo / USD stage: Z-up, metres; quat wxyz; GLB vertices are in body-local frame",
                arm1_joints=ARM1, tip_body="mechanism_01", tip_local=tip, decimation_ratio=ratio,
                glb_bytes=total, faces=nface, bodies=bodies)
    (OUT / "tree.json").write_text(json.dumps(tree, indent=1))
    print("bodies", len(bodies), "glb total %.2f MB" % (total / 1e6), "faces", nface, "ratio", round(ratio, 3), "tip", tip)


if __name__ == "__main__":
    main()
