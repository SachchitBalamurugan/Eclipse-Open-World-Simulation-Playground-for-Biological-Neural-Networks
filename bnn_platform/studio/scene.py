"""`studio scene`: viewport geometry. Body poses are WORLD, geom poses LOCAL to their body,
mesh vertices in the geom frame (MuJoCo's compiled mesh frame). Meshes are decimated by vertex
clustering so the total exported vertex count stays under MAX_VERTS."""
from __future__ import annotations

import numpy as np

from .common import r6
from .robots import load_scene_model

MAX_VERTS = 60_000
GEOM_TYPES = {0: "plane", 1: "hfield", 2: "sphere", 3: "capsule", 4: "ellipsoid", 5: "cylinder", 6: "box",
              7: "mesh", 8: "sdf"}


def decimate(v: np.ndarray, f: np.ndarray, target: int):
    """Vertex-clustering decimation to <= target vertices."""
    if len(v) <= target:
        return v, f
    target = max(target, 8)
    lo = v.min(0)
    ext = np.maximum(v.max(0) - lo, 1e-9)
    cell = float(ext.max()) / max(2.0, 4 * np.sqrt(target))
    for _ in range(80):
        key = np.floor((v - lo) / cell).astype(np.int64)
        _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
        inv = inv.reshape(-1)
        if len(cnt) <= target:
            break
        cell *= 1.1
    nv = np.zeros((len(cnt), 3))
    np.add.at(nv, inv, v)
    nv /= cnt[:, None]
    nf = inv[f]
    keep = (nf[:, 0] != nf[:, 1]) & (nf[:, 1] != nf[:, 2]) & (nf[:, 0] != nf[:, 2])
    nf = nf[keep]
    if len(nf):  # drop duplicate triangles (same vertex set), keep first winding
        _, first = np.unique(np.sort(nf, 1), axis=0, return_index=True)
        nf = nf[np.sort(first)]
    return nv, nf


def scene(name: str) -> dict:
    m, d, ee, box = load_scene_model(name)
    bodies = [{"name": m.body(b).name or f"body{b}", "parent": int(m.body_parentid[b]),
               "pos": r6(d.xpos[b]), "quat": r6(d.xquat[b])} for b in range(m.nbody)]
    gsel = [g for g in range(m.ngeom) if m.geom_group[g] < 3 and m.geom_rgba[g][3] > 0
            and int(m.geom_type[g]) in GEOM_TYPES and GEOM_TYPES[int(m.geom_type[g])] not in ("hfield", "sdf")]
    mesh_geoms = [g for g in gsel if m.geom_type[g] == 7]
    total = sum(int(m.mesh_vertnum[m.geom_dataid[g]]) for g in mesh_geoms)
    ratio = min(1.0, MAX_VERTS / total) if total else 1.0
    geoms, nverts = [], 0
    for g in gsel:
        rgba = m.geom_rgba[g]
        if m.geom_matid[g] >= 0 and np.allclose(rgba, [0.5, 0.5, 0.5, 1]):
            rgba = m.mat_rgba[m.geom_matid[g]]
        item = {"name": m.geom(g).name or None, "body": int(m.geom_bodyid[g]), "type": GEOM_TYPES[int(m.geom_type[g])],
                "size": r6(m.geom_size[g]), "pos": r6(m.geom_pos[g]), "quat": r6(m.geom_quat[g]),
                "rgba": r6(rgba, 3)}
        if m.geom_type[g] == 7:
            mi = m.geom_dataid[g]
            va, vn = m.mesh_vertadr[mi], m.mesh_vertnum[mi]
            fa, fn = m.mesh_faceadr[mi], m.mesh_facenum[mi]
            v = m.mesh_vert[va:va + vn].astype(float)
            f = m.mesh_face[fa:fa + fn].astype(np.int64)
            v, f = decimate(v, f, int(vn * ratio))
            nverts += len(v)
            item["mesh"] = {"vertices": np.round(v.reshape(-1), 4).tolist(), "faces": f.reshape(-1).tolist()}
        geoms.append(item)
    return {"robot": name, "bodies": bodies, "geoms": geoms, "ee": ee, "target_box": box,
            "n_mesh_vertices": nverts, "biology": "simulated"}
