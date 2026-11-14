"""Convert NVIDIA SimReady OpenUSD props into MuJoCo meshes + MJCF fragments (no mujoco USD plugin needed).

For each asset folder hf/<name>/simready_usd/sm_<name>_01.usd:
  * open the stage with pxr, select the MuJoCo physics variant (MuJoCo=Enabled, PhysX=Disabled),
  * traverse (incl. instance proxies) every UsdGeom.Mesh, triangulate (fan), bake its transform relative
    to the asset root, convert metersPerUnit -> metres and Y-up -> Z-up,
  * write meshes/<name>__<part>.obj, a GLB (glb/<name>.glb, flat colours) and mjcf/<name>.xml,
  * mass per part from UsdPhysics MassAPI (physics:mass); MuJoCo derives inertia from the convex hull
    with that mass (matches the asset's mjc:inertia="convex").
Run:  .venv/Scripts/python -I assets/simready/usd_to_mjcf.py
"""
import os, glob, json
import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Gf

HERE = os.path.dirname(os.path.abspath(__file__))
HF, MESH, MJ, GLB = (os.path.join(HERE, d) for d in ("hf", "meshes", "mjcf", "glb"))

# Flat RGBA by material keyword (SimReady colours live in PNG albedo textures, which we do not ship).
PALETTE = [("rubber", (0.06, 0.06, 0.06)), ("corn", (0.93, 0.78, 0.25)), ("organic", (0.55, 0.65, 0.25)),
           ("blue", (0.10, 0.30, 0.75)), ("white", (0.90, 0.90, 0.88)), ("wood", (0.72, 0.55, 0.35)),
           ("paper", (0.80, 0.75, 0.65)), ("painted", (0.85, 0.45, 0.10)), ("steel", (0.55, 0.57, 0.60)),
           ("aluminium", (0.75, 0.76, 0.78)), ("metal", (0.45, 0.47, 0.50))]


def color_of(prim):
    mat, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
    name = mat.GetPrim().GetName().lower() if mat else ""
    if mat:  # constant colour if the material has one
        for p in Usd.PrimRange(mat.GetPrim()):
            for inp in ("diffuseColor", "diffuse_color_constant"):
                a = p.GetAttribute("inputs:" + inp)
                if a and a.HasAuthoredValue() and not a.HasAuthoredConnections() and a.Get() is not None:
                    c = a.Get(); return (float(c[0]), float(c[1]), float(c[2])), name
    for k, c in PALETTE:
        if k in name: return c, name
    return (0.6, 0.6, 0.6), name


def mass_of(prim):
    for p in (prim, prim.GetParent()):
        if p.HasAPI(UsdPhysics.MassAPI):
            m = UsdPhysics.MassAPI(p).GetMassAttr().Get()
            if m and m > 0: return float(m)
    return None


def load_asset(name):
    usd = glob.glob(os.path.join(HF, name, "simready_usd", "sm_*.usd"))[0]
    st = Usd.Stage.Open(usd)
    root = st.GetDefaultPrim()
    st.SetEditTarget(st.GetSessionLayer())  # never touch the downloaded files
    vs = root.GetVariantSets()
    for n, sel in (("MuJoCo", "Enabled"), ("PhysX", "Disabled"), ("Newton", "Disabled")):
        if vs.HasVariantSet(n): vs.GetVariantSet(n).SetVariantSelection(sel)
    mpu = UsdGeom.GetStageMetersPerUnit(st) or 1.0
    up = UsdGeom.GetStageUpAxis(st)
    conv = np.eye(4) * mpu; conv[3, 3] = 1
    if up == "Y":  # Y-up -> Z-up: (x, y, z) -> (x, -z, y)
        conv = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]) @ conv
    xc = UsdGeom.XformCache()
    rootinv = np.array(xc.GetLocalToWorldTransform(root).GetInverse()).T
    parts = []
    for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if not p.IsA(UsdGeom.Mesh): continue
        if UsdGeom.Imageable(p).ComputePurpose() in ("guide", "proxy"): continue
        m = UsdGeom.Mesh(p)
        pts = np.array(m.GetPointsAttr().Get(), dtype=np.float64)
        cnt = np.array(m.GetFaceVertexCountsAttr().Get()); idx = np.array(m.GetFaceVertexIndicesAttr().Get())
        tris, o = [], 0
        for c in cnt:  # fan triangulation
            f = idx[o:o + c]; tris += [(f[0], f[k], f[k + 1]) for k in range(1, c - 1)]; o += c
        tris = np.array(tris, dtype=np.int64)
        if m.GetOrientationAttr().Get() == UsdGeom.Tokens.leftHanded: tris = tris[:, [0, 2, 1]]
        M = conv @ rootinv @ np.array(xc.GetLocalToWorldTransform(p)).T  # USD row-vector -> column
        v = (np.c_[pts, np.ones(len(pts))] @ M.T)[:, :3]
        if np.linalg.det(M[:3, :3]) < 0: tris = tris[:, [0, 2, 1]]
        rgb, mat = color_of(p)
        parts.append(dict(name=p.GetName(), v=v, f=tris, rgb=rgb, material=mat, mass=mass_of(p)))
    return parts, dict(usd=os.path.relpath(usd, HERE), upAxis=up, metersPerUnit=mpu)


def write_obj(path, v, f):
    with open(path, "w") as fh:
        fh.write("".join(f"v {a:.6f} {b:.6f} {c:.6f}\n" for a, b, c in v))
        fh.write("".join(f"f {a+1} {b+1} {c+1}\n" for a, b, c in f))


# Collision overrides for clearly non-convex assets: list of (centre xyz, half-size xyz) boxes, asset frame.
def fence_boxes(v):
    lo, hi = v.min(0), v.max(0)
    cx, zt = (lo[0] + hi[0]) / 2, hi[2]
    hx = (hi[0] - lo[0]) / 2
    return [((cx, 0, (0.12 + zt) / 2), (hx, 0.02, (zt - 0.12) / 2)),             # mesh panel
            ((lo[0] + 0.06, 0, 0.06), (0.06, (hi[1] - lo[1]) / 2, 0.06)),          # feet
            ((hi[0] - 0.06, 0, 0.06), (0.06, (hi[1] - lo[1]) / 2, 0.06))]
BOX_COLLIDERS = {"barrier_fence_metal_a02": fence_boxes}


def convert(name):
    parts, meta = load_asset(name)
    allv = np.vstack([p["v"] for p in parts])
    lo, hi = allv.min(0), allv.max(0)
    assets, geoms, info = [], [], []
    for i, p in enumerate(parts):
        mn = f"{name}__{i}"
        write_obj(os.path.join(MESH, mn + ".obj"), p["v"], p["f"])
        assets.append(f'    <mesh name="sr_{mn}" file="../meshes/{mn}.obj"/>')
        r, g, b = p["rgb"]
        assets.append(f'    <material name="sr_{mn}_mat" rgba="{r:.3f} {g:.3f} {b:.3f} 1" specular="0.2"/>')
        info.append(dict(part=p["name"], verts=len(p["v"]), tris=len(p["f"]), mass_kg=p["mass"], material=p["material"]))
    xml = [f"<!-- SimReady {name} (NVIDIA, CC-BY-4.0). Generated by usd_to_mjcf.py; mesh paths are relative to this file. -->",
           "<mujoco>", "  <asset>", *assets, "  </asset>", "</mujoco>"]
    with open(os.path.join(MJ, f"{name}_assets.xml"), "w") as fh: fh.write("\n".join(xml) + "\n")
    total = sum(p["mass"] or 0 for p in parts)
    return dict(name=name, **meta, bbox_min=lo.round(4).tolist(), bbox_max=hi.round(4).tolist(),
                mass_kg=round(total, 3), parts=info, box_colliders=BOX_COLLIDERS[name](allv) if name in BOX_COLLIDERS else None,
                n_parts=len(parts))


def body_xml(a, bname, pos, yaw_deg=0.0, free=True, indent="    "):
    """MJCF <body> for converted asset `a` (dict from convert()). Visual geoms group 2, collision group 3."""
    L = [f'{indent}<body name="{bname}" pos="{pos[0]:.3f} {pos[1]:.3f} {pos[2]:.3f}" euler="0 0 {yaw_deg:.1f}">']
    if free: L.append(f"{indent}  <freejoint/>")
    n = a["name"]
    for i, p in enumerate(a["parts"]):
        mass = p["mass_kg"] if p["mass_kg"] else None
        L.append(f'{indent}  <geom type="mesh" mesh="sr_{n}__{i}" material="sr_{n}__{i}_mat" class="sr_visual"/>')
        if not a["box_colliders"]:
            m = f' mass="{mass:.4f}"' if (mass and free) else ""
            L.append(f'{indent}  <geom type="mesh" mesh="sr_{n}__{i}" class="sr_collision"{m}/>')
    if a["box_colliders"]:
        per = (a["mass_kg"] or 10.0) / len(a["box_colliders"])
        for c, h in a["box_colliders"]:
            L.append(f'{indent}  <geom type="box" pos="{c[0]:.3f} {c[1]:.3f} {c[2]:.3f}" size="{h[0]:.3f} {h[1]:.3f} {h[2]:.3f}" class="sr_collision"' + (f' mass="{per:.3f}"' if free else "") + "/>")
    L.append(f"{indent}</body>")
    return "\n".join(L)


ASSETS = ["barrier_fence_metal_a02", "construction_wheelbarrow_metal_a01", "drum_steel_a01",
          "drum_plastic_blue_a01", "crate_plywood_b01", "pallet_a01", "food_produce_corn_b01"]


def build_props(conv):
    """Farm-field layout in a 30 m x 30 m field centred on the origin (fence on the boundary)."""
    A = {a["name"]: a for a in conv}
    inc = "\n".join(f'  <include file="{n}_assets.xml"/>' for n in ASSETS)
    fence = A["barrier_fence_metal_a02"]
    seg = fence["bbox_max"][0] - fence["bbox_min"][0] - 0.23  # panel pitch (feet overlap)
    x0 = fence["bbox_min"][0] + 0.115
    bodies, k, half = [], 0, 15.0
    nseg = int(np.ceil(2 * half / seg))
    for side, (yaw, start, step) in enumerate([(0, (-half, -half), (1, 0)), (90, (half, -half), (0, 1)),
                                               (180, (half, half), (-1, 0)), (270, (-half, half), (0, -1))]):
        for j in range(nseg):
            # leave a 3 m gate gap in the south side for the rover
            if side == 0 and abs(-half + (j + 0.5) * seg) < 1.5: continue
            px, py = start[0] + step[0] * (j * seg - x0), start[1] + step[1] * (j * seg - x0)
            bodies.append(body_xml(fence, f"sr_fence_{k:03d}", (px, py, 0), yaw, free=False)); k += 1
    # shed / yard area, north-east corner
    bodies.append(body_xml(A["construction_wheelbarrow_metal_a01"], "sr_wheelbarrow", (10.5, 11.0, 0.002), 30))
    for i, (x, y) in enumerate([(12.0, 12.6), (12.7, 12.0), (12.8, 12.9)]):
        bodies.append(body_xml(A["drum_steel_a01"], f"sr_drum_steel_{i}", (x, y, 0.002), 20 * i))
    for i, (x, y) in enumerate([(11.0, 13.2), (11.4, 13.4)]):
        bodies.append(body_xml(A["drum_plastic_blue_a01"], f"sr_drum_plastic_{i}", (x, y, 0.002), 0))
    bodies.append(body_xml(A["pallet_a01"], "sr_pallet_0", (9.0, 12.8, 0.002), 0))
    # crates at the west ends of the crop rows (rows run along x at y = -6..6 step 3)
    for i, y in enumerate([-6.0, 0.0, 6.0]):
        bodies.append(body_xml(A["crate_plywood_b01"], f"sr_crate_{i}", (-11.5, y, 0.002), 0))
    # harvested corn on the middle crate (lid top at ~1.02 m)
    for i in range(3):
        bodies.append(body_xml(A["food_produce_corn_b01"], f"sr_corn_{i}", (-11.6 + 0.12 * i, -0.25 + 0.2 * i, 1.08), 90))
    out = ["<!-- SimReady farm-field props (NVIDIA SimReady, CC-BY-4.0; see ../ATTRIBUTION.md).",
           "     Generated by assets/simready/usd_to_mjcf.py. Include inside your <mujoco> root:",
           '       <include file="../assets/simready/mjcf/props.xml"/>   (path relative to the including file)',
           "     Defines default classes sr_visual / sr_collision, mesh/material assets and a worldbody of props.",
           "     Field: 30 m x 30 m centred on origin, fence on |x|,|y| = 15 with a 3 m gate at y=-15, x in [-1.5, 1.5]. -->",
           "<mujoco>", "  <default>",
           '    <default class="sr_visual"><geom contype="0" conaffinity="0" group="2" density="0"/></default>',
           '    <default class="sr_collision"><geom group="3" rgba="0.8 0.2 0.2 0.3" friction="0.6 0.005 0.0001" condim="3" solref="0.005 1"/></default>',
           "  </default>", inc, "  <worldbody>", *bodies, "  </worldbody>", "</mujoco>"]
    with open(os.path.join(MJ, "props.xml"), "w") as fh: fh.write("\n".join(out) + "\n")


def build_glb(conv):
    import trimesh
    for a in conv:
        sc = trimesh.Scene()
        for i, p in enumerate(a["parts"]):
            mesh = trimesh.load(os.path.join(MESH, f"{a['name']}__{i}.obj"), process=False)
            # three.js is Y-up: rotate Z-up -> Y-up
            mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
            rgb = [int(255 * c) for c in a["rgb"][i]] + [255]
            mesh.visual = trimesh.visual.ColorVisuals(mesh, face_colors=np.tile(rgb, (len(mesh.faces), 1)))
            sc.add_geometry(mesh, node_name=p["part"])
        sc.export(os.path.join(GLB, a["name"] + ".glb"))


if __name__ == "__main__":
    for d in (MESH, MJ, GLB): os.makedirs(d, exist_ok=True)
    conv = []
    for n in ASSETS:
        parts, _ = load_asset(n)
        a = convert(n); a["rgb"] = [p["rgb"] for p in parts]; conv.append(a)
        print(n, a["n_parts"], "parts", a["mass_kg"], "kg", a["bbox_min"], a["bbox_max"], flush=True)
    build_props(conv)
    with open(os.path.join(HERE, "conversion_report.json"), "w") as fh:
        json.dump([{k: v for k, v in a.items() if k != "rgb"} for a in conv], fh, indent=1)
    try:
        build_glb(conv); print("GLB ok")
    except Exception as e:
        print("GLB skipped:", e)
