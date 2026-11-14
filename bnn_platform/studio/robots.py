"""Robots for the Studio: builtins, user imports (MJCF / URDF / zip), and the generic kinematic
model wrapper (RobotModel) that the BNN reach task and the scene exporter run on.

Any MuJoCo-loadable robot becomes trainable by the simulated culture as long as it has at least one
hinge/slide joint between the world and its end-effector (EE):
  * EE: --ee (site or body name), else the deepest body of the longest kinematic chain
    (if that body carries a site, the site is used).
  * controlled joints: the hinge/slide joints on the EE chain, proximal -> distal, capped at
    MAX_JOINTS (the most proximal ones are kept: they move the EE the most).
  * targets: FK of random joint configs around the home pose (reachable by construction); the
    axis-aligned bounding box of 400 such targets is the "target_box".
"""
from __future__ import annotations

import datetime as _dt
import json
import pathlib
import re
import shutil
import tempfile
import xml.etree.ElementTree as ET
import zipfile

import numpy as np

from .common import NAME_RE, ROOT, USER_ROBOTS, StudioError, log, r6, rel

MAX_JOINTS = 12
EE_SITE = "bnn_ee"
MESH_EXT = {".stl", ".obj", ".msh"}

BUILTINS = {
    "surgical_reach": dict(
        name="surgical_reach", kind="builtin", title="SimReady surgical cart (arm 1)",
        source="surgical/model/cart.xml", tasks=["reach"], trainable=True, ee="tip"),
    "rover": dict(
        name="rover", kind="builtin", title="Frost-guard rover (trained via its own pipeline)",
        source="sim/assets/rover.xml", tasks=[], trainable=False, ee=None),
}


def _mj():
    import mujoco
    return mujoco


# --------------------------------------------------------------------------------------------
# Generic model wrapper
# --------------------------------------------------------------------------------------------
class RobotModel:
    """Compiled MuJoCo model + EE site + controlled joints + kinematic helpers."""

    def __init__(self, name, m, ee_site, joint_names, home=None, base_body=None, min_target_z=None,
                 ee_label=None, target_mocap=None):
        mujoco = _mj()
        self.mujoco = mujoco
        self.name = name
        self.m = m
        self.d = mujoco.MjData(m)
        self.site = m.site(ee_site).id
        self.ee_label = ee_label or ee_site
        self.jid = np.array([m.joint(n).id for n in joint_names], int)
        self.joint_names = list(joint_names)
        self.qadr = m.jnt_qposadr[self.jid]
        self.dadr = m.jnt_dofadr[self.jid]
        self.is_slide = m.jnt_type[self.jid] == mujoco.mjtJoint.mjJNT_SLIDE
        lo, hi = [], []
        for j in self.jid:
            if m.jnt_limited[j]:
                lo.append(m.jnt_range[j, 0]); hi.append(m.jnt_range[j, 1])
            elif m.jnt_type[j] == mujoco.mjtJoint.mjJNT_SLIDE:
                lo.append(-0.5); hi.append(0.5)
            else:
                lo.append(-np.pi); hi.append(np.pi)
        self.lo, self.hi = np.array(lo, float), np.array(hi, float)
        self.qbase = (m.key_qpos[0] if m.nkey else m.qpos0).copy()
        h = self.qbase[self.qadr] if home is None else np.asarray(home, float)
        mg = 0.02 * (self.hi - self.lo)
        self.home = np.clip(h, self.lo + mg, self.hi - mg)
        if base_body is None:
            # error is encoded in the frame of the first controlled link (rotates with joint 0, like
            # the surgical arm's handBase frame): a base-yaw joint then always sees the same mapping
            base_body = int(m.jnt_bodyid[self.jid[0]])
        elif isinstance(base_body, str):
            base_body = m.body(base_body).id
        self.base_body = int(base_body)
        self.target_mocap = target_mocap
        if min_target_z is None:
            min_target_z = -np.inf
            planes = [g for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE
                      and m.geom_bodyid[g] == 0]
            if planes:
                min_target_z = max(m.geom_pos[g][2] for g in planes) + 0.02
        self.min_target_z = float(min_target_z)
        # workspace scale from unconstrained samples around home
        self.tip0 = self.fk(self.home)
        rng = np.random.default_rng(12345)
        pts = np.array([self.fk(q) for q in self._sample_q(rng, 300)])
        self.extent = float(max(np.ptp(pts, 0).max(), 0.02))
        self.success_r = float(np.clip(0.04 * self.extent, 0.01, 0.05))
        self.min_dist = max(2 * self.success_r, 0.1 * self.extent)
        # encoding thresholds (mm) scaled from the surgical arm's 2/8/25/60 mm @ ~25 cm reach
        k = float(np.clip(self.extent / 0.25, 0.5, 8.0))
        self.thresh_mm = [2 * k, 8 * k, 25 * k, 60 * k]
        # per-step joint delta at |action| = 1 (surgical: 0.03 rad / 6 mm)
        self.scale = np.where(self.is_slide, np.clip(0.02 * (self.hi - self.lo), 0.002, 0.02), 0.04)
        self._box = None

    # -------- kinematics --------
    def set_q(self, q, d=None, target=None):
        mujoco = self.mujoco
        d = d or self.d
        d.qpos[:] = self.qbase
        d.qpos[self.qadr] = q
        if target is not None and self.target_mocap is not None:
            d.mocap_pos[self.target_mocap] = target
        mujoco.mj_kinematics(self.m, d)
        return d

    def fk(self, q):
        return self.set_q(q).site_xpos[self.site].copy()

    def jac(self, d):
        mujoco = self.mujoco
        mujoco.mj_comPos(self.m, d)
        jp = np.zeros((3, self.m.nv))
        mujoco.mj_jacSite(self.m, d, jp, None, self.site)
        return jp[:, self.dadr]

    def _sample_q(self, rng, n):
        span = 0.35 * (self.hi - self.lo)
        span = np.where(self.is_slide, span, np.minimum(span, 0.9))
        return np.clip(self.home + rng.uniform(-span, span, (n, len(self.jid))), self.lo, self.hi)

    def sample_targets(self, n, rng, max_tries=50_000):
        out, tries = [], 0
        while len(out) < n and tries < max_tries:
            tries += 1
            p = self.fk(self._sample_q(rng, 1)[0])
            if p[2] < self.min_target_z or np.linalg.norm(p - self.tip0) < self.min_dist:
                continue
            out.append(p)
        if len(out) < n:
            raise StudioError(f"could not sample reachable targets for {self.name} "
                              f"(EE barely moves or is below the floor)")
        return np.array(out)

    def target_box(self):
        if self._box is None:
            pts = self.sample_targets(400, np.random.default_rng(7))
            self._box = {"min": r6(pts.min(0), 4), "max": r6(pts.max(0), 4)}
        return self._box

    def joints_info(self):
        mujoco = self.mujoco
        m = self.m
        out = []
        for j in range(m.njnt):
            t = int(m.jnt_type[j])
            if t not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                continue
            rng = r6(m.jnt_range[j], 4) if m.jnt_limited[j] else None
            out.append({"name": m.joint(j).name, "type": "hinge" if t == mujoco.mjtJoint.mjJNT_HINGE else "slide",
                        "range": rng, "controlled": bool(j in set(self.jid.tolist()))})
        return out


# --------------------------------------------------------------------------------------------
# Building models
# --------------------------------------------------------------------------------------------
def _add_actuators(spec) -> int:
    """Position actuators on every hinge/slide joint (only if the model has no actuators)."""
    mujoco = _mj()
    if len(spec.actuators):
        return 0
    n = 0
    for j in spec.joints:
        if j.type not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE) or not j.name:
            continue
        a = spec.add_actuator(name=f"bnn_pos_{j.name}", target=j.name, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        a.set_to_position(kp=100.0 if j.type == mujoco.mjtJoint.mjJNT_HINGE else 500.0)
        n += 1
    return n


def _name_unnamed_joints(spec):
    mujoco = _mj()
    for i, j in enumerate(spec.joints):
        if not j.name and j.type in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            j.name = f"joint_{i}"


def _chain_joints(m, body):
    mujoco = _mj()
    js = []
    b = body
    while b > 0:
        adr, num = m.body_jntadr[b], m.body_jntnum[b]
        for j in range(adr + num - 1, adr - 1, -1) if num > 0 else []:
            if m.jnt_type[j] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                js.append(j)
        b = m.body_parentid[b]
    return js[::-1]  # proximal -> distal


def _auto_ee(m):
    """Deepest body of the longest kinematic chain (most hinge/slide joints to the world, then
    deepest in the tree). Returns (body_id, site_id or None)."""
    best, key = None, None
    for b in range(1, m.nbody):
        nj = len(_chain_joints(m, b))
        depth, x = 0, b
        while x > 0:
            depth += 1
            x = m.body_parentid[x]
        k = (nj, depth, m.body_mocapid[b] < 0)
        if nj > 0 and (key is None or k > key):
            best, key = b, k
    if best is None:
        raise StudioError("robot has no hinge/slide joints: nothing for the culture to control")
    sites = [s for s in range(m.nsite) if m.site_bodyid[s] == best]
    return best, (sites[0] if sites else None)


def resolve_ee(m, ee=None):
    """-> (ee_label, body_name, local_pos)"""
    if ee:
        try:
            s = m.site(ee)
            return ee, m.body(int(m.site_bodyid[s.id])).name, [float(v) for v in m.site_pos[s.id]]
        except KeyError:
            pass
        try:
            b = m.body(ee)
            return ee, b.name, [0.0, 0.0, 0.0]
        except KeyError:
            raise StudioError(f"--ee {ee!r} is neither a site nor a body of the model")
    b, s = _auto_ee(m)
    if s is not None:
        return m.site(s).name or m.body(b).name, m.body(b).name, [float(v) for v in m.site_pos[s]]
    return m.body(b).name, m.body(b).name, [0.0, 0.0, 0.0]


def compile_user_model(src: pathlib.Path, ee_body=None, ee_pos=None):
    """MjSpec from MJCF/URDF, + position actuators if none, + the EE site. -> (model, n_added)."""
    mujoco = _mj()
    try:
        spec = mujoco.MjSpec.from_file(str(src))
    except Exception as e:  # noqa: BLE001
        raise StudioError(f"MuJoCo could not load {src.name}: {e}")
    _name_unnamed_joints(spec)
    added = _add_actuators(spec)
    if ee_body is not None:
        b = spec.body(ee_body)
        if b is None:
            raise StudioError(f"EE body {ee_body!r} not found")
        b.add_site(name=EE_SITE, pos=list(ee_pos or [0, 0, 0]), size=[0.01, 0.01, 0.01], rgba=[1, 0.2, 0.2, 1],
                   group=4)
    try:
        m = spec.compile()
    except Exception as e:  # noqa: BLE001
        raise StudioError(f"MuJoCo could not compile {src.name}: {e}")
    return m, added


def _robot_from_manifest(man: dict) -> RobotModel:
    src = USER_ROBOTS / man["name"] / man["source"]
    m, _ = compile_user_model(src, man["ee_body"], man["ee_pos"])
    return RobotModel(man["name"], m, EE_SITE, man["controlled"], home=man.get("home"),
                      ee_label=man.get("ee"))


def _surgical_model() -> RobotModel:
    import sys
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from surgical.arm import CONTROLLED, HOME_Q, TABLE_Z, load_model
    m = load_model()
    return RobotModel("surgical_reach", m, "tip", CONTROLLED, home=HOME_Q, base_body="handBase_01",
                      min_target_z=TABLE_Z + 0.02, ee_label="tip",
                      target_mocap=int(m.body("target").mocapid[0]))


def load_robot(name: str) -> RobotModel:
    if name == "surgical_reach":
        return _surgical_model()
    if name in BUILTINS:
        raise StudioError(f"{name!r} is not trainable in Studio (trained via its own pipeline)")
    man = read_manifest(name)
    if not man.get("trainable", True):
        raise StudioError(f"robot {name!r} is not trainable")
    return _robot_from_manifest(man)


def load_scene_model(name: str):
    """-> (MjModel, posed MjData, ee label, target_box or None)"""
    mujoco = _mj()
    if name == "rover":
        p = ROOT / BUILTINS["rover"]["source"]
        m = mujoco.MjModel.from_xml_path(str(p))
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        return m, d, None, None
    rm = load_robot(name)
    d = rm.set_q(rm.home, target=rm.tip0)
    return rm.m, d, rm.ee_label, rm.target_box()


# --------------------------------------------------------------------------------------------
# Manifests / list
# --------------------------------------------------------------------------------------------
def read_manifest(name: str) -> dict:
    if not NAME_RE.match(name or ""):
        raise StudioError(f"invalid robot name {name!r} (use [a-z0-9_], 1-40 chars)")
    p = USER_ROBOTS / name / "robot.json"
    if not p.exists():
        raise StudioError(f"unknown robot {name!r} (no {rel(p)})")
    return json.loads(p.read_text(encoding="utf-8"))


LIST_KEYS = ("name", "kind", "title", "source", "n_bodies", "n_joints", "n_actuated", "tasks", "trainable", "ee")


def builtin_entry(name: str) -> dict:
    e = dict(BUILTINS[name])
    try:
        if name == "surgical_reach":
            rm = _surgical_model()
            e.update(n_bodies=int(rm.m.nbody), n_joints=int(rm.m.njnt), n_actuated=len(rm.jid))
        else:
            mujoco = _mj()
            m = mujoco.MjModel.from_xml_path(str(ROOT / e["source"]))
            e.update(n_bodies=int(m.nbody), n_joints=int(m.njnt), n_actuated=int(m.nu))
    except Exception as ex:  # noqa: BLE001
        log(f"warning: builtin {name}: {ex}")
        e.update(n_bodies=None, n_joints=None, n_actuated=None)
    return e


def list_robots() -> list:
    out = [builtin_entry(n) for n in BUILTINS]
    if USER_ROBOTS.exists():
        for p in sorted(USER_ROBOTS.glob("*/robot.json")):
            try:
                man = json.loads(p.read_text(encoding="utf-8"))
                out.append({k: man.get(k) for k in LIST_KEYS})
            except Exception as ex:  # noqa: BLE001
                log(f"warning: skipping {rel(p)}: {ex}")
    return out


# --------------------------------------------------------------------------------------------
# Import
# --------------------------------------------------------------------------------------------
def _model_title(m, fallback):
    raw = bytes(m.names)
    t = raw[: raw.find(bytes(1))].decode("utf-8", "ignore")
    return t or fallback


def _safe_name(s: str) -> str:
    s = re.sub(r"[^a-z0-9_]+", "_", s.lower()).strip("_")[:40]
    return s or "robot"


def _copy_mjcf(src: pathlib.Path, dest: pathlib.Path, warnings: list) -> pathlib.Path:
    """Copy an MJCF file and the asset files it references (best effort, relative paths kept)."""
    seen = set()

    def walk(xml: pathlib.Path):
        if xml in seen:
            return
        seen.add(xml)
        try:
            root = ET.parse(xml).getroot()
        except ET.ParseError as e:
            raise StudioError(f"{xml.name} is not valid XML: {e}")
        comp = {}
        for c in root.iter("compiler"):
            comp.update(c.attrib)
        assetdir = comp.get("assetdir", "")
        dirs = {"mesh": comp.get("meshdir", assetdir), "skin": comp.get("meshdir", assetdir),
                "hfield": comp.get("meshdir", assetdir), "texture": comp.get("texturedir", assetdir)}
        for el in root.iter():
            f = el.get("file")
            if not f:
                continue
            sub = pathlib.Path(dirs.get(el.tag, "")) / f
            s = (src.parent / sub)
            if not s.exists():
                warnings.append(f"referenced file not found: {sub.as_posix()}")
                continue
            if ".." in sub.parts or sub.is_absolute():
                warnings.append(f"asset outside the model folder not copied: {sub.as_posix()}")
                continue
            t = dest / sub
            t.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(s, t)
            if el.tag == "include":
                walk(s)

    walk(src)
    out = dest / src.name
    shutil.copy2(src, out)
    return out


def _find_mesh(fn: str, base: pathlib.Path, search_roots: list) -> pathlib.Path | None:
    if fn.startswith("file://"):
        fn = fn[7:]
    rest = fn
    if fn.startswith("package://"):
        rest = fn[len("package://"):]
        parts = pathlib.PurePosixPath(rest).parts
        cands = [base / pathlib.Path(*parts[1:])] + [r / rest for r in search_roots] + \
                [r / pathlib.Path(*parts[1:]) for r in search_roots]
    else:
        cands = [base / fn, pathlib.Path(fn)]
    for c in cands:
        if c.is_file():
            return c
    name = pathlib.PurePosixPath(rest).name
    for r in search_roots:
        try:
            for hit in r.rglob(name):
                if hit.is_file():
                    return hit
        except OSError:
            continue
    return None


def _copy_urdf(src: pathlib.Path, dest: pathlib.Path, warnings: list) -> pathlib.Path:
    """Copy a URDF, resolve package:// mesh paths, drop meshes MuJoCo can't read (.dae ...),
    add <mujoco><compiler .../></mujoco> so visuals are kept and fixed links are not fused."""
    try:
        tree = ET.parse(src)
    except ET.ParseError as e:
        raise StudioError(f"{src.name} is not valid XML: {e}")
    root = tree.getroot()
    if root.tag != "robot":
        raise StudioError(f"{src.name}: URDF root element must be <robot>")
    roots = [src.parent, src.parent.parent, src.parent.parent.parent]
    (dest / "meshes").mkdir(parents=True, exist_ok=True)
    used = {}
    parent_of = {c: p for p in root.iter() for c in p}
    for mesh in list(root.iter("mesh")):
        fn = mesh.get("filename", "")
        hit = _find_mesh(fn, src.parent, roots)
        geo = parent_of.get(mesh)
        owner = parent_of.get(geo) if geo is not None else None
        if hit is None or hit.suffix.lower() not in MESH_EXT:
            why = "not found" if hit is None else f"unsupported format {hit.suffix}"
            warnings.append(f"dropped mesh {fn} ({why})")
            if owner is not None and owner.tag in ("visual", "collision"):
                parent_of[owner].remove(owner)
            continue
        key = str(hit.resolve())
        if key not in used:
            nm = hit.name
            i = 1
            while nm in used.values():
                nm = f"{hit.stem}_{i}{hit.suffix}"
                i += 1
            used[key] = nm
            shutil.copy2(hit, dest / "meshes" / nm)
        mesh.set("filename", f"meshes/{used[key]}")
    # MuJoCo needs mass on moving links
    for link in root.iter("link"):
        if link.find("inertial") is None:
            inertial = ET.SubElement(link, "inertial")
            ET.SubElement(inertial, "mass", value="0.1")
            ET.SubElement(inertial, "inertia", ixx="1e-4", iyy="1e-4", izz="1e-4", ixy="0", ixz="0", iyz="0")
    for old in root.findall("mujoco"):
        root.remove(old)
    mj = ET.Element("mujoco")
    ET.SubElement(mj, "compiler", discardvisual="false", fusestatic="false", strippath="false",
                  balanceinertia="true")
    root.insert(0, mj)
    out = dest / src.name
    tree.write(out, encoding="utf-8", xml_declaration=True)
    return out


def _pick_model_file(folder: pathlib.Path) -> pathlib.Path:
    cands = []
    for p in folder.rglob("*"):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext == ".xml":
            try:
                head = p.read_text(encoding="utf-8", errors="ignore")[:4000]
            except OSError:
                continue
            if "<mujoco" not in head:
                continue
            rank = 0 if p.stem.lower().startswith("scene") else 1
        elif ext == ".urdf":
            rank = 2
        else:
            continue
        cands.append((len(p.relative_to(folder).parts), rank, p.name, p))
    if not cands:
        raise StudioError("zip contains no MJCF (.xml with <mujoco>) or .urdf file")
    cands.sort()
    return cands[0][3]


def import_robot(file: str, name: str | None = None, ee: str | None = None) -> dict:
    src = pathlib.Path(file).expanduser()
    if not src.is_absolute():
        src = (pathlib.Path.cwd() / src)
    src = src.resolve()
    if not src.is_file():
        raise StudioError(f"file not found: {file}")
    ext = src.suffix.lower()
    if ext not in (".xml", ".urdf", ".zip"):
        raise StudioError(f"unsupported file type {ext!r} (use .xml MJCF, .urdf or .zip)")
    name = name or _safe_name(src.stem)
    if not NAME_RE.match(name):
        raise StudioError(f"invalid robot name {name!r} (use [a-z0-9_], 1-40 chars)")
    if name in BUILTINS:
        raise StudioError(f"{name!r} is a builtin robot name")
    USER_ROBOTS.mkdir(exist_ok=True)
    warnings: list = []
    tmp = pathlib.Path(tempfile.mkdtemp(prefix=f".import_{name}_", dir=USER_ROBOTS))
    try:
        if ext == ".zip":
            ex = tmp / "_unzipped"
            with zipfile.ZipFile(src) as z:
                for info in z.infolist():
                    p = pathlib.PurePosixPath(info.filename)
                    if p.is_absolute() or ".." in p.parts:
                        raise StudioError(f"unsafe path in zip: {info.filename}")
                z.extractall(ex)
            model_src = _pick_model_file(ex)
            if model_src.suffix.lower() == ".urdf":
                main = _copy_urdf(model_src, tmp, warnings)
                shutil.rmtree(ex)
            else:
                # keep the whole folder layout (includes, meshdir) and point at the model file
                for item in ex.iterdir():
                    shutil.move(str(item), str(tmp / item.name))
                ex.rmdir()
                main = tmp / model_src.relative_to(ex)
        elif ext == ".urdf":
            main = _copy_urdf(src, tmp, warnings)
        else:
            main = _copy_mjcf(src, tmp, warnings)

        m0, _ = compile_user_model(main)
        ee_label, ee_body, ee_pos = resolve_ee(m0, ee)
        m, added = compile_user_model(main, ee_body, ee_pos)
        chain = _chain_joints(m, m.body(ee_body).id)
        if not chain:
            raise StudioError(f"EE {ee_label!r} has no hinge/slide joints between it and the world")
        if len(chain) > MAX_JOINTS:
            warnings.append(f"{len(chain)} joints on the EE chain; controlling the {MAX_JOINTS} most proximal")
            chain = chain[:MAX_JOINTS]
        controlled = [m.joint(j).name for j in chain]
        rm = RobotModel(name, m, EE_SITE, controlled, ee_label=ee_label)
        box = rm.target_box()
        man = dict(
            name=name, kind="user", title=_model_title(m, name),
            source=main.relative_to(tmp).as_posix(), format={".xml": "mjcf", ".urdf": "urdf"}.get(
                main.suffix.lower(), "mjcf"), original=src.name,
            n_bodies=int(m.nbody), n_joints=int(m.njnt), n_actuated=len(controlled),
            tasks=["reach"], trainable=True, ee=ee_label, ee_body=ee_body, ee_pos=r6(ee_pos),
            controlled=controlled, home=r6(rm.home), added_actuators=added, n_actuators=int(m.nu),
            joints=rm.joints_info(), target_box=box, reach_extent_m=round(rm.extent, 4),
            success_radius_mm=round(1000 * rm.success_r, 1), warnings=warnings,
            created=_dt.datetime.now().isoformat(timespec="seconds"), biology="simulated")
        (tmp / "robot.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
        dest = USER_ROBOTS / name
        if dest.exists():
            shutil.rmtree(dest)
        tmp.rename(dest)
        return man
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
