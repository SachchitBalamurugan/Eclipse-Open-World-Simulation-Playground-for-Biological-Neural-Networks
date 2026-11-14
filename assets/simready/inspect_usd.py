"""Inspect SimReady USD assets with pxr: units, meshes, UsdPhysics (rigid body / collision / mass / joints)."""
import sys, glob, os, json
from pxr import Usd, UsdGeom, UsdPhysics, Gf

def inspect(path):
    st = Usd.Stage.Open(path)
    # select MuJoCo physics variant if present
    root = st.GetDefaultPrim()
    vs = root.GetVariantSets()
    info = {"file": os.path.relpath(path), "upAxis": UsdGeom.GetStageUpAxis(st), "metersPerUnit": UsdGeom.GetStageMetersPerUnit(st),
            "variantSets": {n: vs.GetVariantSet(n).GetVariantNames() for n in vs.GetNames()},
            "selected": {n: vs.GetVariantSet(n).GetVariantSelection() for n in vs.GetNames()}}
    for n in vs.GetNames():
        if "MuJoCo" in vs.GetVariantSet(n).GetVariantNames() or n == "MuJoCo":
            pass
    meshes, bodies, cols, masses, joints, other = [], [], [], [], [], {}
    bb = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy", "guide"])
    for p in Usd.PrimRange(st.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        if p.IsA(UsdGeom.Mesh):
            m = UsdGeom.Mesh(p); pts = m.GetPointsAttr().Get()
            meshes.append((str(p.GetPath()), len(pts) if pts else 0, UsdGeom.Imageable(p).ComputePurpose()))
        if p.HasAPI(UsdPhysics.RigidBodyAPI): bodies.append(str(p.GetPath()))
        if p.HasAPI(UsdPhysics.CollisionAPI):
            approx = UsdPhysics.MeshCollisionAPI(p).GetApproximationAttr().Get() if p.HasAPI(UsdPhysics.MeshCollisionAPI) else None
            cols.append((str(p.GetPath()), p.GetTypeName(), approx))
        if p.HasAPI(UsdPhysics.MassAPI):
            ma = UsdPhysics.MassAPI(p)
            masses.append((str(p.GetPath()), ma.GetMassAttr().Get(), ma.GetDensityAttr().Get(), ma.GetCenterOfMassAttr().Get()))
        if p.IsA(UsdPhysics.Joint): joints.append((str(p.GetPath()), p.GetTypeName()))
        for a in p.GetAttributes():
            if a.GetName().startswith("mjc:") and a.HasAuthoredValue():
                other.setdefault(a.GetName(), a.Get())
        for s in p.GetAppliedSchemas():
            if s.startswith("Mjc"): other.setdefault("schema:" + s, str(p.GetPath()))
    r = bb.ComputeWorldBound(root).ComputeAlignedRange()
    info.update(bbox_min=list(r.GetMin()), bbox_max=list(r.GetMax()), n_meshes=len(meshes), n_points=sum(m[1] for m in meshes),
                meshes=meshes[:12], rigid_bodies=bodies, colliders=cols[:12], n_colliders=len(cols), mass=masses, joints=joints, mjc=other)
    return info

if __name__ == "__main__":
    out = {}
    for d in sys.argv[1:]:
        f = glob.glob(os.path.join(d, "simready_usd", "sm_*.usd"))[0]
        out[os.path.basename(d.rstrip("/"))] = inspect(f)
    print(json.dumps(out, indent=1, default=str))
