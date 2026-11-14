"""Load mjcf/scene_test.xml, simulate 2 s, check NaNs/resting box, render PNG (offscreen)."""
import os, sys, numpy as np, mujoco
HERE = os.path.dirname(os.path.abspath(__file__))
m = mujoco.MjModel.from_xml_path(os.path.join(HERE, "mjcf", "scene_test.xml")); d = mujoco.MjData(m)
print(f"nbody={m.nbody} ngeom={m.ngeom} nmesh={m.nmesh} nq={m.nq}")
b = m.body("drop_box").id
for i in range(int(2.0 / m.opt.timestep)): mujoco.mj_step(m, d)
print("NaN:", bool(np.isnan(d.qpos).any() or np.isnan(d.qvel).any()), "time", round(d.time, 3))
print("box z", round(d.xpos[b][2], 4), "box speed", round(float(np.linalg.norm(d.cvel[b])), 5))
cr = m.body("sr_crate_0").id
print("crate_0 z", round(d.xpos[cr][2], 4), "contacts box-crate:",
      sum(1 for c in d.contact[:d.ncon] if {m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]} == {b, cr}))
moved = [(m.body(i).name, round(float(np.linalg.norm(d.cvel[i])), 3)) for i in range(1, m.nbody) if np.linalg.norm(d.cvel[i]) > 0.05]
print("bodies still moving >0.05:", moved)
for i in range(1, m.nbody):
    if m.body_jntnum[i]: print(f"  {m.body(i).name:24s} mass={m.body_mass[i]:8.3f} z={d.xpos[i][2]:.3f}")
if len(sys.argv) > 1:
    r = mujoco.Renderer(m, 720, 1280); imgs = []
    for cam in ("overview", "crates", "yard"):
        r.update_scene(d, camera=cam); imgs.append(r.render())
    from PIL import Image
    Image.fromarray(np.vstack([imgs[0], np.hstack([imgs[1][::2, ::2], imgs[2][::2, ::2]])])).save(sys.argv[1]); print("saved", sys.argv[1])
