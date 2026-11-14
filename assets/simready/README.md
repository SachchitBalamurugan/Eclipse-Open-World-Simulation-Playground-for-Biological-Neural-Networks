# SimReady props for the MuJoCo farm field

Seven NVIDIA SimReady OpenUSD props, converted for MuJoCo and three.js. They are CC-BY-4.0; see `ATTRIBUTION.md`.

## Use from sim/
Add this line inside the `<mujoco>` root of `sim/rover.xml`. The path is relative to the including file.

```xml
<include file="../assets/simready/mjcf/props.xml"/>
```

- **No compiler or meshdir settings are needed.** Mesh paths in the fragments are `../meshes/*.obj`, and MuJoCo 4.0.1 resolves them relative to the included file. This was tested from a sibling directory, from an absolute include path, and with `meshdir` set in the main file.
- **What props.xml defines:**
  - default classes `sr_visual` (group 2, no contact) and `sr_collision` (group 3, translucent red)
  - mesh and material assets
  - a `<worldbody>` that merges into yours
  - body names all start with `sr_`
- **No ground plane:** props.xml has none, so keep your own.
- **Layout:** a 30 m x 30 m field centred on the origin.
  - **Fence:** 121 static panels (1 m pitch) on |x| = 15 and |y| = 15. There is a 3 m gate on the south side (y = -15, x in [-1.5, 1.5]).
  - **Yard, north-east corner:** a wheelbarrow at (10.5, 11) and a pallet at (9, 12.8), with 3 steel drums and 2 small plastic drums around (12, 12.5).
  - **Crop rows:** plywood crates at the west end of the rows, at (-11.5, -6/0/6), with 3 corn cobs on the middle crate.
  - **Clear area:** the centre (|x|, |y| < 9) is free for crop rows and the rover.
- **Movable props:** every prop except the fence has a `<freejoint>`, so the rover can push them. Fence panels are static.
- **Free joints and qpos:** each free prop adds 7 qpos. If `rover_env.py` indexes qpos/qvel by position, use named joints or bodies instead, or put the rover first.
- **Per-asset files:** `mjcf/<asset>_assets.xml` holds each asset's mesh and material assets. `body_xml()` in `usd_to_mjcf.py` emits single bodies.

## Files
| Path | What | Size |
|---|---|---|
| `hf/<asset>/` | SimReady folders as published (USD, physics payloads, MDL, licence, packaging json), minus PNG textures | 21 MB |
| `meshes/*.obj` | Triangulated parts (Z-up, metres, asset-local frame) | 8.4 MB |
| `mjcf/props.xml`, `mjcf/*_assets.xml`, `mjcf/scene_test.xml` | MJCF | 86 KB |
| `glb/*.glb` | One GLB per asset (Y-up, flat face colours) for the three.js app | 4.9 MB |
| `fetch_simready.py` | Resumable HF downloader with retries and backoff (`ls`, `size`, `get`; set `SIMREADY_TEXTURES=1` to include PNGs) | |
| `inspect_usd.py` -> `inspection.json` | pxr inspection: up-axis, units, meshes, UsdPhysics | |
| `usd_to_mjcf.py` -> `conversion_report.json` | Converter (meshes, MJCF, GLB) | |
| `verify_scene.py` -> `scene_test.png` | Loads scene_test.xml, steps 2 s, renders | |

To regenerate, run:

```
.venv/Scripts/python -I assets/simready/usd_to_mjcf.py
.venv/Scripts/python -I assets/simready/verify_scene.py assets/simready/scene_test.png
```

## What the USD contains (inspection.json)
- **Units and axes:** every asset is Z-up, metersPerUnit = 1, with its origin at floor level. The converter still handles Y-up and other units generically.
- **Physics variants:** the root prim has variant sets `PhysX`, `Newton` and `MuJoCo`, each Enabled or Disabled. PhysX is enabled by default. The converter selects MuJoCo=Enabled in a session layer, so the downloaded files are never modified.
- **Rigid bodies:** each part is an Xform with `PhysicsRigidBodyAPI`. Its Mesh has `PhysicsCollisionAPI`, `PhysicsMassAPI` and `MeshCollisionAPI`, with `physics:approximation = sdf` by default.
- **What the MuJoCo payload sets:** it is `payloads/MuJoCo/enabled.usda`, which references `instances_mujoco.usda`. It adds:
  - `MjcCollisionAPI` and `MjcMeshCollisionAPI`
  - `mjc:group=2`, `mjc:inertia="convex"`, `mjc:maxhullvert=-1`
  - `physics:approximation="convexHull"`
  - physics:mass, centerOfMass, diagonalInertia and principalAxes
- **Physics materials:** each has density, static/dynamic friction and restitution. For example, steel has density 8000 and friction/restitution 0.3 / 0.3 / 0.1.
- **Joints:** the wheelbarrow wheel is a `PhysicsRevoluteJoint`. The steel drum lid is joined to the drum.

| Asset | Parts | Mesh points | Size (m) | Mass (from USD) |
|---|---|---|---|---|
| barrier_fence_metal_a02 | 1 | 1,892 | 1.23 x 0.23 x 1.07 | 45.4 kg |
| construction_wheelbarrow_metal_a01 | 2 (body, wheel) | 34,010 | 0.65 x 1.72 x 0.73 | 15.0 + 9.3 kg |
| drum_steel_a01 | 2 (drum, lid) | 50,238 | 0.64 x 0.66 x 0.87 | 86.0 + 16.3 kg |
| drum_plastic_blue_a01 | 1 | 13,976 | 0.28 x 0.28 x 0.39 | 0.175 kg (as authored, which looks low) |
| crate_plywood_b01 | 1 | 12,802 | 1.00 x 1.01 x 1.02 | 94.6 kg |
| pallet_a01 | 1 | 1,064 | 1.20 x 0.80 x 0.21 | 50.7 kg |
| food_produce_corn_b01 | 1 | 12,675 | 0.04 x 0.18 x 0.04 | 0.1 kg |

## Conversion route
1. **The official route is unavailable on this machine.** That route is MJCF `<model file=... content_type="text/usd"/>`.
   - The `mujoco==4.0.1` pip wheel on Windows ships no USD decoder plugin, so loading fails with `could not find decoder for resource 'x.usda'`.
   - The wheel declares no `usd` extra.
   - Its `mujoco/usd` module only exports MuJoCo to USD; it cannot import USD.
2. **The custom route works (`usd_to_mjcf.py`).** Steps:
   - Open the stage with pxr and select the MuJoCo variant.
   - Traverse the meshes, including instance proxies, and fan-triangulate them.
   - Bake each mesh's transform relative to the asset root, apply the metersPerUnit and up-axis conversion, and write one OBJ per part.
   - Visual geoms use the mesh. Collision geoms use the same mesh, which MuJoCo convexifies into a hull. This matches the asset's own MuJoCo variant (`convexHull`).
   - The geom mass is `physics:mass` from UsdPhysics. MuJoCo derives the inertia from the hull, the same as `mjc:inertia="convex"`.
   - The fence is a non-convex open mesh panel, so its collision is 3 fitted boxes instead: the panel (4 cm thick) and two feet.

## Verification (verify_scene.py)
- **Model size:** 136 bodies, 520 geoms, 9 meshes.
- **Stability:** 2 s at a 2 ms timestep with no NaNs.
- **Drop test:** a 5 kg box dropped from z = 2 m comes to rest on `sr_crate_0` at z = 1.166. That equals the crate top (1.0165) plus the box half-height (0.15). The run shows 2 box-crate contacts and a final speed of 0.02.
- **Settling:** all props settle on the ground. The corn cobs keep rolling slowly on the crate lid because they are cylinders.
- **Rendering:** offscreen `mujoco.Renderer` works with the default GL on Windows. The output is `scene_test.png`.

## Lost or simplified in conversion
- **Materials:** the MDL/OmniPBR and UsdPreviewSurface materials are dropped, and so are all PNG textures (albedo, normal, ORM). The textures are about 99% of each folder's size: the wheelbarrow alone has 558 MB of them. Colours are flat RGBA. They come from a constant diffuseColor where one exists and are otherwise guessed from the material name.
- **Articulation:** the wheelbarrow wheel's revolute joint and the steel drum's lid joint are dropped. Each asset becomes one rigid body.
- **Physics:**
  - USD friction and restitution are not copied. All props share one MuJoCo friction (0.6) with `solref 0.005 1`.
  - centerOfMass, diagonalInertia and principalAxes are recomputed by MuJoCo from the hull, not copied.
  - Hulls fill concave shapes: the wheelbarrow tray, the crate and the gaps in the pallet.
- **Metadata:** semantic labels and SimReady metadata are lost, as are GeomSubsets, UVs and normals.
