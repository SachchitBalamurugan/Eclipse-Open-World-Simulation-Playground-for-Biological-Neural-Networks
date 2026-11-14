# Simulated culture reaching with arm 1 of the SimReady surgical cart

## Conversion
- The official route failed: the pip `mujoco` 4.0.1 Windows wheel has no OpenUSD decoder (`MjSpec.from_file(*.usd)`
  gives "could not find decoder"), and `pip install "mujoco[usd]"` timed out on download.
- Fallback that worked: `usd_to_mjcf.py` (pxr only). It builds 47 bodies from RigidBodyAPI prims and 46 joints
  (40 hinge, 6 slide; USD limits converted from degrees to radians) from body0/body1 with localPos/localRot. The
  anchors of both joint sides agree to 1e-7 m and every authored joint value is 0. Masses, inertia and COM come from
  the per-mesh UsdPhysics MassAPI. The meshes are triangulated OBJs; collision uses MuJoCo convex hulls, with 46
  touching-at-rest and grandparent pairs excluded. `base_01` is welded to the world.
- USD drives are force-type with stiffness 0, so the position servos use stated gains: 400 N·m/rad and 4000 N/m,
  damping 25 / 300, and armature 0.05. Every link has gravity compensation (`gravcomp=1`).
- Validation: holding a zero command for 2 s under gravity gives no NaNs, max |dq| below 1e-7 and 0.00 mm of tip
  drift (`validate_model.py`).

## Task
- Arm 1 chain: base_05 -> handBase_01 (setup yaw, held fixed) -> slider_01 (slide) -> junction_01 -> elbowBig_01 ->
  elbowSmall_01 -> elbowLong_01 -> elbowMiddle_01 -> elbowShort_01 -> mechanism_01. All 8 joints from slider to
  mechanism are controlled.
- Instrument tip: the centroid of the mechanism_01 mesh vertices within 1 cm of its lowest point. This is the
  distal end of the instrument holder, which points down at the patient.
- Targets: forward kinematics of random collision-free configurations within ±45% of the joint range around a home
  "port" pose. Each target lies 5-25 cm from the home tip, below it and at least 2 cm above a virtual patient table
  at z = 0.70 m, so every target is reachable by construction. The workspace sampled over the full joint limits
  spans x 0.20-1.78, y -0.47-1.76, z 0.48-2.06 m.
- Evaluation uses 50 fixed held-out targets (seed 999). Training targets come from a separate pool.
- Env (`reach_env.py`): 20 Hz control, 4 s episodes, actions are joint-position deltas (0.03 rad / 6 mm per step)
  passed through a low-pass filter. Reward is -10·dist, minus action, joint-limit and collision penalties, plus 10
  on success. Success means the tip stays within 1 cm for 0.5 s.
- Domain randomization: link mass ×U(0.8, 1.2), joint damping ×U(0.7, 1.3), target noise N(0, 3 mm).

## Data
`resources/challenges/*` has no surgical or robot-motion data; the only "surgery" mentions are trial titles in
clinical-trial-evidence-cards. JIGSAWS needs a registration form, so it is not openly downloadable. Targets come
only from the arm's own defined workspace.

## Results (50 held-out targets, same seeds and DR draws for every controller)
| controller | success (1 cm, 0.5 s hold) | final error median / mean (mm) | reached 1 cm | median time to 1 cm (s) | joint-limit steps | collision contacts (episodes) |
|---|---|---|---|---|---|---|
| ik_dls (engineering reference) | 94% | 2.6 / 4.4 | 94% | 0.65 | 125 | 440 (5) |
| culture_teach (2400 teaching steps) | 0% | 223.5 / 237.8 | 2% | 1.25 | 8905 | 272 (7) |
| culture_frozen | 0% | 446.8 / 440.5 | 0% | - | 241 | 0 (0) |
| random | 0% | 274.9 / 272.7 | 0% | - | 135 | 649 (20) |
| ppo | not evaluated: training was still running at the time box (see below) | | | | | |

Collisions are all arm 1's instrument holder hull touching arm 2's (mechanism_01 vs mechanism_02).

### Culture learning curve (`results/learning_curve.png`, `curve_teach.json`)
Median final error on 6 held-out targets as teaching steps increase:

| teaching steps | 0 | 800 | 1600 | 2400 |
|---|---|---|---|---|
| median error (mm) | 406 | 186 | 74 | 219 |

- The frozen culture stays at 406 mm on the same 6 targets and 447 mm on all 50. It drives away from the target.
- Teaching cut the error by about 5.5× at its best (74 mm), so paired-stimulation STDP does imprint a usable
  error-to-joint-direction map.
- Between 1600 and 2400 steps the error got worse again. lr = 0.08 drives the plastic weights toward saturation:
  the mean plastic weight rose from about 1.0 to 2.0 in an earlier lr = 0.02 run. The pools then fire in bulk, and
  the arm pins against joint limits (8905 limit-steps).
- On the full 50 targets the final taught culture is 2× better than frozen (224 vs 447 mm median) and no better
  than random (275 mm, which goes nowhere). It does not succeed on any target.
- Honest conclusion: the teaching signal is real but unstable. The culture is far behind IK, as expected.
- Untried fixes: lower lr with weight normalization, early stopping at the best checkpoint, and 2-electrode pools
  to cut readout noise.

## Artifacts
- `results/eval.json`: per-controller metrics plus the held-out targets.
- `results/trajectory.json`: 3 culture episodes with per-step q, qref, tip, target, distance, action, sensory
  levels and the 64-channel spike counts.
- `results/culture_episode0_spikes.csv` and `results/culture_episode0_stims.csv`: per-spike and per-stimulus
  logs in the style of record_run.py.
- `results/culture_teach_reach.mp4` (and `.gif`): offscreen `mujoco.Renderer` (default GL works on Windows) at
  20 fps, 240 frames. The red sphere is the tip, the green sphere is the target.
- PPO: `ppo_train.py` was still training when the time box ran out. The host was shared with another agent's
  6-process PPO, so it reached only about 86k steps in about 440 s of learner time. When `results/ppo_reach.zip`
  appears, rerun `evaluate.py` and then `make_report.py` to add the PPO row.

## Not realistic / known gaps
- No tissue or table contact physics (the table is visual only). No instrument shaft, wrist or jaw, and no
  remote-centre-of-motion port constraint. The cart joints are servoed at rest.
- The servo gains and gravity compensation are assumed values, not taken from the USD. The convex hulls
  over-approximate the link shapes.
- The culture is the neurorover Izhikevich model (1024 neurons), not tissue. Each control step uses a 60 ms trial
  plus an 85 ms teaching replay, and the teacher uses the true Jacobian, which is privileged information.
- Results are single-seed. The learning curve uses only 6 targets per point.
