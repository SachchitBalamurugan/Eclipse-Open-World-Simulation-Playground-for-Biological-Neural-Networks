# BNN Robot Studio — UI ↔ backend contract

The Electron app (`app/`) talks to the Python backend only through this CLI, run from the repo root with
`.venv\Scripts\python.exe` (Windows) or `python3`. Every command prints JSON on stdout; long-running commands
print one JSON object per line (JSON Lines). Human logs go to stderr. Exit code 0 on success.

All biology is the simulated Izhikevich culture in `neurorover/culture/` — no living neurons. Every payload that
describes a controller includes `"biology": "simulated"`.

## Storage
- Imported robots: `user_robots/<name>/` (original files + `robot.json` manifest). Name: `[a-z0-9_]{1,40}`.
- Studio runs: `user_runs/<run_id>/` with `config.json`, `events.jsonl`, `curve.json`, `culture.pkl`, `summary.json`.

## Commands

### `python -m bnn_platform studio list`
```json
{"robots":[{"name":"surgical_reach","kind":"builtin","title":"SimReady surgical cart (arm 1)","source":"surgical/model/...",
  "n_bodies":47,"n_joints":46,"n_actuated":8,"tasks":["reach"],"trainable":true,"ee":"tip_01"}]}
```
`kind` is `builtin` or `user`. Builtins: `surgical_reach`, `rover` (rover: `trainable:false` in Studio, `tasks:[]`,
it is shown but trained via its own pipeline).

### `python -m bnn_platform studio import --file PATH [--name NAME] [--ee BODY_OR_SITE]`
Accepts `.xml` (MJCF), `.urdf`, or `.zip` (folder with an MJCF/URDF + meshes). Copies into `user_robots/<name>/`,
loads with MuJoCo, picks an end-effector (deepest body of the longest kinematic chain unless `--ee`), adds
position actuators to hinge/slide joints if the model has none. Output:
```json
{"ok":true,"robot":{...same shape as list item...,"joints":[{"name":"j1","type":"hinge","range":[-3.14,3.14]}]}}
```
On failure: `{"ok":false,"error":"human readable message"}` and exit code 1.

### `python -m bnn_platform studio scene --robot NAME`
Geometry for the viewport, posed at the model's default configuration.
```json
{"robot":"arm3","bodies":[{"name":"link1","parent":0,"pos":[x,y,z],"quat":[w,x,y,z]}],
 "geoms":[{"body":1,"type":"box|sphere|capsule|cylinder|ellipsoid|plane|mesh","size":[...MuJoCo geom_size...],
           "pos":[...],"quat":[w,x,y,z],"rgba":[r,g,b,a],
           "mesh":{"vertices":[flat xyz...],"faces":[flat i,j,k...]}}],
 "ee":"tip","target_box":{"min":[x,y,z],"max":[x,y,z]}}
```
Body `pos/quat` are WORLD poses; geom `pos/quat` are LOCAL to their body. `bodies[0]` is the world body.
Mesh vertices are in the geom's local frame. Total mesh vertices are decimated/capped at ~60k.

### `python -m bnn_platform studio train --robot NAME [--task reach] [--platform local|fleet] [--episodes 20]
###   [--lr 0.02] [--neurons 1024] [--electrodes 64] [--rule teach|frozen] [--seed 0] [--vms 4] [--dry-run]`
Streams JSON Lines:
```json
{"type":"start","run_id":"arm3_teach_s0_20261114-1530","run_dir":"user_runs/...","config":{...},"biology":"simulated"}
{"type":"frame","episode":3,"step":12,"bodies":[[x,y,z,qw,qx,qy,qz],...one per body incl. world...],
 "target":[x,y,z],"tip":[x,y,z],"err_mm":41.2,"spikes":[64 ints]}
{"type":"episode","episode":3,"episodes":20,"success":false,"final_err_mm":38.0,"median_err_mm":52.1,"success_rate":0.25}
{"type":"log","message":"..."}
{"type":"done","run_dir":"user_runs/...","summary":{"success_rate":0.3,"median_final_err_mm":30.1,"episodes":20}}
{"type":"error","message":"..."}
```
`frame` is throttled to ≤10 per second. With `--platform fleet`, the backend instead launches `--vms` Freestyle
jobs (seeds `seed..seed+vms-1`) and emits
`{"type":"fleet","jobs":[{"slug":"bnn-arm3-s0","vm":"vm-...","state":"starting|running|done|failed"}]}`, then
`done`. `--dry-run` prints the plan as a `fleet` event with `"dry_run":true` and never creates VMs.

Implementation notes (backend, `bnn_platform/studio/`):
- `--lr` defaults to **0.02** (was 0.05 in the first draft: 0.05 over-potentiates the generic reach task).
  `--electrodes` must be 64 (fixed 8x8 MEA model); `--neurons` 64..4096 (rounded down to a multiple of 64).
  Extra optional flags: `--steps N` (control steps per episode, default 60 = 3 s), `--no-frames`.
- Every `train` run: `start`, `log`, an eval on 10 held-out targets with plasticity off (`log`), the training
  `episode`s, a second held-out eval (`log`), `done`. `episode` also has `"steps"`, `"teaching_steps"`;
  `median_err_mm`/`success_rate` are running values over the episodes so far. `episode` numbers are 1-based.
  `done.summary` additionally has `eval_before`/`eval_after` (`{"success_rate","median_final_err_mm"}` on the
  held-out targets: the honest before/after learning number), `teaching_steps`, `success_radius_mm`, `wall_s`.
  Success = EE within `success_radius_mm` (4 % of the workspace extent, 10-50 mm) for 5 consecutive steps.
- The reach sim is kinematic (ideal joint-position servo; joints off the EE chain stay at the home pose), so
  any imported robot behaves, including floating-base ones. `frame.bodies` has one entry per scene body.
- Fleet: `fleet` events also carry `"run_id"` (the parent run, `user_runs/<run_id>/` with config + payload)
  and each job `"seed"`, `"run_id"` (= `<parent>_s<seed>`, collected into `user_runs/<that>/`). A failed
  launch adds `"error"`. Dry run: jobs have `"vm":null,"state":"planned"` plus `"command"`, `"job_sh"` and the
  `"freestyle"` CLI lines; the event has `"payload":{"path","files","built":false}`; then
  `{"type":"done","run_dir":null,"dry_run":true,"summary":{"platform":"fleet","jobs":N,"launched":0}}`.
  Fleet state lives in `user_runs/fleet_jobs.json` (never `sweep/jobs_state.txt`).
- `culture.pkl` is a pickled `bnn_platform.studio.bnn.BNNReachController` (`.culture`, `.nj`, `.ctx_ch`,
  `.pos_pool`, `.neg_pool`, `.thresh`).

### Sim-to-real export: `python -m bnn_platform.transfer export --run RUN_DIR [--target generic|maxwell|mcs|cl1] [--out DIR]`
Packages a trained Studio run (or `surgical/results/culture_teach.pkl`) into a hardware-agnostic deployment
bundle under `user_runs/<run_id>/export/<target>/` (also zipped). Prints:
```json
{"ok":true,"bundle":"user_runs/.../export/generic","zip":"...zip","files":["manifest.json","electrode_map.csv",...],
 "target":"generic","warnings":["simulated weights are not transferable; run calibration on the living culture first"]}
```
`python -m bnn_platform.transfer targets` → `{"targets":[{"id","name","status":"tested-mock|sdk-untested|stub-needs-vendor-sdk","vendor","n_electrodes","sdk"}]}`
(cl1: backend written against the cl SDK but never executed.) Export also accepts `--target all` or a comma list,
which prints `{"ok":..,"exports":[...]}`. The legacy pkl source writes to `surgical/results/export/<target>/`.
`python -m bnn_platform.transfer validate --bundle DIR` → `{"ok","bundle","target","errors":[],"warnings":[],"n_checks","n_passed"}`.
`run` options: `--backend mock|cl1-sim|cl1|maxwell|mcs` (maxwell and mcs return an `error` event because they are stubs), `--mock-culture trained|naive`,
`--no-teach`, `--no-calibrate`. Extra event types: `{"type":"calibration","stage":"baseline|probe|remap|decoder",...}`
and `{"type":"safety","event":"workspace|watchdog|backend_fault|estop|non_finite_action","action":"hold"}`.
`frame.spikes` is 64 ints in sim-grid order. The `start` event includes `"biology"` (backend) and `"biology_of_source":"simulated"`.
Session files go to `<bundle>/sessions/<run_id>/`. See docs/SIM_TO_REAL.md.
`python -m bnn_platform.transfer run --bundle DIR --backend mock [--episodes 3]` → JSON Lines like `train`
(`start`/`frame`/`episode`/`done`), replaying the bundle in closed loop against a simulated culture as a dry run.

### `python -m bnn_platform studio fleet-status [--no-remote]`  → `{"jobs":[{"slug","vm","state","last":{...episode event...}}]}`
Polls each unfinished VM (a few seconds), and once a job is `done`/`failed` pulls its run back into
`user_runs/<run_id>/`. Each job also has `"run_id","parent_run","robot","seed","collected","summary"`
(the run's summary.json once collected) and `"error"`. `--no-remote` returns the cached state instantly.
### `python -m bnn_platform studio runs` → `{"runs":[{"run_id","robot","rule","platform","created","summary"}]}`
Newest first; also `"parent_run"` (fleet seed runs), `"episodes"`, `"lr"`, `"seed"`. `summary` is null while
a run is in progress. A fleet parent run has `platform:"fleet"` and summary `{"platform","jobs","launched"}`.
`runs` lists ALL runs, Studio and legacy: every item has `"run_id","robot","kind","created","summary","replayable","path"`
(+ `rule`, `platform`). `kind`: `studio` (user_runs/*), `legacy_culture` (runs/*, id `legacy_runs_<dir>`),
`legacy_rover` (sim/runs/*, id `legacy_sim_<dir>`), `legacy_surgical` (surgical/results/curve_<mode>.json,
id `legacy_surgical_<mode>`).

### `python -m bnn_platform studio replay --run RUN_ID`
Re-streams a run as the same JSON Lines as `train` (`start`/`frame`/`episode`/`log`/`done`, every event has
`"replay":true`), frames throttled to ~20 fps. Studio runs: their events.jsonl. Legacy: frames from the
trajectory json if one exists (surgical/results/trajectory.json for `legacy_surgical_teach`; its frame fields are
the legacy recording's own, not the Studio `bodies` layout), then `episode` events from the curve, then `done`.

### `python -m bnn_platform studio pack --run RUN_ID [--out PATH]`
Zips the run (dir, minus fleet payload tarballs) → `{"ok":true,"zip":"user_runs/_packs/<run_id>.zip","files":N,"bytes":B}`.
Failure: `{"ok":false,"error":...}`, exit 1.
