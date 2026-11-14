# Research context for the surgical reach experiments

This project uses the SimReady `robotic_system_a01` asset as a MuJoCo control model. It is not an implementation
of any of the papers below. The model has 47 bodies and 46 joints; only arm 1's eight joints are controlled. The
task is state-based tip reaching over a virtual table. There is no tissue manipulation, camera perception, surgical
workflow, or clinical control layer. The neural controller is a simulated 1,024-neuron Izhikevich culture.

## What the primary papers contribute

| Primary work | Paper contribution | Practical CPU/MuJoCo translation | What this repository does not implement |
|---|---|---|---|
| [SuFIA: Language-Guided Augmented Dexterity for Robotic Surgical Assistants (2024)](https://arxiv.org/html/2405.05226v1) | Uses a pretrained LLM to turn a surgeon's language request into a plan and robot API calls. Object perception grounds the request in RGB-D scenes; the system checks object state, replans after a mismatch, and returns control to a surgeon when it cannot resolve uncertainty. It reports four simulated subtasks and two physical dVRK subtasks. | Keep the subtask boundary explicit: this project evaluates one reachable tip target at a time, with seeded target and sensor/control disturbances. Add an operator-confirmation state before any real actuator integration. | No LLM, language interface, perception stack, surgical object, replanning based on visual feedback, teleoperation handoff, or physical robot. A good reach score is not SuFIA-style augmented dexterity. |
| [Orbit-Surgical: An Open-Simulation Framework for Learning Surgical Augmented Dexterity (2024)](https://arxiv.org/html/2404.16027v1) | Builds a surgical simulation and benchmark layer on Orbit/Isaac Sim: articulated dVRK and STAR assets, GPU-parallel physics, contact-rich rigid and deformable objects, teleoperation, RL/IL workflows, synthetic images, and sim-to-real experiments. It provides 14 surgical tasks and reports thousands of parallel environments and GPU throughput; those resources are central to its training scale. | For a Windows/CPU project, use MuJoCo for the rigid articulated arm, small state observations, a few serial or process-parallel environments, seeded targets, and explicit baselines such as damped-least-squares IK. Keep soft tissue and high-throughput training outside the claim. | Not the Isaac Sim/Orbit runtime; no GPU vectorization, FEM soft tissue, dVRK/STAR task suite, expert teleoperation, surgical objects, or sim-to-real result. The converted cart model is a rigid reach-control test, not an Orbit-Surgical port. |
| [SuFIA-BC: Generating High Quality Demonstration Data for Visuomotor Policy Learning in Surgical Subtasks (2025)](https://arxiv.org/html/2504.14857v1) | Extends surgical digital twins with anatomical scenes and evaluates visual behavior cloning from 50 human teleoperation demonstrations per task. It compares camera and point-cloud observations with ACT and 3D Diffusion Policy across five tasks, including needle lift/handover, tissue retraction, suture-pad threading, and block transfer; it also tests instance and viewpoint generalization. | A modest CPU analogue would log joint/target/action trajectories for this reach task, train a small behavior-cloning baseline, and hold out target regions or disturbances. This needs to be compared on the same seeded episodes as the culture and IK controllers. | No expert demonstration dataset, teleoperation, RGB-D/point-cloud pipeline, anatomy, ACT, Diffusion Policy, needle transfer, or visual generalization experiment. The culture's paired-stimulation teaching uses task labels and privileged Jacobian directions; it is not expert imitation. |

The papers' task success rates are not directly comparable with this project's reach error. Their robots, contacts,
observations, task definitions, and evaluation protocols differ.

## What runs in this repository

`usd_to_mjcf.py` converts the articulated USD asset and per-body meshes to MJCF/OBJ. `validate_model.py` holds the
converted pose for two seconds under gravity. `reach_env.py` defines the 20 Hz joint-delta task, target sampling,
domain randomization, reward, and the 1 cm / 0.5 s hold success rule. `controllers.py` supplies the IK reference
and episode runner. `culture_ctrl.py` encodes tip error into stimulation and reads motor pools; paired teaching
uses the true Jacobian to decide which joint-direction pool to stimulate. This Jacobian is privileged information.
`train_culture.py`, `ppo_train.py`, and `evaluate.py` train/evaluate the culture, PPO, and non-learning baselines.
`record_and_render.py` writes the per-step culture, joint, tip, and spike-count trajectory used by the app.
`world_model.py` fits a small supervised forward predictor from synthetic IK transitions; the scenario evaluator
can report its held-out prediction metrics alongside control outcomes.

The scenario stream in `scenarios.py` makes six seeded evaluation families: nominal targets, reachable targets
25–38 cm from home (beyond the original 25 cm sampler limit), a 10 mm calibration bias, 3 mm measurement noise,
0.8–1.2 actuator gain, and combined stress. These magnitudes are explicit stress assumptions, not measurements.
The procedural stream can produce indefinitely many seeded episodes, but it is a fixed sampler over specified
distributions. It does not learn transition dynamics or invent new physical laws.

## The learned model is bounded, not open-ended

None of the three papers above supplies an open-ended learned generative world model for this robot. SuFIA's LLM
generates plans and code; Orbit-Surgical is a hand-authored physics simulator; SuFIA-BC's diffusion policy
generates action sequences from observations. Those are different functions.

This repository does include `surgical/world_model.py`, a learned forward-dynamics baseline. A NumPy random-
Fourier-feature ridge ensemble maps 61 pre-step features (state, action, actuator/filter state, and domain
randomization context) to a 19-dimensional next-state delta. It was trained on 317 noisy-DLSIK transitions from
12 synthetic episodes and evaluated on 87 transitions from three disjoint episodes. Teacher-forced one-step tip
RMSE is 14.09 mm versus 18.29 mm for zero-delta persistence; on recorded-action open-loop rollouts it is 70.33
versus 81.75 mm at five steps, then 115.33 versus 99.87 mm at ten steps, where the learned model loses to
persistence. The uncertainty and nearest-training-distance values are uncalibrated diagnostics.

That is a useful short-horizon transition predictor, but it is not an open-ended generative world model: its
training data are synthetic trajectories from this fixed MuJoCo arm; its actions come from the recorded IK
teacher; its state is a compact vector rather than a scene; and it drifts over multi-step rollouts. It does not
generate new robot designs, anatomy, contacts, tasks, or independently imagined action sequences. A generated
plan, physics engine, or seeded episode sampler is not interchangeable with a learned dynamics model.

This repository's `neurorover/world/model.py` is a separate, bounded weather model: per-station seasonal means
plus a VAR(1) over daily low/high temperature anomalies, fitted on the freeze-task training split. It can sample
weather sequences, and `neurorover/envdata/generator.py` adds analog-sampled weather and assumed rover physics
effects. Those models serve the frost-guard rover; they do not model surgical robot dynamics. The surgical
scenario evaluator's seeded target/disturbance stream is also not a learned dynamics model.

## CPU operating envelope and limits

- The converted MuJoCo scene is cheap enough for local, state-based experiments, but it has none of Orbit-Surgical's GPU-parallel throughput. The recorded PPO run used four `DummyVecEnv` environments in one process and reached 235,540 steps in its 1,200-second time box.
- The surgical VM learning-rate runs used one CPU VM per rate and 60 training episodes per run. The four curves are noisy on six checkpoint targets; all four final 50-target evaluations had 0% success. They support no selected best learning rate.
- The cart's servo gains, gravity compensation, convex collision hulls, and domain-randomization ranges are assumptions. The virtual table is visual only. There is no tissue, contact-force safety, trocar remote-center constraint, instrument shaft/wrist/jaw model, or clinical validation.
- The culture is simulated, not biological. No real surgical robot was connected, and no sim-to-real transfer was attempted.

## Commands

Run from the repository root with the project `.venv` (Windows paths shown). The converted model and app meshes
are already present in this checkout.

```powershell
.venv\Scripts\python.exe surgical\validate_model.py
.venv\Scripts\python.exe surgical\train_culture.py teach 60 0.05
.venv\Scripts\python.exe surgical\evaluate.py
.venv\Scripts\python.exe surgical\record_and_render.py teach 3
.venv\Scripts\python.exe -m surgical.world_model --train-episodes 12 --eval-episodes 3 --output surgical\results\world_model_metrics.json
```

Run one reproducible scenario-evaluation shard and aggregate collected shards with:

```powershell
.venv\Scripts\python.exe -m surgical.research_eval --seed 100 --episodes 3 --out surgical\results\research_vm_0.json
.venv\Scripts\python.exe surgical\research_summary.py
```

The evaluator compares fixed culture checkpoints with the selected baselines on matched scenario IDs and records
per-episode disturbances, metrics, and optional learned-dynamics prediction errors. The summary script refuses
duplicate run seeds. These are exploratory reach robustness experiments, not surgical subtask reproductions.
