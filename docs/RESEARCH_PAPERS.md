# Surgical robot-learning papers: what we borrowed and what we did not build

We used three NVIDIA/Toronto/Berkeley surgical-autonomy papers as design references for `surgical/`. This page maps each paper's ideas to code in this repo. It also says plainly where we are nowhere close. Our surgical task is a **state-based MuJoCo reach** on arm 1 of a SimReady surgical cart (`surgical/reach_env.py`). The controller is a **simulated** Izhikevich culture (`surgical/culture_ctrl.py`, `neurorover/culture/core.py`). No living tissue is involved, and nothing here has clinical validity.

## 1. ORBIT-Surgical (arXiv:2404.16027)

**Paper:** An open, physics-based surgical simulation framework built on NVIDIA Isaac Sim / Omniverse, with photorealistic ray-traced rendering and GPU-parallel physics. It ships 14 benchmark tasks for the dVRK and STAR robots: reach, reach with obstacles, needle lift, peg transfer, needle handover, threaded needle through ring, gauze pick, shunt insertion, and others. Rigid and deformable objects are included. It supports RL, imitation learning, teleoperation and synthetic-data generation, and shows sim-to-real transfer onto a physical dVRK.

**What this repo implements, by analogy:**
- The simplest ORBIT-Surgical task, **Reach**: drive the tool tip to a target position. In our version, arm 1 of a SimReady surgical cart (converted from USD to MJCF, `surgical/usd_to_mjcf.py`) reaches a target 1 cm from the tip and must hold it for 0.5 s (`surgical/reach_env.py`).
- **Randomised initial and goal states.** Episodes randomise link masses, joint damping and target noise, and `surgical/scenarios.py` adds seed-addressable target regions (`novel_targets`).
- **Parallel training.** ORBIT-Surgical parallelises on the GPU. We parallelise across Freestyle VMs instead: `sweep/run_jobs.sh`, or `python -m bnn_platform fleet sweep`. Each VM runs one CPU MuJoCo job, such as one culture learning rate.

**What it does not implement:** Isaac Sim/Omniverse, photorealistic or endoscopic rendering, a dVRK or STAR model, remote-centre-of-motion constraints, deformable objects, needles, gauze, suturing, any of the other 13 tasks, teleoperation, and sim-to-real on a physical robot.

## 2. SuFIA: language-guided augmented dexterity (arXiv:2405.05226)

**Paper:** An LLM turns a surgeon's natural-language command into a high-level plan plus low-level Python control code. A perception module grounds objects such as needles. The system re-plans when it fails and hands control back to the surgeon (human-in-the-loop) when information is insufficient. It needs no learning. The authors evaluated it on four simulated sub-tasks and two physical dVRK sub-tasks under randomised object/gripper starts and estimation/control noise.

**What this repo implements, by analogy:**
- **The robustness axes from SuFIA's evaluation**, turned into procedural stress families in `surgical/scenarios.py`: `calibration_bias` (constant 10 mm error in the perceived target), `measurement_noise` (3 mm Gaussian noise on perception), `actuator_mismatch` (action gain x U(0.8, 1.2)), and `combined_stress`. `surgical/research_eval.py` wraps the env in a `ScenarioView`, so the culture and the DLS-IK baseline see the same corrupted error signal and neither sees ground truth.
- A matched comparison of a learned controller (culture) against an analytic one (DLS-IK) on identical scenario seeds.

**What it does not implement:** any LLM, language interface, task planning, code generation, re-planning, perception or object grounding, or human-in-the-loop handover. Our controller is a closed-loop spiking reach policy, not a planner.

## 3. SuFIA-BC: demonstration data for visuomotor behaviour cloning (arXiv:2504.14857)

**Paper:** Extends ORBIT-Surgical into a surgical digital twin with photorealistic human anatomical organs and textures. The authors use it to generate synthetic teleoperation demonstrations and train visual behaviour-cloning policies (multi-view RGB and point clouds from an endoscope view). They benchmark these policies on tissue retraction, needle lift, needle handover, suture pad and block transfer. They report that state-of-the-art BC struggles on these contact-rich tasks and that generalisation to unseen instances and viewpoints is hard.

**What this repo implements, by analogy:**
- **Learning from an expert signal rather than a reward.** The culture's `teach` rule is closest in spirit to behaviour cloning: after each step, the Jacobian-transpose direction `J^T e` (an analytic "expert") selects which motor pools get paired stimulation (`surgical/culture_ctrl.py`). Training curves come from `surgical/train_culture.py`.
- **Synthetic data from the simulator.** `surgical/world_model.py` collects IK-generated transitions and fits a small supervised one-step forward model. `research_eval.py` reports how well that model predicts the tip position (mm RMSE, plus an open-loop rollout). This is a dynamics model, not a BC policy, and the module docstring says so.
- **Generalisation to unseen instances** becomes held-out targets (50 fixed targets with seed 999 in `train_culture.py`) plus the `novel_targets` scenario family.

**What it does not implement:** photorealistic anatomy, any camera or visual observations, point clouds, tissue or contact-rich manipulation, needle handover, suturing, human teleoperation demonstrations, and diffusion/transformer BC policies. Our observations are a 20-d state vector.

## Summary

| Idea | Paper | In this repo |
|---|---|---|
| Reach to a goal position | ORBIT-Surgical | Yes: `surgical/reach_env.py` (MuJoCo, SimReady cart arm 1) |
| Randomised starts/goals, domain randomisation | ORBIT-Surgical, SuFIA | Yes: env DR plus `surgical/scenarios.py` |
| Perception bias/noise, actuator mismatch stress tests | SuFIA | Yes: `scenarios.py` families, `research_eval.py` ScenarioView |
| Parallel training at scale | ORBIT-Surgical (GPU) | Different approach: one job per Freestyle VM (`sweep/run_jobs.sh`, `bnn_platform fleet`) |
| Learning from an expert signal | SuFIA-BC (BC) | Analogue: paired-stimulation `teach` driven by a J^T e expert |
| Synthetic simulator data for a learned model | SuFIA-BC | Partial: IK-data forward model (`world_model.py`), not a policy |
| Photorealistic Isaac Sim / Omniverse rendering | ORBIT-Surgical, SuFIA-BC | No |
| Tissue, deformables, needles, suturing | all three | No |
| LLM planning, language, human-in-the-loop | SuFIA | No |
| Visual (camera / point cloud) policies | SuFIA-BC | No |
| dVRK/STAR hardware, sim-to-real | all three | No |

The new contribution here is not surgical fidelity. It is the controller: a simulated neuron culture trained only through electrode stimulation, a protocol that could in principle run on a real multi-electrode array. It is evaluated with the same kind of stress tests these papers use for their policies.
