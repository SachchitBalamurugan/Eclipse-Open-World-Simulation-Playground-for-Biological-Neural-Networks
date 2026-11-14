# Handoff: state of the project and what to finish

Written for the next model or person picking this up. Last updated 2026-11-14, around 13:15 PT. The hackathon deadline is 17:00 PT on 2026-11-14: judges review `main` at that commit, so `main` must stay runnable and README.md must stay accurate.

Read README.md first. This file only covers status, open work and rules.

## Rules this project follows (keep them)

1. Nothing here is a living neural network. Everything is a simulated culture (Izhikevich model). Every UI keeps a "Simulated culture, no living neurons" badge. Never claim biological results.
2. Report results honestly, including negative ones. Don't tune on the evaluation set (the fixed 300-row stratified validation subset in `neurorover/env/frost.py`). Hyperparameter variants are saved with a `tag` and kept out of headline tables.
3. Don't use the NVIDIA name or logo, or the word "Cosmos", in our UI. We copy the visual style only. The product name is "NeuroRover Sim".
4. Freestyle VMs cost money, about $0.27/h each; the free tier caps at 10 running. Always collect results and confirm they landed before teardown (we lost 16 seeds once by tearing down too early). Never leave VMs running. Check with `npx -y freestyle@latest vm list`.
5. Third-party assets: NVIDIA SimReady assets are CC-BY-4.0, so keep attribution. The hackathon resources repo is private: don't commit its data. It lives in `resources/`, which is gitignored.
6. README.md is still missing the team ID (`TEAM_ID_HERE`). Ask the user for it; don't guess.

## Done and verified

| Component | Where | Verified how |
|---|---|---|
| Simulated culture (1,024 Izhikevich neurons, 64-electrode MEA) | `neurorover/culture/core.py` | firing about 5 Hz, bursts about 0.4/s; stim evokes about 9x sham |
| Frost task from the resources freeze data | `neurorover/env/frost.py` | loads `resources/challenges/freeze-preparation/09-freeze/data.json.gz` |
| Benchmark, 10 conditions x 50 seeds | `neurorover/bench.py`, `neurorover/results/` | 10 seeds local, 40 on Freestyle; summary in `neurorover/analysis/summary.md` |
| Paired-stimulation teaching (`culture_teach`) | `rule="teach"` in core.py and codec.py | AUC 0.55 -> 0.73 (25 -> 200 samples, 50 seeds), +0.24 over frozen at 200 |
| CL API path (real Cortical Labs SDK, simulator) | `neurorover/culture/izh_source.py`, `neurorover/run_cl.py` | matches the direct harness; `--hardware` path never run on hardware |
| Held-out freeze submissions | `neurorover/submissions/` | pass the challenge's `check_submission.py` |
| Weather world model + imagination experiment | `neurorover/world/`, `neurorover/world/REPORT.md` | 1-step RMSE 2.27 C vs persistence 2.51; imagination +0.11 AUC at 25 real (confounded with extra trials, stated) |
| Per-run signal logs | `neurorover/record_run.py`, `runs/` | CSVs of every spike and stim |
| Web demo, lab console, fleet dashboard | `demo/index.html`, `demo/lab.html`, `fleet/` | checked in Chrome; serve demo with `python -m http.server 8765 -d demo`, fleet with `python fleet/server.py` (port 8770) |
| 3D desktop app (Electron + three.js) | `app/` | selftest has zero console errors; rover built from `hardware/BOM.md` parts; Fleet tab wall plus pop-out windows |
| Freestyle sweep tooling | `sweep/`, `fleet/server.py` | used for 2 sweeps; teardown confirmed |

## In progress when this was written (may be partial; check before trusting)

Three background agents were still writing when this was pushed. Each folder may hold partial code. Check for a REPORT.md or README.md with results before claiming anything.

1. **`sim/`: MuJoCo RL physics sim of the rover.** The plan (Open Duck Mini style):
   - MJCF rover from BOM numbers (`sim/params.py`, every number sourced)
   - Gymnasium env with domain randomization
   - stable-baselines3 PPO on CPU
   - eval vs a hand-written controller on 100 fixed episodes
   - `night_loop.py`: culture decides DEPLOY/HOLD, PPO drives
   - rendered video and `trajectory.json` for the app
   - `vm_train.sh` for Freestyle seeds

   To finish: check `sim/runs/<run>/progress.csv` shows improvement and that `eval.json` and the video exist. Then add a short README section with the eval table.
2. **`assets/simready/`: NVIDIA SimReady props for the field** (fences, wheelbarrow, drums, crates; CC-BY-4.0).
   - SimReady assets ship their own MuJoCo payload (`payloads/MuJoCo/enabled.usda`).
   - Plan: convert to `assets/simready/mjcf/props.xml` (includable by `sim/assets/rover.xml`), GLBs for the app, and `ATTRIBUTION.md`.
   - Hugging Face downloads were flaky (connection resets).

   To finish: run `assets/simready/mjcf/scene_test.xml` and confirm a box rests on a crate. Include `props.xml` in the rover scene, then credit SimReady in README.md.
3. **`surgical/`: NVIDIA SimReady `robotic_system_a01` surgical robot** (four arms, 47 bodies, 40 revolute + 6 prismatic joints, CC-BY-4.0).
   - The source zip is in the user's Downloads, extracted to `assets/surgical/` (gitignored, 624 MB).
   - Plan:
     - `surgical/usd_to_mjcf.py` converts it to MJCF in `surgical/model/`
     - arm 1 reaches its instrument tip to targets above a virtual patient table
     - the simulated culture is the controller: error encoded as stim, motor pools drive joint directions, trained with paired-stimulation teaching using the Jacobian-transpose direction
     - baselines: IK controller, PPO, random, frozen culture
     - outputs: `surgical/REPORT.md`, `results/eval.json`, a video, `trajectory.json`

   To finish: check the robot holds pose under gravity without exploding, that the eval table exists, and that the report is honest (the culture is expected to lose to IK). The resources repo has no surgical data; say so.

## What's left, in priority order

1. **Team ID** in README.md (from the user).
2. **Finish or trim the three in-progress pieces above.** If one isn't working by about 16:15 PT, remove its claim from README.md rather than ship a broken one.
3. **README updates:** add a section for each finished piece (sim, SimReady props, surgical) with its result table and limits. Keep the existing honesty sections.
4. **App replay:** load `sim/runs/*/trajectory.json` (rover) and `surgical/results/trajectory.json` (arm) in the 3D app as new scenes or tabs.
5. **Re-run** `demo/build_lab_data.py` and `demo/export_demo.py` if new runs are added. Then reload the demo pages with a cache-busting `?v=N`.
6. **DEMO_SCRIPT.md:** update the talk track for the 3D app, the surgical arm and the teaching result.
7. **Final check before 17:00 PT:**
   - fresh clone of `main`
   - follow README "Run or view the demo"
   - `python -m neurorover.analysis.summarize` reproduces summary.md
   - `cd app && npm install && npm start` opens
   - no VMs running

## Known gaps (don't hide these)

- Simulated culture only. Whether real neurons are more sample-efficient is untested.
- `culture_teach` still trails logistic regression (0.73 vs 0.89 at 200 samples). It learns, but it is slower than silicon.
- Reward-only rules (DishBrain-style feedback, R-STDP) stay near chance. R-STDP at lr=5 shows a weak trend (0.52 -> 0.59).
- The world-model gain is confounded with extra teaching trials.
- Lab-console voltage waveforms are synthesized from spike times.
- App physics runs at 2x real time for pacing.
- BOM prices and masses are partly estimates (marked in `hardware/BOM.md`).
- No real hardware, no sim-to-real transfer.

## How to run (short)

```sh
python -m venv .venv && .venv/Scripts/pip install cl-sdk==1.0.0 numpy matplotlib
# clone the hackathon resources repo into resources/ (private; needs access)
.venv/Scripts/python -m neurorover.bench --condition culture_teach --seed 0
.venv/Scripts/python -m neurorover.record_run --condition culture_teach --seed 0 --budget 100
.venv/Scripts/python -m http.server 8765 -d demo        # http://localhost:8765
.venv/Scripts/python -u fleet/server.py                  # http://localhost:8770
cd app && npm install && npm start                       # 3D desktop app
```
