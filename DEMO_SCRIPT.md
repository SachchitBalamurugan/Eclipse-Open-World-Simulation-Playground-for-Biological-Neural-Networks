# NeuroRover demo script (about 3 minutes)

## Before judges arrive

- Open the desktop app with `app\start.bat`.
- Serve the web demo with `.venv/Scripts/python -m http.server 8765 -d demo`.
- The app labels every neural controller as a simulated culture. There are no living cells or connected robots.
- If training workers are running, the Fleet tab can show them live. Ten Freestyle VMs cost about $2.70/hour; collect results and tear them down after use.

## Hook (20 seconds)

Companies like Cortical Labs rent out living neuron cultures and say they may learn with fewer examples than silicon. Testing that is expensive, and much of the work is the plumbing: encoding a task as stimulation, reading activity back, closing the loop, and comparing against fair baselines. We built that full pipeline with a simulated culture, and also tested the control loop on rover and surgical-robot simulations.

## The app (60 seconds)

1. **Simulate tab.** The dish is a model of 1,024 Izhikevich neurons on a 64-electrode array. Blue electrodes carry sensory input; amber and teal groups vote DEPLOY and HOLD; green pulses show paired-stimulation teaching.
2. **The field.** Click Robot to see the frost-guard rover assembled from the bill of materials. Its field props are licensed SimReady assets; their source and CC-BY-4.0 attribution are in `assets/simready/ATTRIBUTION.md`.
3. **Rover replay.** Click **MuJoCo replay**, then choose the `ppo_bc_s0` deployment trajectory from the dropdown. Play or scrub the recorded MuJoCo run; the metadata panel shows the policy, decision and episode result.
4. **Surgical tab.** Choose an episode and play or scrub the arm-1 reach trajectory. The tab shows the four-arm patient-side cart, target and instrument-tip path, plus the culture's electrode counts for each step. The robot meshes and their license attribution are documented in `surgical/README.md`.
5. **Fleet tab** (when workers are active). Show each VM's training status and log tail.

## Results (60 seconds)

Show `neurorover/analysis/fig_learning_curves.png` for the frost task. AUC is measured on 300 held-out nights across 50 seeds.

- **Paired-stimulation teaching learns in the simulator.** `culture_teach` rises from 0.55 AUC at 25 examples to 0.73 at 200, but logistic regression reaches 0.89 at 200. Reward-only feedback and R-STDP stay near chance.
- **The rover policy improves on the hand-written controller in simulation.** With 120 demonstration drives used to initialize PPO, success was 72% versus 59% for the controller over 100 fixed-seed randomized episodes. In the nominal model it was 80% versus 72%. This is a single training seed and not a real-robot result.
- **The surgical reach test.** On 50 held-out targets, PPO succeeded on 98% and the IK reference on 94%; the 1,024-neuron taught culture succeeded on 0% and had 223.5 mm median final error. Teaching improved median error from 446.8 mm for the frozen culture, but it remained far from the engineering references.
- **Environment data adds a small signal.** The no-leakage weather-context experiment improves validation AUC from 0.898 to 0.908. The archive is partial and the generated weather has documented biases in `neurorover/envdata/README.md`.

## Honest limitations (20 seconds)

The culture is a simulation, not wetware. The rover and surgical arm are also simulated; neither result demonstrates sim-to-real transfer. The surgical model omits tissue contact, a remote-centre-of-motion constraint and clinical safety controls. Its culture controller is unstable and does not reach the target reliably.

## Likely questions

**Does the surgical result mean it can operate?** No. It is a reach-control experiment on one arm of a simulated cart. The culture has 0% success on the held-out targets; the IK reference is the engineering baseline.

**Why is the rover policy initialized from demonstrations?** Plain PPO stayed at 0% success in this run. Behavior cloning from 120 demonstration drives gave PPO a useful starting controller; it then reached 72% success on randomized held-out episodes.

**Why no hardware claim?** We have no real robot or living-cell data. The simulator gives a reproducible control experiment, not evidence of biological performance or safe clinical use.

**What did the environment data change?** Lagged weather covariates raised frost-prediction AUC by about 0.01. Target-night covariates are excluded from that result to avoid leakage.
