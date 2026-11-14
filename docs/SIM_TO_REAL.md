# Sim to real: from a simulated culture to a living one

Package: `bnn_platform/transfer/`. CLI: `python -m bnn_platform.transfer targets|export|validate|run|to-nwb`.

## The honest summary
Every controller in this repo was trained on a **simulated** Izhikevich culture
(`neurorover/culture/core.py`). Its synaptic weights **cannot** be loaded into living neurons: a dish has
its own connectivity, and nothing can write weights into it. What does transfer is the **closed-loop
protocol**:

| transfers | does not transfer |
|---|---|
| electrode roles and layout (sensory / motor pools / reward) | simulated weights and connectivity |
| observation -> pulse-count encoding, timing (40 ms stim, 50 ms response, 60 ms trial, 85 ms teach block) | the trained performance (a living culture starts untrained) |
| decoder *structure* (pool difference); sim weights as an initial guess | stimulation amplitude in physical units (the sim has none) |
| the paired-stimulation 'teach' schedule (STDP pairing, J^T e teacher) | any guarantee that the 3 ms pairing potentiates a real pathway |
| safety envelope, calibration procedure, sim reference activity | |

## Pipeline
1. **Train in simulation**: Studio (`python -m bnn_platform studio train ...` -> `user_runs/<id>/`) or legacy
   `surgical/train_culture.py` -> `surgical/results/culture_teach.pkl`.
2. **Export**: `python -m bnn_platform.transfer export --run <run dir | culture_teach.pkl> --target generic|cl1|mcs|maxwell|all`.
   Output goes to `<run>/export/<target>/` plus a zip (legacy source: `surgical/results/export/<target>/`).
   Bundle: `manifest.json`, `electrode_map.csv`, `encoder.json`, `decoder.json`, `training_protocol.json`,
   `stim_schedule.csv`, `calibration_protocol.json`, `safety.json`, `README.md`, `sim_reference.npz`
   (+ `sim_reference.nwb` only when pynwb is installed) and `sim_reference_README.md`.
3. **Validate**: `python -m bnn_platform.transfer validate --bundle <dir>`. This checks the schema, that the
   map, encoder, decoder and schedule agree, that no role sits on a ground/reference/absent electrode,
   amplitude/charge/phase-width/burst-rate/mean-rate limits, stimulation-unit count (MaxWell <= 32),
   watchdog, hold-on-missing-data, joint soft limits and workspace.
4. **Dry run**: `python -m bnn_platform.transfer run --bundle <dir> --backend mock --episodes 2`. This runs
   calibration and then the closed loop against a simulated culture and streams JSON Lines (`start`,
   `calibration`, `frame`, `safety`, `episode`, `done`, `error`). `--mock-culture naive` uses a fresh
   untrained culture, which is closer to what a real dish will be. Session files go to `<bundle>/sessions/<id>/`.
5. **Calibrate on the living culture** (`calibration_protocol.json`): a 300 s baseline; a probe sweep of
   single biphasic pulses on every electrode (30 reps, random order, 200 ms ISI, with an amplitude ladder
   capped by safety.json); role remap onto responsive electrodes; guided trials where the teacher moves
   the robot while the culture is stimulated; then a ridge refit of the decoder toward the teacher
   direction, shrunk toward the sim decoder and accepted only if it beats it on held-out trials.
6. **Closed loop with safety**: work in step-and-hold mode at about 16.7 Hz (inference only) or about 6.9 Hz
   (with teaching). Hold position when data is missing, the backend fails, the action is non-finite, or
   the EE leaves the workspace. After 3 consecutive faults, e-stop. A physical robot e-stop is still
   mandatory. Start with the simulated robot (`--robot sim`).
7. **Train and test**: follow the session plan and controls in `training_protocol.json`: a sham session
   with shuffled pairing times, a decoder-only control, and a no-culture well.
8. **Compare with the sim**: record the session (NWB recommended). Compare spontaneous rates, evoked
   responses and sensory->motor maps with `sim_reference.npz` and `decoder.json:sim_reference_io`.

## Targets
| id | status | what is adapted |
|---|---|---|
| generic | tested-mock | 8x8, id = sim channel, 200 um pitch, current-mode 1 uA / 160 us biphasic defaults |
| cl1 | sdk-untested* | cl ChannelSet ids; grounds 0,7,56,63 and reference 4 (from cl-sdk 1.0.0 analysis layout) are avoided by remapping roles; StimDesign limits (<=3 uA, <=3 nC, 20 us steps, <=200 Hz) |
| mcs | stub-needs-vendor-sdk | 60MEA 8x8 without corners, labels `<col><row>`, internal reference 15 avoided; voltage-mode defaults |
| maxwell | stub-needs-vendor-sdk | each sim electrode = 4x4 HD region (1024 routed recording electrodes), 1 stim electrode per region, <=32 stimulation units checked |

\* `CL1Backend` (runtime.py) and `cl_source.py` were written against the cl API but **have not yet
been run end to end**. The deadline cut this off, so treat cl1 as untested until
`run --backend cl1-sim` has been run.

All stimulation defaults carry the flag "must be validated against vendor/lab limits". The MCS values
(500 mV, 400 us/phase) follow Wagenaar et al. 2004. The MaxWell values (75/150 mV) follow DishBrain
(Kagan et al. 2022) as far as I know. Treat both as placeholders.

## Vendor integration steps
Implement `MEABackend` (`bnn_platform/transfer/runtime.py`). It has three methods: `stimulate(pattern)`
queues pulses relative to the next window, `read_spikes(window_ms)` advances the device and returns
`[(t_ms, electrode_id)]`, and `close()`.
- **MaxWell**: using the `maxlab` API, route `hw_record_electrodes` and connect each `hw_stim_electrode` to
  a StimulationUnit. Build pulses as `maxlab.Sequence`. Read spikes from the MaxLab Live real-time
  stream and deduplicate within each region.
- **MCS MEA2100**: use McsUsbNet through pythonnet (data stream and STG/stimulus generator), or the
  MEA2100 DSP real-time feedback. Map labels to hardware channels with your layout file.
- **CL1**: `CL1Backend(simulate=False)` (untested). It uses one-shot StimPlans on the 25 kHz clock.
- **Robot**: `UDPJSONRobotIO` implements a documented UDP JSON protocol but is untested with hardware.
  `ROS2RobotIO` is a stub that documents the topic contract (`/joint_states`, `/bnn/target`,
  `/bnn/joint_position_targets`, `/bnn/hold`).

## Known limitations
- Verified only with the legacy surgical source (`surgical/results/culture_teach.pkl`). The Studio-run
  loader is written against `bnn_platform/studio` (BNNReachController + config.json) but has not been run
  on a real Studio run yet.
- The bundle runtime reproduces the original sim controller's actions exactly (max diff 0.0 on a held-out
  episode) when robot soft limits are off. With the safety layer on (2 % joint margins), actions near
  joint limits change, which is intended.
- Remapping roles for cl1/mcs breaks the trained sim culture. In the mcs sim reference the final error
  rises from ~181 mm to ~522 mm, which shows directly that learned weights are tied to electrode location.
- The trained sim controller itself is weak: 0/50 successes in final eval, with a median error of about 220 mm.
- pynwb is not installed, so NWB writing (`to-nwb`) is untested. The npz reference is always written.
