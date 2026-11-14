# Sim-to-real bundle: surgical_reach / reach -> Cortical Labs CL1 / Cortical Cloud (cl API)

**Source biology: SIMULATED.** This bundle was exported from a controller trained on a *simulated*
Izhikevich culture (run `surgical_reach_teach_legacy`). The simulated synaptic weights do **not** transfer to living
neurons. What this bundle gives you is the closed-loop *protocol*: which electrodes stimulate and record,
how robot observations become pulses, how spikes become joint commands, how to train (paired
stimulation), how to calibrate, and the safety envelope. A living culture must be calibrated and trained
in the dish.

Target: **Cortical Labs CL1 / Cortical Cloud (cl API)** -- status `sdk-sim-tested`.

## Files
| file | what |
|---|---|
| manifest.json | schema, source run, robot/task, observation & action spaces (units, scaling), timing, provenance |
| electrode_map.csv | every electrode: role (sensory/motor/reward/feedback/unused/unavailable), pool, joint, sign, grid x/y, um |
| encoder.json | observation -> pulse count per sensory electrode; biphasic waveform (**must be validated against vendor/lab limits**) |
| decoder.json | spike counts -> joint deltas (initial weights from the sim; **refit during calibration**) |
| training_protocol.json + stim_schedule.csv | paired-stimulation 'teach' rule as a timed schedule; session plan; controls |
| calibration_protocol.json | baseline, probe sweep, role remap, guided trials, decoder refit |
| safety.json | stimulation limits, robot velocity/workspace limits, watchdog, e-stop = hold position |
| sim_reference.npz (+ .nwb if pynwb was available) | simulated spikes/stims/behaviour of a reference episode |

## Loop timing
- one control step = 60 ms trial: pulses in 0-40 ms (10 ms apart),
  spikes counted 0-50 ms; + 85 ms paired-stimulation block when teaching
- loop rate 16.67 Hz (inference) / 6.9 Hz (teaching)
- robot: step-and-hold, one 50 ms joint-position step per decoded action

## Procedure (summary; see docs/SIM_TO_REAL.md in the repo)
1. Check every value in safety.json and encoder.json waveform against your hardware manual, electrode
   type and approved protocol. Then `python -m bnn_platform.transfer validate --bundle <this dir>`.
2. Dry run: `python -m bnn_platform.transfer run --bundle <this dir> --backend mock --episodes 2`.
3. Implement / configure the MEABackend for your rig (`bnn_platform/transfer/runtime.py`).
4. Session: calibration_protocol.json -> pre-test -> training blocks -> post-test, with the sham and
   decoder-only controls in training_protocol.json. Start with the **simulated robot**; move to the real
   robot only with a hardware e-stop and the workspace limits in safety.json.
5. Save the session recording (NWB recommended) and compare with sim_reference (rates, evoked
   responses, sensory->motor maps) -- differences are expected and are the point of the comparison.

## Target notes
- Ground channels 0, 7, 56, 63 and reference channel 4 taken from cl.analysis common MEA layout (cl-sdk 1.0.0). VERIFY on your device.
- stim_schedule maps 1:1 to cl.StimDesign(phase_us, -I, phase_us, +I) one-shot StimPlans scheduled on the 25 kHz device clock.
