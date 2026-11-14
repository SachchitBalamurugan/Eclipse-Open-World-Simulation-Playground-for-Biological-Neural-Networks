# sim_reference -- SIMULATED reference recording

Recorded by the exporter: the trained *simulated* culture driven by this bundle's own runtime (mock
backend, teaching off) on one held-out target, preceded by spontaneous activity with no stimulation.
It is NOT data from living neurons. Use it to compare a living culture's recordings with what the
simulation assumed (spontaneous rates per electrode, evoked responses after encoder pulses, motor-pool
counts per step, behaviour).

## sim_reference.npz (numpy.load)
| key | shape | meaning |
|---|---|---|
| spike_times_ms, spike_electrode, spike_sim_channel | (n_spikes,) | multi-unit spikes (all model neurons at an electrode) |
| stim_times_ms, stim_electrode, stim_amplitude, stim_kind | (n_stims,) | delivered pulses (kind: encode / teach_*) |
| segment_names, segment_start_ms, segment_stop_ms | (n_seg,) | 'spontaneous', 'episode' |
| step_t_ms, step_err_mm, step_action, step_counts, step_tip, step_target | per control step | trial start, error after the step, decoded action, spike counts per decoder input electrode, EE and target (m) |
| decoder_inputs | (n_inputs,) | electrode ids for step_counts columns |
| meta_json | () | JSON string: bundle, target, timing, units |

## Converting to NWB
`pip install pynwb` then `python -m bnn_platform.transfer to-nwb --bundle <bundle dir>`. Mapping:
electrodes -> NWB electrodes table (x/y in um, group = MEA, location = 'simulated culture'); spikes ->
Units table, one multi-unit per electrode (spike_times in s); pulses -> TimeIntervals 'stimulation'
(start/stop, electrode, amplitude, kind); behaviour -> processing/behavior TimeSeries (err_mm,
action, tip, target). Real recordings exported by vendor software (MaxLab, MCS DataManager ->
HDF5/NWB, CL recordings) can then be loaded with the same tools for side-by-side comparison.
