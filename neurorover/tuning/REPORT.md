# Plastic culture tuning report

## Method

I instrumented 200 training trials for each rule at seed 0. The tuning run used `bench.run_job` at its 200-trial checkpoint, with a balanced inner evaluation set of 75 freeze and 75 no-freeze rows drawn from the training split. The inner rows are excluded from the 500-row training stream, and their features are excluded when fitting quantile edges and normalization. The held-out validation split was not used. Each setting ran with seeds 0–2.

The grid covered R-STDP `lr ∈ {0.02, 0.2, 1, 5}`, `stim_gain ∈ {15, 30, 60}`, and `noise ∈ {3.5, 4, 4.5}`. FEP covered gain and noise at the same values; `lr` was held at 0.02 because the FEP condition is routed to plain STDP, whose update uses fixed constants and ignores `Culture.lr`.

## Diagnosis

| Rule | True label 0: mean DEPLOY−HOLD | True label 1: mean DEPLOY−HOLD | Mean `|dW|` per plastic synapse/trial | Mean `|eligibility|` at reward | Final mean absolute drift per plastic synapse | Firing rate per neuron |
|---|---:|---:|---:|---:|---:|---:|
| R-STDP | −1.366 | +0.596 | 9.69e−5 | 0.00484 | 0.00161 | 6.03 Hz (p05–p95: 5.57–6.55) |
| FEP | +1.525 | +0.374 | 0.00840 | n/a | 0.716 | 8.05 Hz (p05–p95: 6.34–10.24) |

R-STDP is learning in the expected motor direction, but the default update is tiny: the requested and applied mean absolute update are both about `9.69e−5` per plastic synapse per reward. Across the run the summed absolute drift on plastic weights is 11.44, or 0.00161 per plastic weight. No plastic weights reached 0 or `w_max=6`. The activity is neither silent nor runaway. This points to an underpowered learning signal at `lr=0.02`, not saturation or a sign inversion.

FEP has no scalar eligibility/reward signal. Its STDP drift is much larger (summed absolute drift 5,077.7 on the designated sensory-to-motor synapses; 19,563.4 across all weights), but only 0.32% of those plastic weights are at zero and 0.04% at `w_max`. It is not broadly saturated. Its mean motor difference is positive for both classes, so it favors DEPLOY even for true label 0; this is a direction/bias problem. Neither rule had a silent trial in the diagnostic run.

Detailed per-trial records are in [R-STDP diagnostics](results/diagnose_culture_rstdp_s0.json) and [FEP diagnostics](results/diagnose_culture_fep_s0.json). They include each trial’s label, motor difference, reward-time eligibility and update, firing rate, and cumulative drift.

## Sign and implementation review

The R-STDP signs are internally consistent: `codec.py:33-35` defines positive motor difference as DEPLOY; label 1 is frost and maps to DEPLOY. `codec.py:71-72` sends a positive reward for a correct action and punishment for an incorrect action, and `core.py:83-85` maps those electrodes to positive and negative `r`. In `core.py:136-140`, a prior presynaptic spike followed by a postsynaptic spike adds positive eligibility; the reverse order subtracts it. I found no sign inversion in this path.

Two implementation issues are worth fixing before drawing a final conclusion:

1. **R-STDP eligibility leaks across trials** (`core.py:94-95, 138-140`; consecutive training calls in `bench.py:144`). `Culture.reward()` applies the update but leaves eligibility intact, and `_plasticity()` decays it by only 0.995 per millisecond when there is at least one spike. Consecutive reward events are 400 ms apart, so with continuous decay an eligibility trace retains about `0.995^400 ≈ 13.5%` of its previous value; zero-spike steps skip that decay, and post-reward activity can add more trace. The next label’s reward can therefore reinforce or punish synapses based partly on the prior trial. Proposed patch: clear `c.elig` at the start of each labeled `run_trial()` (or add an explicit trial-boundary reset) so credit is local to the current station-night. I did not apply this patch.

2. **FEP is implemented as unmodulated STDP plus feedback stimulation** (`bench.py:136`; `core.py:124, 141-150`; feedback schedule in `codec.py:47-53`). `bench.py` constructs FEP cultures with `rule="stdp"`, and the non-R-STDP branch applies ordinary STDP to all excitatory synapses; it has no update gated by feedback predictability. The `lr` kwarg is consequently unused for FEP. Proposed patch: either add a distinct FEP plasticity branch that uses the predictable-versus-unpredictable feedback signal to gate task-relevant updates, or rename the condition to make clear that it tests plain STDP under different feedback patterns. I did not change these source files.

## Inner-split winners

Scores are mean AUC ± population SD across three seeds at 200 training trials.

| Rule | Best settings | Inner-split AUC |
|---|---|---:|
| R-STDP | `lr=5`, `stim_gain=30`, `noise=4` | **0.629 ± 0.084** |
| FEP | `stim_gain=30`, `noise=3.5` (`lr` has no effect) | **0.560 ± 0.050** |

The R-STDP winner improves over chance but remains well below the reservoir benchmark reported in the task context. Its three-seed spread is substantial, and the trial-boundary eligibility leak above remains in the implementation. These results support “default R-STDP is under-tuned,” but do not show that either plastic rule matches the reservoir baseline.

The complete per-seed grid is in [tune summary](results/tune_summary.json), with individual run JSON files in `results/`.

## Commands to rerun on the real evaluation set

These commands use `bench.py`’s real validation subset. They are provided for the final evaluation run and were not executed during tuning.

```powershell
.venv\Scripts\python -m neurorover.bench --condition culture_rstdp --seeds 10 --culture-kw '{"lr": 5.0, "stim_gain": 30.0, "noise": 4.0}' --tag tuned_rstdp_lr5_gain30_noise4
.venv\Scripts\python -m neurorover.bench --condition culture_fep --seeds 10 --culture-kw '{"stim_gain": 30.0, "noise": 3.5}' --tag tuned_fep_gain30_noise3_5
```
