# Procedural surgical scenario evaluation

4 VM runs; seeds [100, 101, 102, 103]. Simulated Izhikevich culture; no living neurons or hardware.

These are reach-control perturbation experiments inspired by the cited papers, not reproductions of their surgical tasks.

| Scenario | Controller | Episodes | Success | Median error (mm) | Episodes with contacts |
|---|---|---:|---:|---:|---:|
| actuator_mismatch | culture_frozen | 12 | 0.0% | 469.8 | 0.0% |
| actuator_mismatch | culture_teach | 12 | 0.0% | 244.4 | 50.0% |
| actuator_mismatch | ik_dls | 12 | 100.0% | 2.7 | 8.3% |
| calibration_bias | culture_frozen | 12 | 0.0% | 463.7 | 0.0% |
| calibration_bias | culture_teach | 12 | 0.0% | 260.7 | 8.3% |
| calibration_bias | ik_dls | 12 | 66.7% | 9.3 | 8.3% |
| combined_stress | culture_frozen | 12 | 0.0% | 579.6 | 0.0% |
| combined_stress | culture_teach | 12 | 0.0% | 373.4 | 16.7% |
| combined_stress | ik_dls | 12 | 50.0% | 9.8 | 8.3% |
| measurement_noise | culture_frozen | 12 | 0.0% | 496.7 | 0.0% |
| measurement_noise | culture_teach | 12 | 0.0% | 244.8 | 25.0% |
| measurement_noise | ik_dls | 12 | 100.0% | 3.1 | 0.0% |
| nominal | culture_frozen | 12 | 0.0% | 485.1 | 0.0% |
| nominal | culture_teach | 12 | 0.0% | 287.5 | 16.7% |
| nominal | ik_dls | 12 | 100.0% | 3.3 | 0.0% |
| novel_targets | culture_frozen | 12 | 0.0% | 600.4 | 0.0% |
| novel_targets | culture_teach | 12 | 0.0% | 345.2 | 25.0% |
| novel_targets | ik_dls | 12 | 83.3% | 5.1 | 25.0% |

Each controller sees matched scenario seeds and targets. Contact counts use the existing convex collision geometry; they are not tissue-injury measures.
Small episode counts are exploratory. Fixed culture checkpoints are evaluated without selecting a learning rate on these targets.

See individual JSON files for dynamics-model metrics, disturbances, and per-episode outcomes.
