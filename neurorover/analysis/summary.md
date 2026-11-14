# Benchmark summary

Loaded 90 condition/seed result records from neurorover/results/ (90 files with valid AUC data).
This script scans the folder again each time it runs. Confidence intervals are percentile bootstrap intervals over seeds (20,000 resamples). Intervals are unavailable for a single seed.

## AUC by condition and sample budget

| Condition | 25 | 50 | 100 | 200 |
|---|---:|---:|---:|---:|
| culture_frozen | 0.521 [0.477, 0.566]; n=10 | 0.513 [0.468, 0.558]; n=10 | 0.528 [0.483, 0.575]; n=10 | 0.519 [0.469, 0.567]; n=10 |
| culture_fep | 0.537 [0.489, 0.589]; n=10 | 0.528 [0.479, 0.576]; n=10 | 0.517 [0.471, 0.562]; n=10 | 0.557 [0.492, 0.613]; n=10 |
| culture_rstdp | 0.523 [0.476, 0.570]; n=10 | 0.513 [0.468, 0.561]; n=10 | 0.519 [0.459, 0.581]; n=10 | 0.516 [0.475, 0.563]; n=10 |
| culture_reservoir | 0.838 [0.812, 0.857]; n=10 | 0.857 [0.847, 0.867]; n=10 | 0.870 [0.865, 0.875]; n=10 | 0.864 [0.856, 0.873]; n=10 |
| random | 0.506 [0.487, 0.524]; n=10 | 0.498 [0.486, 0.509]; n=10 | 0.484 [0.462, 0.506]; n=10 | 0.495 [0.483, 0.508]; n=10 |
| heuristic_tmin | 0.863 [0.863, 0.863]; n=10 | 0.863 [0.863, 0.863]; n=10 | 0.863 [0.863, 0.863]; n=10 | 0.863 [0.863, 0.863]; n=10 |
| logistic | 0.838 [0.814, 0.860]; n=10 | 0.864 [0.846, 0.877]; n=10 | 0.879 [0.873, 0.885]; n=10 | 0.889 [0.885, 0.892]; n=10 |
| mlp | 0.812 [0.754, 0.852]; n=10 | 0.838 [0.814, 0.858]; n=10 | 0.857 [0.846, 0.870]; n=10 | 0.860 [0.852, 0.868]; n=10 |
| logistic_rawfeat | 0.866 [0.850, 0.880]; n=10 | 0.886 [0.876, 0.896]; n=10 | 0.900 [0.893, 0.906]; n=10 | 0.908 [0.904, 0.911]; n=10 |

Each cell gives mean AUC [95% bootstrap CI]; n=number of seeds.

## H1: plastic culture conditions vs frozen culture

Paired comparisons use the intersection of seeds at each budget. The reported interval bootstraps the per-seed AUC difference (left condition minus right condition); a difference is called significant when its two-sided 95% interval excludes zero.

- **culture_fep − culture_frozen, 25 samples:** mean paired Δ=+0.0152 (95% bootstrap CI [-0.0138, +0.0433], n=10): **not significant** at the 5% level (CI includes zero).
- **culture_fep − culture_frozen, 50 samples:** mean paired Δ=+0.0152 (95% bootstrap CI [-0.0098, +0.0419], n=10): **not significant** at the 5% level (CI includes zero).
- **culture_fep − culture_frozen, 100 samples:** mean paired Δ=-0.0110 (95% bootstrap CI [-0.0368, +0.0136], n=10): **not significant** at the 5% level (CI includes zero).
- **culture_fep − culture_frozen, 200 samples:** mean paired Δ=+0.0382 (95% bootstrap CI [+0.0002, +0.0833], n=10): **significant** at the 5% level (CI excludes zero).
- **culture_rstdp − culture_frozen, 25 samples:** mean paired Δ=+0.0018 (95% bootstrap CI [-0.0267, +0.0283], n=10): **not significant** at the 5% level (CI includes zero).
- **culture_rstdp − culture_frozen, 50 samples:** mean paired Δ=-0.0003 (95% bootstrap CI [-0.0311, +0.0267], n=10): **not significant** at the 5% level (CI includes zero).
- **culture_rstdp − culture_frozen, 100 samples:** mean paired Δ=-0.0085 (95% bootstrap CI [-0.0346, +0.0137], n=10): **not significant** at the 5% level (CI includes zero).
- **culture_rstdp − culture_frozen, 200 samples:** mean paired Δ=-0.0025 (95% bootstrap CI [-0.0242, +0.0195], n=10): **not significant** at the 5% level (CI includes zero).

## H2: reservoir culture vs logistic and MLP

The same paired bootstrap rule is used at each sample budget.
Significance calls are unadjusted across the tested budgets and contrasts.

- **culture_reservoir − logistic, 25 samples:** mean paired Δ=+0.0000 (95% bootstrap CI [-0.0128, +0.0143], n=10): **not significant** at the 5% level (CI includes zero).
- **culture_reservoir − logistic, 50 samples:** mean paired Δ=-0.0076 (95% bootstrap CI [-0.0210, +0.0066], n=10): **not significant** at the 5% level (CI includes zero).
- **culture_reservoir − logistic, 100 samples:** mean paired Δ=-0.0089 (95% bootstrap CI [-0.0178, -0.0008], n=10): **significant** at the 5% level (CI excludes zero).
- **culture_reservoir − logistic, 200 samples:** mean paired Δ=-0.0244 (95% bootstrap CI [-0.0319, -0.0158], n=10): **significant** at the 5% level (CI excludes zero).
- **culture_reservoir − mlp, 25 samples:** mean paired Δ=+0.0262 (95% bootstrap CI [+0.0009, +0.0592], n=10): **significant** at the 5% level (CI excludes zero).
- **culture_reservoir − mlp, 50 samples:** mean paired Δ=+0.0186 (95% bootstrap CI [-0.0008, +0.0406], n=10): **not significant** at the 5% level (CI includes zero).
- **culture_reservoir − mlp, 100 samples:** mean paired Δ=+0.0130 (95% bootstrap CI [-0.0033, +0.0275], n=10): **not significant** at the 5% level (CI includes zero).
- **culture_reservoir − mlp, 200 samples:** mean paired Δ=+0.0046 (95% bootstrap CI [-0.0047, +0.0140], n=10): **not significant** at the 5% level (CI includes zero).

These seed counts are small and the intervals describe uncertainty across the observed seeds; treat the significance calls as seed-level, exploratory comparisons.
