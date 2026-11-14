# Benchmark summary

Loaded 484 condition/seed result records from neurorover/results/ (484 files with valid AUC data).
This script scans the folder again each time it runs. Confidence intervals are percentile bootstrap intervals over seeds (20,000 resamples). Intervals are unavailable for a single seed.
Skipped 60 unreadable, incomplete, or unsupported result file(s); they will be reconsidered on the next run.

## AUC by condition and sample budget

| Condition | 25 | 50 | 100 | 200 |
|---|---:|---:|---:|---:|
| culture_frozen | 0.489 [0.467, 0.511]; n=50 | 0.484 [0.461, 0.506]; n=50 | 0.490 [0.468, 0.512]; n=50 | 0.490 [0.467, 0.512]; n=50 |
| culture_fep | 0.490 [0.466, 0.515]; n=50 | 0.492 [0.467, 0.516]; n=50 | 0.499 [0.473, 0.525]; n=50 | 0.528 [0.498, 0.558]; n=50 |
| culture_rstdp | 0.485 [0.463, 0.507]; n=50 | 0.491 [0.471, 0.512]; n=50 | 0.488 [0.463, 0.513]; n=50 | 0.492 [0.472, 0.513]; n=50 |
| culture_teach | 0.561 [0.536, 0.586]; n=34 | 0.595 [0.566, 0.622]; n=34 | 0.678 [0.654, 0.700]; n=34 | 0.726 [0.698, 0.752]; n=34 |
| culture_reservoir | 0.832 [0.819, 0.844]; n=50 | 0.855 [0.849, 0.861]; n=50 | 0.869 [0.865, 0.873]; n=50 | 0.872 [0.868, 0.876]; n=50 |
| random | 0.501 [0.491, 0.512]; n=50 | 0.501 [0.493, 0.509]; n=50 | 0.500 [0.490, 0.510]; n=50 | 0.497 [0.488, 0.505]; n=50 |
| heuristic_tmin | 0.863 [0.863, 0.863]; n=50 | 0.863 [0.863, 0.863]; n=50 | 0.863 [0.863, 0.863]; n=50 | 0.863 [0.863, 0.863]; n=50 |
| logistic | 0.838 [0.824, 0.851]; n=50 | 0.863 [0.857, 0.869]; n=50 | 0.877 [0.874, 0.880]; n=50 | 0.887 [0.885, 0.889]; n=50 |
| mlp | 0.808 [0.784, 0.829]; n=50 | 0.840 [0.829, 0.849]; n=50 | 0.851 [0.845, 0.857]; n=50 | 0.859 [0.853, 0.864]; n=50 |
| logistic_rawfeat | 0.866 [0.855, 0.875]; n=50 | 0.885 [0.880, 0.890]; n=50 | 0.897 [0.893, 0.900]; n=50 | 0.905 [0.903, 0.907]; n=50 |

Each cell gives mean AUC [95% bootstrap CI]; n=number of seeds.

## H1: plastic culture conditions vs frozen culture

Paired comparisons use the intersection of seeds at each budget. The reported interval bootstraps the per-seed AUC difference (left condition minus right condition); a difference is called significant when its two-sided 95% interval excludes zero.

- **culture_fep − culture_frozen, 25 samples:** mean paired Δ=+0.0007 (95% bootstrap CI [-0.0118, +0.0128], n=50): **not significant** at the 5% level (CI includes zero).
- **culture_fep − culture_frozen, 50 samples:** mean paired Δ=+0.0082 (95% bootstrap CI [-0.0026, +0.0192], n=50): **not significant** at the 5% level (CI includes zero).
- **culture_fep − culture_frozen, 100 samples:** mean paired Δ=+0.0093 (95% bootstrap CI [-0.0063, +0.0251], n=50): **not significant** at the 5% level (CI includes zero).
- **culture_fep − culture_frozen, 200 samples:** mean paired Δ=+0.0387 (95% bootstrap CI [+0.0189, +0.0589], n=50): **significant** at the 5% level (CI excludes zero).
- **culture_rstdp − culture_frozen, 25 samples:** mean paired Δ=-0.0045 (95% bootstrap CI [-0.0164, +0.0077], n=50): **not significant** at the 5% level (CI includes zero).
- **culture_rstdp − culture_frozen, 50 samples:** mean paired Δ=+0.0078 (95% bootstrap CI [-0.0023, +0.0174], n=50): **not significant** at the 5% level (CI includes zero).
- **culture_rstdp − culture_frozen, 100 samples:** mean paired Δ=-0.0025 (95% bootstrap CI [-0.0142, +0.0085], n=50): **not significant** at the 5% level (CI includes zero).
- **culture_rstdp − culture_frozen, 200 samples:** mean paired Δ=+0.0027 (95% bootstrap CI [-0.0087, +0.0141], n=50): **not significant** at the 5% level (CI includes zero).
- **culture_teach − culture_frozen, 25 samples:** mean paired Δ=+0.0623 (95% bootstrap CI [+0.0485, +0.0763], n=34): **significant** at the 5% level (CI excludes zero).
- **culture_teach − culture_frozen, 50 samples:** mean paired Δ=+0.0948 (95% bootstrap CI [+0.0785, +0.1104], n=34): **significant** at the 5% level (CI excludes zero).
- **culture_teach − culture_frozen, 100 samples:** mean paired Δ=+0.1773 (95% bootstrap CI [+0.1610, +0.1933], n=34): **significant** at the 5% level (CI excludes zero).
- **culture_teach − culture_frozen, 200 samples:** mean paired Δ=+0.2289 (95% bootstrap CI [+0.2080, +0.2503], n=34): **significant** at the 5% level (CI excludes zero).

## H2: reservoir culture vs logistic and MLP

The same paired bootstrap rule is used at each sample budget.
Significance calls are unadjusted across the tested budgets and contrasts.

- **culture_reservoir − logistic, 25 samples:** mean paired Δ=-0.0062 (95% bootstrap CI [-0.0180, +0.0048], n=50): **not significant** at the 5% level (CI includes zero).
- **culture_reservoir − logistic, 50 samples:** mean paired Δ=-0.0077 (95% bootstrap CI [-0.0147, -0.0007], n=50): **significant** at the 5% level (CI excludes zero).
- **culture_reservoir − logistic, 100 samples:** mean paired Δ=-0.0078 (95% bootstrap CI [-0.0126, -0.0030], n=50): **significant** at the 5% level (CI excludes zero).
- **culture_reservoir − logistic, 200 samples:** mean paired Δ=-0.0146 (95% bootstrap CI [-0.0183, -0.0110], n=50): **significant** at the 5% level (CI excludes zero).
- **culture_reservoir − mlp, 25 samples:** mean paired Δ=+0.0239 (95% bootstrap CI [+0.0068, +0.0418], n=50): **significant** at the 5% level (CI excludes zero).
- **culture_reservoir − mlp, 50 samples:** mean paired Δ=+0.0155 (95% bootstrap CI [+0.0064, +0.0254], n=50): **significant** at the 5% level (CI excludes zero).
- **culture_reservoir − mlp, 100 samples:** mean paired Δ=+0.0184 (95% bootstrap CI [+0.0125, +0.0240], n=50): **significant** at the 5% level (CI excludes zero).
- **culture_reservoir − mlp, 200 samples:** mean paired Δ=+0.0137 (95% bootstrap CI [+0.0085, +0.0190], n=50): **significant** at the 5% level (CI excludes zero).

These seed counts are small and the intervals describe uncertainty across the observed seeds; treat the significance calls as seed-level, exploratory comparisons.
