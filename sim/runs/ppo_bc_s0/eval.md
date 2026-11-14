100 fixed-seed episodes (seeds 500000..500099), half DEPLOY / half HOLD, full task (random headland start).

| metric | ppo_randomized | controller_randomized | ppo_nominal | controller_nominal |
|---|---:|---:|---:|---:|
| success_rate | 0.72 | 0.59 | 0.80 | 0.72 |
| success_deploy | 0.82 | 0.52 | 0.84 | 0.68 |
| success_hold | 0.62 | 0.66 | 0.76 | 0.76 |
| time_to_goal_s | 15 | 18 | 14 | 19 |
| energy_J | 1092 | 1588 | 1090 | 1610 |
| energy_J_successful | 1059 | 1234 | 1019 | 1407 |
| crop_collision_rate | 0.32 | 0.49 | 0.28 | 0.63 |
| tip_rate | 0.00 | 0.00 | 0.00 | 0.00 |
| cover_coverage_on_deploy | 0.47 | 0.28 | 0.48 | 0.40 |
| final_dist_m | 0.15 | 0.16 | 0.15 | 0.15 |
