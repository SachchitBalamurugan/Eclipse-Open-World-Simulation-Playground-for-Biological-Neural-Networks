# surgical/ - simulated culture driving one arm of a SimReady surgical cart

Robot: NVIDIA SimReady `robotic_system_a01` (CC-BY-4.0, see
`assets/surgical/robotic_system_a01/.metadata/LICENSE.txt`), a four-arm patient-side cart.

| file | what |
|---|---|
| `usd_to_mjcf.py` | pxr walk of the UsdPhysics articulation -> `model/cart.xml` + OBJ meshes (the pip MuJoCo wheel has no USD decoder) |
| `validate_model.py` | 2 s zero-command hold under gravity + arm-1 kinematic tree |
| `arm.py` | arm-1 joints, tip site, workspace sampling, held-out target generator, collision check |
| `reach_env.py` | Gymnasium reach env (DR on masses/damping, target noise) |
| `controllers.py` | damped-least-squares IK, random, episode runner |
| `culture_ctrl.py` | `neurorover.culture.core.Culture` encoder/decoder + paired-stimulation teaching |
| `train_culture.py` | learning curve + final 50-target eval for `teach` / `frozen` |
| `ppo_train.py` | SB3 PPO baseline, time-boxed |
| `evaluate.py` | all controllers on the same 50 held-out targets -> `results/eval.json` |
| `record_and_render.py` | culture replay -> `results/trajectory.json`, spikes/stims CSV, MP4/GIF |
| `make_report.py` | eval table + `results/learning_curve.png` |

Run order (from repo root, `.venv/Scripts/python`):
`surgical/usd_to_mjcf.py` -> `surgical/validate_model.py` -> `surgical/train_culture.py teach 30 0.08`
and `... frozen 0` -> `surgical/ppo_train.py 1200` -> `surgical/evaluate.py` -> `surgical/record_and_render.py teach 3`
-> `surgical/make_report.py`. See `REPORT.md` for results.
