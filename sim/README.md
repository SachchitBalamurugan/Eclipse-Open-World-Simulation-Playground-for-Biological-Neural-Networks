# sim/ : MuJoCo + PPO model of the frost-guard rover

A physics model of the NeuroRover field robot built from the parts in `hardware/BOM.md`, a Gymnasium task
(drive to a crop row, park aligned, unroll the frost cloth on DEPLOY nights), PPO training with domain
randomisation, an evaluation against a hand-written controller, and a season loop where the simulated
neuron culture makes each night's DEPLOY/HOLD call and the trained policy drives the rover.

The training recipe follows the Open Duck Mini project (a BDX-style biped trained in MuJoCo Playground and
transferred to a real robot): per-episode dynamics randomisation, observation noise, observation and action
latency, a low-pass action filter and an action-rate penalty. This is our own code. Nothing was copied from
Open Duck, so no third-party licence file is needed here. (Open_Duck_Mini is Apache-2.0. The
Open_Duck_Playground repo had no LICENSE file when we checked.)

**Status:** CPU-only, one training seed, roughly 25 minutes of PPO. The results are in the sections below.

## Files

| file | what |
|---|---|
| `params.py` | every physical number, each with its source (BOM / datasheet / Duck recipe / EST) |
| `build_model.py` | generates `assets/rover.xml` from `params.py`, plus `assets/rover_props.xml` with the SimReady props |
| `assets/rover.xml` | MJCF: 4-wheel skid-steer rover, mast, actuator-driven boom, 4-panel hinged cloth chain, heightfield field, 5 crop rows, station marker, sensors |
| `rover_env.py` | `FrostRoverEnv` (Gymnasium) with DR, noise, latency, action filter, energy model, reward, trajectory export |
| `controller.py` | baseline: pure-pursuit path follower + proportional heading/stop logic, same noisy observations |
| `train_ppo.py` | SB3 PPO, `SubprocVecEnv` (5 envs), held-out success curve, checkpoints, `progress.csv` / `progress.png` |
| `eval.py` | 100 fixed-seed randomised episodes, PPO vs controller (+ nominal no-DR model), writes `eval.json` / `eval.md` |
| `night_loop.py` | a real validation season: culture_teach decision -> rover episode, writes `season.csv` / `season_summary.*` |
| `render.py` | offscreen MP4 + stills + `trajectory.json` (per-step poses of chassis, wheels, boom, roll, cloth panels) |
| `vm_train.sh` | run one training seed headless on a Linux VM (not launched) |
| `runs/ppo_s0/` | the trained run (see below) |
| `runs/ppo_v1_abandoned/` | the first reward design, stopped at 1M steps (policy learned to sit still; kept for honesty) |

## Reproduce

```sh
.venv/Scripts/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/pip install mujoco gymnasium stable-baselines3 imageio imageio-ffmpeg pandas matplotlib
.venv/Scripts/python -m sim.build_model
.venv/Scripts/python -m sim.train_ppo --run ppo_s0 --seed 0 --minutes 26 --envs 5
.venv/Scripts/python -m sim.eval --run ppo_s0
.venv/Scripts/python -m sim.render --run ppo_s0 --seed 500001 --deploy 1
.venv/Scripts/python -m sim.night_loop --run ppo_s0 --nights 30
```

Versions used: Python 3.12, mujoco 4.0.1, gymnasium 1.4.0, stable-baselines3 2.9.2, torch 2.15.0+cpu, numpy 2.5.4
(`runs/ppo_s0/versions.json`). Windows 11, 8 CPU cores, no CUDA. The offscreen renderer worked with the default
GL backend on Windows.

## Model

* **Rover:** 0.60 x 0.40 m deck (BOM #18), 2020 rails and an 800 mm mast (#17). Four 120 mm Wild Thumper wheels (#8)
  on Pololu #4753 50:1 gearmotors (#5). Each motor is a MuJoCo `general` actuator with the DC torque-speed line
  `tau = 2.06*u - (2.06/20.9)*omega`: stall 21 kg·cm, no-load 200 rpm at the output. Simulated top speed is
  1.14 m/s against 1.26 m/s on paper (rolling and contact losses). Masses follow the BOM column: battery 2.5 kg,
  deck+rails 3.1 kg, 4x motor 0.22 kg, 4x wheel 0.20 kg, enclosure+Pi+drivers+wiring 0.93 kg, shield+SHT45 0.61 kg,
  GNSS+antenna 0.18 kg, mast 0.6 kg, actuator 0.06 kg, cloth 0.8 kg, tube+arm 0.4 kg. Total **10.86 kg**
  (BOM: 10.8 kg est.).
* **Wheels** collide as spheres (r = 0.06 m), which is cheap and stable on a heightfield. They are drawn as cylinders.
* **Cover:** an Actuonix L16-100-35 (100 mm stroke, 32 mm/s, 50 N) drives a 0.45 m boom through a 64 mm crank. The
  stroke swings the boom 90°, so the actuator can hold at most 50 N x 0.064 m = 3.2 N·m. The env rate-limits the
  boom setpoint to the actuator's stroke speed (about 0.5 rad/s, 3.1 s to deploy). The cloth is a chain of 4 hinged
  0.6 x 0.2 m panels hanging from the roll tube. The panels collide with the ground and with the crop rows, so the
  cloth drapes over the row. We picked a panel chain over a MuJoCo flex cloth to keep the training sim cheap.
* **Field:** a 16 x 12 m heightfield with gentle random sinusoid bumps (amplitude randomised from 0 to 3 cm), five
  6 m crop rows 1.5 m apart, and a station marker (mocap) beside the target row.
* **Sensors:** wheel encoders (jointvel), IMU accelerometer + gyro, GNSS antenna position (framepos on the mast
  site), chassis orientation (framequat, used as the heading sensor) and the boom potentiometer. The SHT45 reading
  is synthesised in `night_loop.py`: the night's air temperature plus N(0, 0.1 °C).

## Task, observation, action, reward

* **Episode:** a random start in the headland (any heading), a random target row and station x. Drive into the lane
  beside the row, stop within 0.2 m and 11° of the station pose with speed under 0.1 m/s, and hold that for
  5 steps. On DEPLOY, the boom must also be past 0.85 of its travel with at least half of the cloth panels over the
  row. Time limit: 30 s at 20 Hz. During training, 50% of episodes start in or near the lane (curriculum). All
  evaluation uses the full task.
* **Observation (22):** goal vector from noisy GNSS (world and body frame), distance, sin/cos heading error,
  station-along-row, noisy gyro and accelerometer, left/right encoder speeds, boom potentiometer, previous action,
  the DEPLOY flag and time remaining. Delayed by 0 to 2 steps.
* **Action (3):** left and right motor duty (-1..1), plus the boom command. Delayed by 0 to 1 step, then a
  first-order low-pass filter (4 Hz cutoff).
* **Reward:** 10 x progress along the path distance (via the lane entry), heading alignment near the station,
  -0.001 x motor electrical energy (J, from a DC current model `I = |tau|/Kt + I0`), -0.05 x |Δa|²,
  -0.5 per step with a wheel over a crop row, a small time cost, braking near the goal, and a boom shaping term
  (reward extension at the station on DEPLOY, penalise it otherwise). Success adds +20, plus 10 x cloth coverage on
  DEPLOY. Tipping costs -20 and leaving the field costs -10.
* **Domain randomisation per episode** (`params.DR`): ground/tyre friction U(0.5, 1.0) [Duck]; body mass x U(0.9, 1.1) [Duck];
  payload ±0.5 kg; CoM shift ±3 cm; motor strength x U(0.85, 1.1); sensor-noise multiplier U(0.5, 1.5); GNSS σ
  2 to 8 cm; obs latency 0 to 2 steps; action latency 0 to 1 step [Duck: 0 to 3 steps at 50 Hz]; terrain amplitude
  0 to 3 cm; wind gusts on the unrolled cloth of 0 to 1.5 N per panel, changing every 0.5 to 2 s. Noise scales
  follow Duck where they exist (gyro 0.1 rad/s, accelerometer 0.05 g).
* **Energy accounting:** motor electrical energy + actuator (12.8 V x 0.65 A while moving, EST) + 0.6 A idle
  load (Pi 5 + sensors, BOM).

## PPO

SB3 PPO with an MLP of 128x128 for both policy and value, `log_std_init` -0.5, n_steps 1024 x 5 envs, batch 1024,
5 epochs, lr 3e-4, γ 0.99, λ 0.95, clip 0.2, entropy 0.003, and VecNormalize on observations and rewards.
Open Duck trains with Brax PPO on GPU, using the MuJoCo Playground config for the Berkeley Humanoid with thousands of
parallel envs and about 1e8 to 3e8 steps. We have 5 CPU envs and a few million steps, so this run is a short
version of that recipe, not a reproduction. A plateau check stops training early once held-out success flattens
after reaching 0.6. Otherwise training stops at the wall-clock budget.

RESULTS_PLACEHOLDER

## Sources

* Open Duck Mini: https://github.com/apirrone/Open_Duck_Mini (Apache-2.0). Training repo:
  https://github.com/apirrone/Open_Duck_Playground. The recipe numbers above come from
  `playground/common/randomize.py` (floor friction U(0.5, 1.0), mass x U(0.9, 1.1), actuator gain x U(0.9, 1.1)) and
  `playground/open_duck_mini_v2/joystick.py` (ctrl_dt 0.02, sim_dt 0.002, action/IMU delay 0 to 3 steps, noise
  scales gyro 0.1 and accelerometer 0.05, action_rate reward -0.5, random pushes of 0.1 to 1.0 every 5 to 10 s, and a
  commented-out LowPassActionFilter). Read on 2026-11-14.
* MuJoCo Playground (Duck's base): https://github.com/google-deepmind/mujoco_playground, paper https://arxiv.org/abs/2502.08844
* Pololu #4753 (50:1 37Dx70L, 12 V, 64 CPR): https://www.pololu.com/product/4753 : 200 rpm, 21 kg·cm and 5.5 A
  stall (extrapolated), 0.2 A no-load, 10 kg·cm recommended continuous, 3200 CPR at the output.
* Actuonix L16 (L16-100-35-12-P): https://www.actuonix.com/l16 : the 100 mm / 32 mm/s / 50 N figures come from
  the BOM. We could not re-fetch the datasheet (HTTP 403), and the 0.65 A current is an estimate.
* Sensirion SHT45: https://sensirion.com/products/catalog/SHT45 (±0.1 °C typ). u-blox NEO-M9N:
  https://www.u-blox.com/en/product/neo-m9n-module (about 1.5 m CEP). Bioenno BLF-1220A (12.8 V, 20 Ah), Dagu Wild
  Thumper 120 mm wheel, Cytron MDD10A: values as listed in `hardware/BOM.md`.
* Everything marked `EST` in `params.py` is our estimate, not a datasheet value.

## Known gaps / what is not realistic

* **Boom torque:** the 50 N L16 cannot hold the loaded boom by itself (gravity load is about 5 N·m against
  3.2 N·m available). The model adds a 3 N·m/rad counterbalance torsion spring that is **not in the BOM**. Even
  with the spring, the boom stalls at about 80° instead of 90°. A real build needs that spring or a gas strut, or
  the 63:1 or 150:1 L16 variant.
* **GNSS:** the NEO-M9N's about 1.5 m CEP is far too coarse for a 0.45 m lane. The sim assumes RTK-grade 2 to 8 cm
  noise (for example a ZED-F9P plus a base station, not in the BOM), or row-following perception.
* **Heading** comes from the chassis orientation plus noise. A real rover needs a magnetometer or dual antennas,
  and neither is in the BOM.
* **Cloth:** 4 rigid hinged panels stand in for a rolled fabric. There is no unrolling, the tube does not spin,
  and the drag model is crude (wind is a random force that scales with boom extension).
* **Wheels and terrain:** sphere-on-heightfield contact with isotropic friction has no tyre lug or soil-sinkage
  model. Skid-steer turning comes from friction only.
* **Motors:** the gearmotors are an ideal linear DC model with no gearbox backlash, no thermal limit and no
  driver current limit. The battery is a constant 12.8 V with no sag (motor strength DR partly covers sag).
* **Crops:** crop rows are rigid boxes for the cloth. Wheel-over-crop damage is detected geometrically, with no
  plant physics.
* Only one training seed, about 25 minutes on CPU. There is no real-robot data, so "sim-to-real-minded" means DR,
  noise, latency, action filtering and a nominal-vs-randomised evaluation. Transfer has not been demonstrated.
