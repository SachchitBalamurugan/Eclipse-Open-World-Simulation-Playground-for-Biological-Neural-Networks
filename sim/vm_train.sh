#!/usr/bin/env bash
# Run one PPO training seed headless on a Linux VM (e.g. a Freestyle VM). Not launched by anything in this repo.
#
#   usage:  bash sim/vm_train.sh <seed> [minutes] [n_envs]
#   e.g.    bash sim/vm_train.sh 3 35 6
#
# Expects the repo checked out; training needs only sim/ (night_loop.py additionally needs neurorover/ + resources/).
# Outputs sim/runs/ppo_s<seed>/ (progress.csv, progress.png, best_model.zip, vecnormalize.pkl, eval.json)
# and a tarball sim/runs/ppo_s<seed>.tgz to collect.
set -euo pipefail
SEED="${1:?seed}"
MINUTES="${2:-35}"
ENVS="${3:-6}"
cd "$(dirname "$0")/.."

PY=${PYTHON:-python3}
if [ ! -d .venv-sim ]; then
  $PY -m venv .venv-sim
fi
. .venv-sim/bin/activate
pip install -q --upgrade pip
# same versions as the local run (see sim/runs/ppo_s0/versions.json)
pip install -q torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
pip install -q mujoco==4.0.1 gymnasium==1.4.0 stable-baselines3==2.9.2 numpy==2.5.4 pandas matplotlib imageio imageio-ffmpeg

# Training and evaluation do not render. Leave MUJOCO_GL unset: Freestyle workers
# do not provide EGL, and importing MuJoCo with EGL forced fails there.
export OMP_NUM_THREADS=1
python -m sim.build_model
python -m sim.train_ppo --run "ppo_s${SEED}" --seed "$SEED" --minutes "$MINUTES" --envs "$ENVS" --bc 120
python -m sim.eval --run "ppo_s${SEED}" || true
tar czf "sim/runs/ppo_s${SEED}.tgz" -C sim/runs "ppo_s${SEED}"
echo "done: sim/runs/ppo_s${SEED}.tgz"
