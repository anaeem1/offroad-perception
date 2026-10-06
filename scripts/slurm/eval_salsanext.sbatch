#!/usr/bin/env bash
# Evaluate the released RELLIS-3D SalsaNext checkpoint. Submit from the repo root.
#SBATCH --job-name=salsanext-eval
#SBATCH --partition=H100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/usace_share/Share_Abdullah/offroad_perception/logs/%x-%j.out
set -euo pipefail
source env/activate.sh
echo "host=$(hostname) job=$SLURM_JOB_ID commit=$(git rev-parse --short HEAD)"
python -m ml.lidar.eval_salsanext --split test --workers 7 --out docs/measurements/lidar_salsanext_test.json "$@"
