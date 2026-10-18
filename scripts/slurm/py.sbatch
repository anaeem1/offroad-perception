#!/usr/bin/env bash
# Run a Python module inside the project env on a compute node. Submit from the repo root:
#   sbatch [--job-name=x -c N --mem=M --gres=gpu:1] scripts/slurm/py.sbatch <module> [args...]
#SBATCH --partition=H100
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/usace_share/Share_Abdullah/offroad_perception/logs/%x-%j.out
set -euo pipefail
source env/activate.sh
echo "host=$(hostname) job=$SLURM_JOB_ID commit=$(git rev-parse --short HEAD) module=$1"
python -m "$@"
