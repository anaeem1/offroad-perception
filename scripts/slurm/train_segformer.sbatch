#!/usr/bin/env bash
# Fine-tune SegFormer on RELLIS-3D. Submit from the repo root: sbatch scripts/slurm/train_segformer.sbatch [extra args]
#SBATCH --job-name=segformer
#SBATCH --partition=H100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/usace_share/Share_Abdullah/offroad_perception/logs/%x-%j.out
set -euo pipefail
source env/activate.sh   # sets HF_HUB_OFFLINE=1 inside jobs: pretrained weights must already be in HF_HOME
RUN="$OFFROAD_DATA_ROOT/runs/segformer_b0_$(date +%Y%m%d)_$SLURM_JOB_ID"
echo "host=$(hostname) job=$SLURM_JOB_ID run=$RUN commit=$(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
python -m ml.camera.train_segformer --run-dir "$RUN" --out docs/measurements/camera_segformer_b0_test.json \
  --workers 14 "$@"
