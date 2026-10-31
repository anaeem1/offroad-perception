#!/usr/bin/env bash
# GOOSE monitor study (pipeline/goose_monitor_eval.py): per-scan safety error and monitor signals on the held-out
# recordings (config/goose_split.yaml), clean and with injected lidar faults. Submit from the repo root; each
# argument is one run, name:fault[,fault...] (empty fault list = clean):
#   sbatch --job-name=goose-a scripts/slurm/goose_faults.sbatch clean: drop50:lidar_drop=0.5 rings64:beams=64
# -> <data root>/runs/goose_heldout_<name>/ (frames.jsonl, metrics.json)
# Env: RUN_DIR=<trained SalsaNext run> (default <data root>/weights/salsanext_goose).
#SBATCH --partition=H100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/usace_share/Share_Abdullah/offroad_perception/logs/%x-%j.out
set -euo pipefail
source env/activate.sh
RUN_DIR="${RUN_DIR:-$OFFROAD_DATA_ROOT/weights/salsanext_goose}"
echo "host=$(hostname) job=$SLURM_JOB_ID commit=$(git rev-parse --short HEAD) run_dir=$RUN_DIR runs=$*"
for item in "$@"; do
  name="${item%%:*}"; specs="${item#*:}"
  args=()
  IFS=',' read -ra fs <<< "$specs"
  for f in "${fs[@]}"; do [ -n "$f" ] && args+=(--fault "$f"); done
  echo "== $name ${args[*]:-}"
  python -m pipeline.goose_monitor_eval --run-dir "$RUN_DIR" --split heldout --tag "$name" "${args[@]}"
done
