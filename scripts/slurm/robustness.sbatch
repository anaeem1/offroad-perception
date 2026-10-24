#!/usr/bin/env bash
# Robustness study (pipeline/faults.py): replay the 00001 test block with injected sensor faults, then score
# tracking exactly like the baseline (ADR-0009: min 30 points, existence gate 0.8, v2 pseudo-GT).
# Submit from the repo root; each argument is one run, name:fault[,fault...] (empty fault list = baseline):
#   sbatch --job-name=robust-a scripts/slurm/robustness.sbatch baseline: drop50:lidar_drop=0.5
# -> <data root>/runs/replay_00001_robust_<name>/ (metrics.json, tracking.json); pipeline/summarize_robustness.py
#SBATCH --partition=H100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/usace_share/Share_Abdullah/offroad_perception/logs/%x-%j.out
set -euo pipefail
source env/activate.sh
echo "host=$(hostname) job=$SLURM_JOB_ID commit=$(git rev-parse --short HEAD) runs=$*"
for item in "$@"; do
  name="${item%%:*}"; specs="${item#*:}"
  args=()
  IFS=',' read -ra fs <<< "$specs"
  for f in "${fs[@]}"; do [ -n "$f" ] && args+=(--fault "$f"); done
  run="$OFFROAD_DATA_ROOT/runs/replay_00001_robust_$name"
  echo "== $name ${args[*]:-}"
  python -m pipeline.run_replay --seq 00001 --scans 0:1047 --tag "robust_$name" --memory-s 5 --default-only \
    --save-grids 0 "${args[@]}"
  python -m pipeline.gt_objects --run "$run"
  python -m pipeline.track_offline --run "$run" --min-points 30 --out tracks_mp30.jsonl
  python -m pipeline.eval_tracking --run "$run" --tracks tracks_mp30.jsonl --gt-file gt_objects_v2.jsonl \
    --min-gt-height 0.5 --gates 0.8 --out "$run/tracking.json"
done
