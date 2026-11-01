#!/usr/bin/env bash
# GOOSE monitor study (pipeline/goose_monitor_eval.py): per-scan safety error and monitor signals on the held-out
# recordings (config/goose_split.yaml), clean and with injected lidar faults. Submit from the repo root; each
# argument is one run, name:fault[,fault...] (empty fault list = clean):
#   sbatch --job-name=goose-a scripts/slurm/goose_faults.sbatch clean: drop50:lidar_drop=0.5 rings64:beams=64
# -> <data root>/runs/goose_<split>_<name>/ (frames.jsonl, metrics.json)
# Env: RUN_DIR=<trained SalsaNext run> (default <data root>/weights/salsanext_goose); SPLIT (default heldout,
#      e.g. heldout:b); EVAL_ARGS = extra goose_monitor_eval arguments ("--mc-passes 8 --ensemble <dir>");
#      PARALLEL=1 runs every argument at once on the one GPU (logs: <data root>/logs/<job>-<name>.log).
#SBATCH --partition=H100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/usace_share/Share_Abdullah/offroad_perception/logs/%x-%j.out
set -euo pipefail
source env/activate.sh
RUN_DIR="${RUN_DIR:-$OFFROAD_DATA_ROOT/weights/salsanext_goose}"
SPLIT="${SPLIT:-heldout}"
echo "host=$(hostname) job=$SLURM_JOB_ID commit=$(git rev-parse --short HEAD) run_dir=$RUN_DIR split=$SPLIT" \
     "eval_args=${EVAL_ARGS:-} runs=$*"
pids=()
for item in "$@"; do
  name="${item%%:*}"; specs="${item#*:}"
  args=()
  IFS=',' read -ra fs <<< "$specs"
  for f in "${fs[@]}"; do [ -n "$f" ] && args+=(--fault "$f"); done
  echo "== $name ${args[*]:-}"
  # shellcheck disable=SC2086  # EVAL_ARGS is a word list
  cmd=(python -m pipeline.goose_monitor_eval --run-dir "$RUN_DIR" --split "$SPLIT" --tag "$name" ${EVAL_ARGS:-} "${args[@]}")
  if [ "${PARALLEL:-0}" = 1 ]; then
    "${cmd[@]}" > "$OFFROAD_DATA_ROOT/logs/$SLURM_JOB_ID-$name.log" 2>&1 &
    pids+=($!)
  else
    "${cmd[@]}"
  fi
done
for p in "${pids[@]}"; do wait "$p"; done  # set -e: a failed run fails the job
