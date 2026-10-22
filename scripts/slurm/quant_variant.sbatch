#!/usr/bin/env bash
# One INT8 calibration variant of SalsaNext: quantize, then score it against FP32 on the same test scans.
# Submit from the repo root:
#   sbatch --job-name=q-<name> scripts/slurm/quant_variant.sbatch <name> [quantize_salsanext args...]
#   -> <data root>/onnx/salsanext_int8_<name>.onnx, docs/measurements/lidar_int8_<name>.json
# CPU only (ONNX Runtime calibration). Percentile calibration keeps histograms of every activation: 200G.
#SBATCH --partition=H100
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --output=/usace_share/Share_Abdullah/offroad_perception/logs/%x-%j.out
set -euo pipefail
source env/activate.sh
NAME="$1"; shift
M="$OFFROAD_DATA_ROOT/onnx"
echo "host=$(hostname) job=$SLURM_JOB_ID commit=$(git rev-parse --short HEAD) variant=$NAME args=$*"
python -m ml.lidar.quantize_salsanext --out "$M/salsanext_int8_$NAME.onnx" "$@"
python -m ml.lidar.eval_lidar_onnx --onnx "$M/salsanext_rellis_op17.onnx" "$M/salsanext_int8_$NAME.onnx" \
  --every 8 --out "docs/measurements/lidar_int8_$NAME.json"
