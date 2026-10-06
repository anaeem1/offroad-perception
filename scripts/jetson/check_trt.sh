#!/usr/bin/env bash
# Run ON the Jetson. Builds an FP16 TensorRT engine for each toy ONNX model and reports parse/build
# success plus median GPU latency. Usage: scripts/jetson/check_trt.sh <dir_with_toy_onnx>
set -uo pipefail
DIR="${1:?dir with toy_*.onnx (from scripts/export_toy_onnx.py)}"
TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
printf '%-24s %-8s %s\n' model result median_gpu_ms
for onnx in "$DIR"/toy_*.onnx; do
  log="${onnx%.onnx}.trtexec.log"
  if "$TRTEXEC" --onnx="$onnx" --fp16 --saveEngine="${onnx%.onnx}.engine" > "$log" 2>&1; then
    ms=$(grep -oE 'GPU Compute Time: .*median = [0-9.]+' "$log" | grep -oE '[0-9.]+$')
    printf '%-24s %-8s %s\n' "$(basename "$onnx")" PASS "${ms:-?}"
  else
    printf '%-24s %-8s %s\n' "$(basename "$onnx")" FAIL "see $log"
  fi
done
