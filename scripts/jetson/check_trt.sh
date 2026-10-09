#!/usr/bin/env bash
# Run ON the Jetson. Builds an FP16 TensorRT engine for each ONNX model in a dir and reports parse/build
# success plus median GPU latency. Usage: scripts/jetson/check_trt.sh <dir_with_onnx>
set -uo pipefail
DIR="${1:?dir with .onnx models (toy models from scripts/export_toy_onnx.py, or real models)}"
TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
printf '%-34s %-8s %s\n' model result median_gpu_ms
for onnx in "$DIR"/*.onnx; do
  log="${onnx%.onnx}.trtexec.log"
  if "$TRTEXEC" --onnx="$onnx" --fp16 --saveEngine="${onnx%.onnx}.engine" > "$log" 2>&1; then
    ms=$(grep -oE 'GPU Compute Time: .*median = [0-9.]+' "$log" | grep -oE '[0-9.]+$')
    printf '%-34s %-8s %s\n' "$(basename "$onnx")" PASS "${ms:-?}"
  else
    printf '%-34s %-8s %s\n' "$(basename "$onnx")" FAIL "see $log"
  fi
done
