#!/usr/bin/env bash
# Run ON the Jetson. Latency of the two segmentation models under the levers we can pull without
# retraining: precision (FP16 / INT8 speed bound), DLA offload, input resolution, and concurrency.
# Usage: scripts/jetson/latency_sweep.sh <models_dir> | tee latency_sweep.txt
# INT8 here is uncalibrated: it bounds the achievable speed only; accuracy needs a calibration cache.
set -uo pipefail
M="${1:?dir with salsanext_rellis_op17.onnx, segformer_b0_640x1024_op17.onnx, segformer_b0_480x768_op17.onnx}"
TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
TIMING="--useSpinWait --warmUp=2000 --duration=15"

echo "== board"
nvpmodel -q 2>/dev/null | head -1
(sudo -n jetson_clocks --show 2>/dev/null | grep -E "GPU|EMC" | head -3) || echo "(jetson_clocks --show needs sudo)"
echo "-- top CPU processes"; ps -eo pid,user,pcpu,pmem,comm --sort=-pcpu | head -8
echo "-- GPU users (tegrastats sample)"; timeout 3 tegrastats --interval 1000 2>/dev/null | head -2

run() {  # name onnx extra-flags...
  local name="$1" onnx="$2"; shift 2
  local log="$M/sweep_${name}.log"
  if "$TRTEXEC" --onnx="$M/$onnx" $TIMING "$@" > "$log" 2>&1; then
    local med p99
    med=$(grep -oE 'GPU Compute Time: .*median = [0-9.]+' "$log" | grep -oE '[0-9.]+$')
    p99=$(grep -oE 'GPU Compute Time: .*percentile\(99%\) = [0-9.]+' "$log" | grep -oE '[0-9.]+$')
    printf '%-26s %-8s median %8s ms  p99 %8s ms\n' "$name" PASS "${med:-?}" "${p99:-?}"
  else
    printf '%-26s %-8s see %s\n' "$name" FAIL "$log"
  fi
}

echo; echo "== single model, sequential"
run salsanext_fp16          salsanext_rellis_op17.onnx --fp16 --saveEngine="$M/salsanext_fp16.engine"
run salsanext_fp16_opt5     salsanext_rellis_op17.onnx --fp16 --builderOptimizationLevel=5
run salsanext_int8_bound    salsanext_rellis_op17.onnx --fp16 --int8
run salsanext_dla_fp16      salsanext_rellis_op17.onnx --fp16 --useDLACore=0 --allowGPUFallback --saveEngine="$M/salsanext_dla.engine"
run segformer640_fp16       segformer_b0_640x1024_op17.onnx --fp16 --saveEngine="$M/segformer640_fp16.engine"
run segformer480_fp16       segformer_b0_480x768_op17.onnx --fp16 --saveEngine="$M/segformer480_fp16.engine"
run segformer640_int8_bound segformer_b0_640x1024_op17.onnx --fp16 --int8
echo "-- DLA placement (layers that fell back to GPU)"
grep -E "Layer\(|falling back|not supported on DLA|DLA Layers|GPU Layers" "$M/sweep_salsanext_dla_fp16.log" | head -15

echo; echo "== concurrent: both models at once (as in the runtime)"
conc() {  # name lidar_engine camera_engine
  local name="$1"
  "$TRTEXEC" --loadEngine="$M/$2" $TIMING ${4:-} > "$M/conc_${name}_lidar.log" 2>&1 &
  local a=$!
  "$TRTEXEC" --loadEngine="$M/$3" $TIMING > "$M/conc_${name}_camera.log" 2>&1 &
  local b=$!
  wait $a $b
  for part in lidar camera; do
    med=$(grep -oE 'GPU Compute Time: .*median = [0-9.]+' "$M/conc_${name}_${part}.log" | grep -oE '[0-9.]+$')
    p99=$(grep -oE 'GPU Compute Time: .*percentile\(99%\) = [0-9.]+' "$M/conc_${name}_${part}.log" | grep -oE '[0-9.]+$')
    printf '%-26s %-7s median %8s ms  p99 %8s ms\n' "$name" "$part" "${med:-FAIL}" "${p99:-?}"
  done
}
conc gpu_gpu_640 salsanext_fp16.engine segformer640_fp16.engine
conc dla_gpu_640 salsanext_dla.engine  segformer640_fp16.engine "--useDLACore=0"
conc dla_gpu_480 salsanext_dla.engine  segformer480_fp16.engine "--useDLACore=0"
