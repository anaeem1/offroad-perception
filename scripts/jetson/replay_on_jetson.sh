#!/usr/bin/env bash
# Run ON the Jetson, outside conda: build the stack, replay a bag clip through it, record the contract
# outputs, and summarize per-stage latency.
# Usage: scripts/jetson/replay_on_jetson.sh <workspace_root> <clip_bag_dir> <onnx_dir> <out_dir> [seq]
#   workspace_root: dir containing src/perception_msgs and src/offroad_perception
set -eo pipefail
set -m  # job control: background jobs get their own process group and do not ignore SIGINT
WS="$(realpath "${1:?workspace}")"; CLIP="$(realpath "${2:?clip bag dir}")"; ONNX="$(realpath "${3:?onnx dir}")"
OUT="$(realpath -m "${4:?output dir}")"; SEQ="${5:-00001}"
TRTEXEC=/usr/src/tensorrt/bin/trtexec
[ -n "${CONDA_PREFIX:-}" ] && { echo "Inside conda: run 'conda deactivate' first."; exit 1; }
[ "$(command -v python3)" = /usr/bin/python3 ] || { echo "python3 is $(command -v python3); need /usr/bin/python3"; exit 1; }
[ "$(date +%Y)" -ge 2024 ] || { echo "System clock is wrong: $(date)"; exit 1; }
mkdir -p "$OUT"
source /opt/ros/humble/setup.bash

echo "== build"
cd "$WS"
colcon build --packages-select perception_msgs offroad_perception --cmake-args -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -20
source install/setup.bash

echo "== engines"
ENG="$OUT/engines"; mkdir -p "$ENG"
build_engine() {  # onnx engine [extra trtexec flags]
  local onnx="$1" eng="$2"; shift 2
  [ -f "$ENG/$eng" ] && { echo "have $eng"; return; }
  [ -f "$ONNX/$onnx" ] || { echo "skip $eng ($onnx not in bundle)"; return; }
  "$TRTEXEC" --onnx="$ONNX/$onnx" --fp16 "$@" --saveEngine="$ENG/$eng" > "$ENG/$eng.build.log" 2>&1 \
    && echo "built $eng" || { echo "FAILED $eng (see $ENG/$eng.build.log)"; exit 1; }
}
build_engine salsanext_rellis_op17_int8qdq.onnx salsanext_int8.engine --int8   # QDQ model: explicit INT8
build_engine salsanext_rellis_op17.onnx salsanext_fp16.engine                  # fallback
build_engine segformer_b0_480x768_op17.onnx segformer480_fp16.engine
LIDAR_ENGINE="${LIDAR_ENGINE:-salsanext_fp16.engine}"; CAMERA_ENGINE="${CAMERA_ENGINE:-segformer480_fp16.engine}"
echo "using lidar=$LIDAR_ENGINE camera=$CAMERA_ENGINE" | tee "$OUT/engines_used.txt"

echo "== replay"
nvpmodel -q 2>/dev/null | head -1 | tee "$OUT/power_mode.txt"
DUR=$(ros2 bag info "$CLIP" | awk '/Duration/ {print int($2)+1}')
(tegrastats --interval 500 > "$OUT/tegrastats.log" 2>&1 &) ; TS=$!
ros2 launch offroad_perception replay.launch.py bag:="$CLIP" seq:="$SEQ" engines:="$ENG" logs:="$OUT" \
  lidar_engine_file:="$LIDAR_ENGINE" camera_engine_file:="$CAMERA_ENGINE" > "$OUT/launch.log" 2>&1 &
LAUNCH=$!
sleep 6
ros2 bag record -s mcap -o "$OUT/outputs" /perception/traversability /perception/obstacles /perception/health /planner/cmd_vel \
  > "$OUT/record.log" 2>&1 &
REC=$!
sleep $((DUR + 12))
kill -INT -- -$REC 2>/dev/null || true; sleep 2      # whole process group (recorder)
kill -INT -- -$LAUNCH 2>/dev/null || true; sleep 6   # whole process group (launch + container)
pgrep -f "node:=perception" >/dev/null && { echo "container still up; sending TERM"; pkill -TERM -f "node:=perception"; } || true
pkill -f tegrastats 2>/dev/null || true
grep -iE "error|exception|fail" "$OUT/launch.log" | head -20 || true

echo "== latency summary (ms)"
python3 - "$OUT" <<'PY' | tee "$OUT/latency_summary.txt"
import csv, statistics, sys, os
out = sys.argv[1]
def q(v, p):
    v = sorted(v); return v[min(len(v) - 1, int(round(p / 100 * (len(v) - 1))))]
for name in ("segmentation_camera.csv", "segmentation_lidar.csv", "bev_tracker.csv", "planner.csv"):
    path = os.path.join(out, name)
    if not os.path.exists(path):
        print(f"{name}: missing"); continue
    rows = list(csv.DictReader(open(path)))
    print(f"\n{name}: {len(rows)} rows")
    for col in rows[0] if rows else []:
        if col in ("stamp", "reason"): continue
        try:
            v = [float(r[col]) for r in rows if r[col] not in ("", "nan")]
        except ValueError:
            continue
        if v:
            print(f"  {col:22s} median {statistics.median(v):9.2f}  p95 {q(v, 95):9.2f}  max {max(v):9.2f}")
PY
echo "outputs in $OUT"
