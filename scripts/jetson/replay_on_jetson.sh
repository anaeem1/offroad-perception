#!/usr/bin/env bash
# Run ON the Jetson, outside conda: build the stack, replay a bag clip through it, record the contract
# outputs, and summarize per-stage latency.
# Usage: scripts/jetson/replay_on_jetson.sh <workspace_root> <clip_bag_dir> <onnx_dir> <out_dir> [seq]
#   workspace_root: dir containing src/perception_msgs and src/offroad_perception
set -eo pipefail
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
build_engine() {  # onnx engine
  [ -f "$ENG/$2" ] && { echo "have $2"; return; }
  "$TRTEXEC" --onnx="$ONNX/$1" --fp16 --saveEngine="$ENG/$2" > "$ENG/$2.build.log" 2>&1 && echo "built $2" || { echo "FAILED $2 (see $ENG/$2.build.log)"; exit 1; }
}
build_engine salsanext_rellis_op17.onnx salsanext_fp16.engine
build_engine segformer_b0_640x1024_op17.onnx segformer640_fp16.engine

echo "== replay"
nvpmodel -q 2>/dev/null | head -1 | tee "$OUT/power_mode.txt"
DUR=$(ros2 bag info "$CLIP" | awk '/Duration/ {print int($2)+1}')
(tegrastats --interval 500 > "$OUT/tegrastats.log" 2>&1 &) ; TS=$!
ros2 launch offroad_perception replay.launch.py bag:="$CLIP" seq:="$SEQ" engines:="$ENG" logs:="$OUT" > "$OUT/launch.log" 2>&1 &
LAUNCH=$!
sleep 6
ros2 bag record -s mcap -o "$OUT/outputs" /perception/traversability /perception/obstacles /perception/health /planner/cmd_vel \
  > "$OUT/record.log" 2>&1 &
REC=$!
sleep $((DUR + 12))
kill -INT $REC 2>/dev/null || true; sleep 2
kill -INT $LAUNCH 2>/dev/null || true; sleep 4
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
