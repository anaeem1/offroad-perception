#!/usr/bin/env bash
# Package what the Jetson needs for a build + replay: the two ROS packages, the replay script, and the ONNX
# models of the deployment configuration (ADR-0010). Usage: scripts/make_jetson_bundle.sh <name> [out_dir]
set -euo pipefail
NAME="${1:?bundle name, e.g. jetson_bundle5}"; OUTDIR="${2:-$(cd .. && pwd)}"
: "${OFFROAD_DATA_ROOT:=/usace_share/Share_Abdullah/offroad_perception}"
ONNX="$OFFROAD_DATA_ROOT/onnx"
STAGE="$(mktemp -d)"; B="$STAGE/$NAME"
mkdir -p "$B/ws/src" "$B/onnx"
cp -r src/perception_msgs src/offroad_perception "$B/ws/src/"
cp scripts/jetson/replay_on_jetson.sh "$B/"
for f in salsanext_rellis_op17.onnx segformer_b0_480x768_op17.onnx; do  # ADR-0010 deployment models
  cp "$ONNX/$f" "$B/onnx/"
done
git rev-parse --short HEAD > "$B/COMMIT"
tar czf "$OUTDIR/$NAME.tar.gz" -C "$STAGE" "$NAME"
rm -rf "$STAGE"
ls -la "$OUTDIR/$NAME.tar.gz"
