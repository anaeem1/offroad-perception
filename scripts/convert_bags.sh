#!/usr/bin/env bash
# Convert RELLIS-3D ROS 1 bags to ROS 2 (mcap) without needing ROS 1 installed.
# Usage: ./convert_bags.sh <dir_with_ros1_bags> <output_dir>
set -euo pipefail
IN="${1:?input dir}"; OUT="${2:?output dir}"
pip install --quiet rosbags
mkdir -p "$OUT"
for bag in "$IN"/*.bag; do
  name="$(basename "$bag" .bag)"
  echo "Converting $name ..."
  rosbags-convert --src "$bag" --dst "$OUT/$name"
done
echo "Done. Inspect topics with: ros2 bag info $OUT/<name>"
