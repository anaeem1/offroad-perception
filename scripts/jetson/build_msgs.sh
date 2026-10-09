#!/usr/bin/env bash
# Run ON the Jetson, outside conda. Builds the perception_msgs contract package with colcon and prints
# every message definition via `ros2 interface show`.
# Usage: scripts/jetson/build_msgs.sh [workspace_dir]   (default: the workspace containing this script)
set -eo pipefail  # no -u: ROS setup scripts read unset variables
if [ -n "${CONDA_PREFIX:-}" ]; then
  echo "Inside conda ($CONDA_PREFIX). Run 'conda deactivate' until (base) disappears, then retry."; exit 1
fi
if [ "$(date +%Y)" -lt 2024 ]; then
  echo "System clock is $(date). Set it first: sudo date -s \"YYYY-MM-DD HH:MM\""; exit 1
fi
source /opt/ros/humble/setup.bash
command -v colcon >/dev/null || { echo "colcon missing: install python3-colcon-common-extensions"; exit 1; }
HERE="$(cd "$(dirname "$0")" && pwd)"
WS="${1:-$(cd "$HERE/../.." && pwd)}"
[ -d "$WS/src/perception_msgs" ] || WS="$HERE/ws"  # bundle layout: ws/ next to this script
cd "$WS"
colcon build --packages-select perception_msgs
source install/setup.bash
for m in TraversabilityGrid Obstacle ObstacleArray SensorStatus PerceptionHealth; do
  echo; echo "== perception_msgs/msg/$m"; ros2 interface show "perception_msgs/msg/$m"
done
