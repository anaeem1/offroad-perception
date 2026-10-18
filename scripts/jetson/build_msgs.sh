#!/usr/bin/env bash
# Run ON the Jetson, outside conda. Builds the perception_msgs contract package with colcon and prints
# every message definition via `ros2 interface show`.
# Usage: scripts/jetson/build_msgs.sh [workspace_dir]   (default: the workspace containing this script)
set -eo pipefail  # no -u: ROS setup scripts read unset variables
if [ -n "${CONDA_PREFIX:-}" ]; then
  echo "Inside conda ($CONDA_PREFIX). Run 'conda deactivate' until (base) disappears, then retry."; exit 1
fi
# CONDA_PREFIX can be unset while conda's bin is still on PATH; CMake would then cache conda's Python.
if [ "$(command -v python3)" != /usr/bin/python3 ]; then
  echo "python3 resolves to $(command -v python3); ROS 2 Humble needs /usr/bin/python3. Fix PATH and retry."; exit 1
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
CACHE="build/perception_msgs/CMakeCache.txt"
if [ -f "$CACHE" ] && grep -q '^Python3_EXECUTABLE' "$CACHE" && ! grep -q '^Python3_EXECUTABLE:[A-Z]*=/usr/bin/python3$' "$CACHE"; then
  echo "Stale CMake cache in $WS/build uses another Python. Remove $WS/build and $WS/install, then retry."; exit 1
fi
colcon build --packages-select perception_msgs
source install/setup.bash
for m in TraversabilityGrid Obstacle ObstacleArray SensorStatus PerceptionHealth; do
  echo; echo "== perception_msgs/msg/$m"; ros2 interface show "perception_msgs/msg/$m"
done
