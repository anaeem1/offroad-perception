#!/usr/bin/env bash
# Convert RELLIS-3D ROS 1 bags to ROS 2 (mcap) without needing ROS 1 installed.
# Keeps only the topics the perception stack replays; stereo, Velodyne and raw Ouster packets are dropped.
# Usage: source env/activate.sh && scripts/convert_bags.sh <dir_with_ros1_bags> <output_dir>
set -euo pipefail
IN="${1:?input dir}"; OUT="${2:?output dir}"
command -v rosbags-convert >/dev/null || { echo "rosbags-convert not found: source env/activate.sh first"; exit 1; }
TOPICS=(
  /os1_cloud_node/points           # lidar input
  /os1_cloud_node/imu              # lidar IMU (deskew, if needed)
  /pylon_camera_node/image_raw     # camera input
  /pylon_camera_node/camera_info   # intrinsics + distortion
  /tf /tf_static                   # extrinsics, base_link
  /odometry/filtered               # ego-motion for the tracker
  /imu/data
  /vectornav/GPS /vectornav/Odom
)
mkdir -p "$OUT"
shopt -s nullglob
bags=("$IN"/*.bag)
[ ${#bags[@]} -gt 0 ] || { echo "no .bag files in $IN"; exit 1; }
for bag in "${bags[@]}"; do
  name="$(basename "$bag" .bag)"
  echo "Converting $name ..."
  rosbags-convert --src "$bag" --dst "$OUT/$name" --dst-storage mcap --include-topic "${TOPICS[@]}"
done
echo "Done. Inspect with: ros2 bag info $OUT/<name>  (no ROS 2 here: python scripts/bag_info.py $OUT/<name>)"
