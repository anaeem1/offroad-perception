# Jetson setup (Orin Nano)

The JetPack version on the Jetson fixes the TensorRT version, which fixes the ONNX opset we export at
and the ROS 2 distro we run. Nothing downstream should be pinned until step 1 is done.

## 1. Identify the stack
On the Jetson, from a clone of this repo:
```bash
scripts/jetson/jetson_info.sh | tee docs/measurements/jetson_info.txt
```
Read the L4T line and look it up below. Orin Nano supports JetPack 5.1.x and 6.x.

| L4T (`/etc/nv_tegra_release`) | JetPack | Ubuntu | TensorRT | CUDA | ROS 2 | ONNX opset |
|---|---|---|---|---|---|---|
| R35.x (35.2–35.6) | 5.1.x | 20.04 | 8.5.2 | 11.4 | Humble via container (no apt binaries for 20.04) | 16 (confirm with step 2) |
| R36.2 / R36.3 | 6.0 | 22.04 | 8.6.2 | 12.2 | Humble (apt) | 17 |
| R36.4.x | 6.1 / 6.2 | 22.04 | 10.3 | 12.6 | Humble (apt) | 17 |

If the board is on JetPack 5, upgrading to 6.2 is worth considering before week 3: native Humble
packages, TensorRT 10, and the Orin Nano "Super" power mode. That choice gets an ADR.

## 2. Verify TensorRT with toy models
On the server: `python scripts/export_toy_onnx.py --out data/onnx/toy` (inside `env/activate.sh`).
Copy `data/onnx/toy/` to the Jetson, then:
```bash
scripts/jetson/check_trt.sh <copied_toy_dir> | tee docs/measurements/jetson_trt_toy.txt
```
The toy models use the same ops as SegFormer (LayerNorm, GELU, attention, Resize) and SalsaNext
(BatchNorm, AvgPool, PixelShuffle), each with ArgMax output. The lowest opset that passes for both
is the one we export the real models at.

## 3. ROS 2 + the contract package
JetPack 6.x (Ubuntu 22.04):
```bash
# ROS 2 Humble apt repo: https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html
sudo apt install ros-humble-ros-base ros-dev-tools
source /opt/ros/humble/setup.bash
cd <repo> && colcon build --packages-select perception_msgs && source install/setup.bash
ros2 interface show perception_msgs/msg/TraversabilityGrid
```
JetPack 5.x: use a Humble container matched to the L4T release (e.g. from `dusty-nv/jetson-containers`),
mount the repo, and run the same `colcon build` / `ros2 interface show` inside it.

## Rules
- TensorRT engines are built on the Jetson only. The server never ships `.engine` files.
- Latency numbers in the README come from this device, with the power mode recorded next to them.
