# Jetson setup (Orin NX 8 GB)

## Identified stack (2026-10-09, `docs/measurements/jetson_info.txt`)
| Item | Value |
|---|---|
| Module | Jetson Orin NX (7.4 GiB RAM → 8 GB variant) on the reference carrier |
| L4T / JetPack | R36.3.0 → **JetPack 6.0** (the `nvidia-jetpack` meta-package is not installed, but the components below are) |
| OS | Ubuntu 22.04.4, aarch64 |
| TensorRT / CUDA / cuDNN | **8.6.2** / 12.2 / 8.9.4 (`trtexec` present; no Python bindings) |
| ROS 2 | **Humble**, already installed in `/opt/ros/humble` |
| Power mode | 20W (mode 3) at the time of measurement. Every latency number records its power mode. |
| Network | None at first (so the clock was at 1970 and copied files looked "56 years in the future"); internet since ~Oct 18. |
| Other | Docker 28.3; 166 GB free disk |

**ONNX opset: 17** (ADR-0006). Both 16 and 17 parse and build on TensorRT 8.6.2
(`docs/measurements/jetson_trt_toy.txt`), and 17 gives TensorRT its native LayerNorm layer.

## Working on this board
- **Check the clock first:** `timedatectl` should show it synchronized (enable with
  `sudo timedatectl set-ntp true`). A wrong clock makes `make`/colcon warn about clock skew,
  makes bag and log timestamps meaningless, and breaks TLS and apt.
- **Leave conda before any ROS 2 work:** `conda deactivate` until the prompt loses `(base)`.
  Conda's Python shadows the system Python 3.10 that ROS 2 Humble and colcon need.
- Files move from the server via a tarball (VS Code download → `~/parklabs/CAT` on the Jetson).
  Extract with `tar xmzf` so file times are taken from the Jetson's clock.

## 1. Identify the stack ✔
`scripts/jetson/jetson_info.sh`. Done; see the table above.

## 2. Verify TensorRT ✔ (toy models); real models optional early
`scripts/jetson/check_trt.sh <dir>` builds an FP16 engine for every `*.onnx` in `<dir>` and reports
median GPU time. The toy models only checked op coverage. Running the real models gives an early read
on the 30 ms / 20 ms budgets (formally a week 3 measurement).

## 3. Build the contract package
`scripts/jetson/build_msgs.sh` (shipped in the bundle next to `ws/src/perception_msgs`): sources
Humble, runs `colcon build --packages-select perception_msgs`, then `ros2 interface show` for every
message. It refuses to run inside conda. If `colcon` is missing, install
`python3-colcon-common-extensions`.

## Rules
- TensorRT engines are built on the Jetson only. The server never ships `.engine` files.
- Latency numbers in the README come from this device, with the power mode recorded next to them.
