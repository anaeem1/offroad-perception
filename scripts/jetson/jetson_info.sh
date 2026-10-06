#!/usr/bin/env bash
# Run ON the Jetson. Prints what pins the deployment stack: JetPack -> TensorRT -> ROS 2 distro -> ONNX opset.
# Usage: scripts/jetson/jetson_info.sh | tee docs/measurements/jetson_info.txt
set -u
section() { echo; echo "== $*"; }
section "Model";      tr -d '\0' < /proc/device-tree/model 2>/dev/null; echo
section "L4T";        cat /etc/nv_tegra_release 2>/dev/null || echo "missing /etc/nv_tegra_release"
section "JetPack";    dpkg-query -W -f='${Package} ${Version}\n' nvidia-jetpack 2>/dev/null || echo "nvidia-jetpack meta-package not installed"
section "OS";         . /etc/os-release; echo "$PRETTY_NAME ($(uname -m)), kernel $(uname -r)"
section "TensorRT";   dpkg-query -W -f='${Package} ${Version}\n' 'libnvinfer[0-9]*' 2>/dev/null | grep -v -- '-dev' | head -3
                      python3 -c 'import tensorrt as t; print("python tensorrt", t.__version__)' 2>&1 | tail -1
section "trtexec";    ls -l /usr/src/tensorrt/bin/trtexec 2>&1
section "CUDA";       ls -d /usr/local/cuda-* 2>/dev/null; /usr/local/cuda/bin/nvcc --version 2>/dev/null | tail -1
section "cuDNN";      dpkg-query -W -f='${Package} ${Version}\n' 'libcudnn[0-9]*' 2>/dev/null | grep -v -- '-dev' | head -2
section "Power mode"; nvpmodel -q 2>/dev/null || echo "(run with sudo for nvpmodel)"
section "Memory";     free -h | head -2
section "Disk";       df -h / | tail -1
section "ROS 2";      ls /opt/ros 2>/dev/null || echo "no /opt/ros"
section "Docker";     docker --version 2>/dev/null || echo "no docker"
