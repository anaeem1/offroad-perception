# Week 1 (Oct 5–11): foundation + contract

## Jetson setup (Orin NX 8 GB; offline, files moved by tarball)
- [x] Confirm JetPack version (`cat /etc/nv_tegra_release`) → pick matching ROS 2 distro
      (L4T R36.3.0 = JetPack 6.0, TensorRT 8.6.2 → Humble; `docs/measurements/jetson_info.txt`)
- [x] Install ROS 2, build `perception_msgs` with colcon, verify `ros2 interface show`
      (Humble was preinstalled; all five messages resolve; `docs/measurements/jetson_build_msgs.txt`)
- [x] Verify TensorRT + trtexec work with a toy ONNX model
      (all four toy models PASS at opset 16 and 17 → ship opset 17, ADR-0006;
      `docs/measurements/jetson_trt_toy.txt`)

## Data
- [x] Download: Ouster SemanticKITTI format + annotations, full images + annotations, calibration files
      (`scripts/download_rellis.sh`; inventory in `data/rellis3d/INVENTORY.tsv`)
- [x] Download at least one raw ROS bag sequence for replay (00004, full-stack merged, 206 s)
- [x] Convert bag → ROS 2 with `scripts/convert_bags.sh`; confirm topic names, fill `config/sensors.yaml`
      (`docs/measurements/bag_00004_inspection.json`)
- [x] Load extrinsics; sanity-check by projecting lidar points onto one camera image
      (`docs/figures/projection_overlay.jpg`; label agreement vs. extrinsic offset in
      `docs/measurements/projection_check.json`; 0.5° pitch residual noted)

## Models (baseline, not SOTA)
- [x] Camera: fine-tune a small SegFormer on RELLIS-3D images; record mIoU
      (B0 640×1024: **42.7** test mIoU; ADR-0004)
- [x] Lidar: pick a range-image segmentation net (TensorRT-friendly); train or adapt; record mIoU
      (released SalsaNext: **41.79** CNN-only, reproduces the published number; ADR-0003)
- [x] Export both to ONNX (opset 16 and 17, parity < 1e-5; `docs/measurements/onnx_*.json`)

## Architecture
- [x] Review interface_contract.md; resolve or defer the open questions (ADR-0005 proposed, needs a decision)
- [x] Write ADR-0002: late vs. early fusion
- [x] Write the RELLIS class → traversability cost table

**Exit criterion:** both models export to ONNX, and the lidar-on-camera projection overlay looks right.
**Status:** met and complete (last item closed Oct 17). Extra: first real-model latency on the Jetson,
`docs/measurements/jetson_trt_models.txt`; both models are over budget.
