# Week 1 (Oct 5–11): foundation + contract

## Jetson setup
- [ ] Confirm JetPack version (`cat /etc/nv_tegra_release`) → pick matching ROS 2 distro
- [ ] Install ROS 2, build `perception_msgs` with colcon, verify `ros2 interface show`
- [ ] Verify TensorRT + trtexec work with a toy ONNX model

## Data
- [ ] Download: Ouster SemanticKITTI format + annotations, full images + annotations, calibration files
- [ ] Download at least one raw ROS bag sequence for replay
- [ ] Convert bag → ROS 2 with `scripts/convert_bags.sh`; confirm topic names, fill `config/sensors.yaml`
- [ ] Load extrinsics; sanity-check by projecting lidar points onto one camera image (save the overlay: it goes in the README)

## Models (baseline, not SOTA)
- [ ] Camera: fine-tune a small SegFormer on RELLIS-3D images; record mIoU
- [ ] Lidar: pick a range-image segmentation net (TensorRT-friendly); train or adapt; record mIoU
- [ ] Export both to ONNX

## Architecture
- [ ] Review interface_contract.md; resolve or defer the open questions
- [ ] Write ADR-0002: late vs. early fusion
- [ ] Write the RELLIS class → traversability cost table

**Exit criterion:** both models export to ONNX, and the lidar-on-camera projection overlay looks right.
