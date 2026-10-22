# Week 3 (Oct 19–26, re-planned): C++/TensorRT runtime on the Jetson

Exit criterion: C++/TensorRT nodes run on the Jetson; README latency table filled with measured numbers; the
planner consumes the contract. The Jetson is reached through its own Claude agent with self-contained bundles
(`scripts/make_jetson_bundle.sh`); every number below comes from `docs/measurements/`.

## C++ core and nodes
- [x] ROS-free C++ core; identical to the Python reference in all 9 stages (`cpp_parity.json`)
- [x] Four ROS 2 components in one multi-threaded container: segmentation_fusion, bev_tracker, health_monitor,
      planner_consumer; params generated from the repo's single sources of truth (`gen_ros_params.py`)
- [x] Lidar-path CPU stages reworked after run 1 (commit 17d4cba; on-device effect measured in run 2)

## Bring-up on the board
- [x] Board identified: Orin NX 8 GB, JetPack 6.0, TensorRT 8.6.2 (`jetson_info.txt`); toy models pass (`jetson_trt_toy.txt`)
- [x] Both packages build on the board: `perception_msgs` (`jetson_build_msgs.txt`), then the nodes with one
      compile fix (`jetson_replay_run1.txt`)
- [x] Run 1: stack runs end to end without crashes, but the lidar path takes 150 ms per scan against a 100 ms
      period; output ~720 ms old, health SAFE_STOP throughout (`jetson_replay_run1.txt`)
- [x] Receive delay split: 123 ms sensor lag recorded in the bag + ~447 ms queueing (`clip_00001_s100-400_lidar_lag.json`)

## Models under budget (ADR-0010)
- [x] Latency sweep: FP16, INT8 speed bound, DLA, 480×768, concurrency (`jetson_latency_sweep.txt`)
- [x] SegFormer 480×768 FP16: 20.0 ms, −1.7 test mIoU (`camera_onnx_480x768_test.json`)
- [x] DLA rejected (3× slower), SegFormer INT8 rejected (10 % faster)
- [ ] SalsaNext INT8: round 1 failed (best 34.80 vs 41.44 FP32, bar is ≤ 2 mIoU loss); round 2 (layer
      sensitivity → worst layers kept in FP16, other percentiles) running. Last round: INT8 closes after it.

## Real-time behaviour
- [x] Sensor queues hold one message, newest wins (ADR-0011); odometry in its own callback group
- [x] Planner watchdog: stop when grids stop arriving (contract rule 8; found while designing the H-6 test)
- [ ] Run 2 (bundle 6): does the lidar path keep up with 10 Hz? Output age against the ADR-0005 bound

## Exit criterion
- [ ] README latency table filled from run 2
- [ ] Planner consumes the contract: share of decisions with cmd_v > 0, and why the others stopped
- [ ] Jetson outputs vs the Python reference on the same clip (`compare_jetson_outputs.py`)
- [ ] Fault injection H-6: lidar dropped for 3 s → FAILED and SAFE_STOP within ~250–350 ms, then recovery

## Owner decisions
- [ ] ADR-0005: output-age bound 150 → 250 ms (the deployed params already use 250)
- [ ] Optional: private GitHub repo (the Jetson would need a deploy key or token)
