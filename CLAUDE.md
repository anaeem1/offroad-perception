# CLAUDE.md — Off-Road Perception Stack

## Why this project exists
Portfolio centerpiece for an application to a Perception Architecture lead role in construction
autonomy (deadline **Oct 30, 2026**). The role is about owning an *end-to-end perception system*:
sensors → perception → fusion → tracking → mapping → safety, with output contracts, decision
records, and risk assessments as deliverables. The portfolio gap this fills: no visible system-level,
multi-sensor, edge-deployed perception work with architecture artifacts.

**The value is in integration, measurement, and documentation, not model accuracy.**
A simple, well-measured, well-documented system beats a SOTA model with no contract.

## Scope guardrails (read before proposing work)
- Prefer off-the-shelf or lightly fine-tuned models. Do not start research detours.
- Every non-trivial design choice gets an ADR in `docs/adr/` (use `0000-template.md`).
- Every performance claim must be backed by a measurement committed to the repo.
- If a task threatens the weekly exit criterion, cut scope and record the cut in an ADR.
- Hard stop on new features after **Oct 26**; the remaining days are for polish and the write-up.

## Dataset: RELLIS-3D (Texas A&M, off-road, Clearpath Warthog)
- Sensors: Ouster OS1-64 lidar, Basler RGB camera, Nerian stereo, Velodyne, GPS, IMU.
- Labels: **semantic only** (per-point lidar, per-pixel camera). **No 3D boxes.** See ADR-0001.
- Raw sensor bags are **ROS 1**. Convert with `scripts/convert_bags.sh` (uses `rosbags`, no ROS 1 needed).
- Downloads are linked from github.com/unmannedlab/RELLIS-3D (mostly Google Drive; `gdown` may help).
  If a link fails, report it instead of guessing alternative sources.
- Needed: Ouster SemanticKITTI format + annotations, full images + annotations, calibration files,
  and at least one raw bag sequence for replay.

## Architecture (decided)
Camera semantic seg (SegFormer) + lidar semantic seg (range-image CNN, TensorRT-friendly)
→ late fusion (project lidar points into the camera image)
→ BEV traversability grid + clustering of non-traversable points
→ multi-object tracker (C++, Kalman filter + Hungarian)
→ outputs on the contract in `src/perception_msgs/`, plus a health monitor with degraded modes.

## Contract rules (do not violate)
- All outputs are in `base_link` (REP-103). `header.stamp` = lidar measurement time, not publish time.
- `cost == 255` is UNKNOWN and must never be treated as free.
- Obstacles are gated on `existence_probability`; COASTING tracks are predictions only.
- `MODE_SAFE_STOP` overrides all other outputs.
- Breaking changes to `perception_msgs` require a major version bump and an ADR.

## Environments
- **Server (this machine):** dataset storage, training, ONNX export, offline evaluation.
  - GPU / scheduler: SLURM, partition `H100` (4 nodes × 1 H100 NVL 94 GB, 32 CPU, 280 GB RAM; QoS
    `h100_normal`: ≤3 GPUs/user, ≤3 days). Submit with `scripts/slurm/*.sbatch` from the repo root.
    Nodes 1, 3 and 4 were down Oct 5–7 after an unexpected reboot; they were back by Oct 7.
  - **Compute nodes have no internet.** Download data, pip packages and HF weights on the login node
    (1 CPU). Jobs run with `HF_HUB_OFFLINE=1`.
  - Env: conda prefix at `<data root>/env` (no Docker/Apptainer access). Always `source env/activate.sh`.
    Pins in `env/requirements-server.txt`, full lock in `env/requirements-server.lock.txt`.
    torch stays on 2.5.x on purpose (TorchScript ONNX exporter, opset 16/17).
  - Data root: `/usace_share/Share_Abdullah/offroad_perception` (repo symlink `data`, gitignored).
    Shared NFS volume, ~95 % full; this project uses ~150 GB. `/home` is 99 % full: keep data off it.
- **Jetson (deployment target):** ROS 2 runtime, TensorRT engines, C++ nodes, latency measurement.
  - **Orin NX 8 GB** (not an Orin Nano, as first assumed), L4T R36.3.0 = **JetPack 6.0**, TensorRT 8.6.2,
    CUDA 12.2, **ROS 2 Humble already installed**, Docker present. Power mode 20W when measured.
    **No network and no SSH:** files move by tarball (VS Code download → USB/scp). Its clock was at 1970,
    so set the date before building, and run `conda deactivate` before any ROS 2 work. See `docs/jetson_setup.md`.
  - ONNX ships at **opset 17** (ADR-0006).
  - TensorRT engines must be built **on the Jetson**, not on the server.
  - Keep exported ONNX opsets compatible with the Jetson's TensorRT version.

## Timeline
| Week | Dates | Exit criterion |
|---|---|---|
| 1 | Oct 5–11 | Both seg models export to ONNX; lidar-on-camera projection overlay verified |
| 2 | Oct 12–18 | Fusion → BEV grid + tracking run on replayed bag; per-class IoU and ID-switch counts recorded |
| 3 | Oct 19–25 | C++/TensorRT nodes on Jetson; README latency table filled with measured numbers; planner consumes the contract |
| 4 | Oct 26–29 | ODD, hazard analysis/FMEA, ADRs complete; README reads as an architecture document; demo video |

## Current state
*Updated 2026-10-09.*
- **Week 1 exit criterion met.** Both models export to ONNX; projection overlay verified; Jetson
  stack identified and TensorRT verified. Remaining: `perception_msgs` build on the Jetson
  (`CAT/jetson_bundle2.tar.gz`, which also carries both real models for an early latency read).
- Done:
  - Repo under git (own repo, branch `main`); env built; RELLIS-3D subset + bag 00004 downloaded.
  - Bag 00004 converted to ROS 2 mcap (topic-filtered, Humble-compatible). Timing, frames and TF
    measured; `config/sensors.yaml` filled.
  - Projection check: overlay in `docs/figures/`. Label agreement 74–89 %. Yaw is correct; pitch peaks
    0.5° below nominal in all sequences (not corrected; week 2 A/B).
  - Camera: SegFormer-B0 640×1024, test mIoU 42.7 (ADR-0004). Lidar: released SalsaNext, 41.79
    CNN-only / 42.41 with KNN; reproduces the published 40.2 under its own convention (ADR-0003).
  - ONNX: both models at opset 16 and 17 in `data/onnx/`, parity < 1e-5 vs PyTorch.
  - ADR-0002 (late fusion), ADR-0003, ADR-0004; cost table v0.1; contract reviewed.
- Not started: fusion, BEV grid, tracker (week 2); C++/TensorRT nodes (week 3).
- Open / needs a decision:
  - **ADR-0005 (proposed):** the 150 ms output-age bound can't hold (lidar arrives 127 ms after its
    sweep-start stamp). The proposal is 250 ms plus contract v1.0. Owner decision needed.
  - Grid extent/resolution: decide in week 2 from the observed-cell fraction per range band.
  - KNN on the Jetson: +4.0 barrier / +2.5 vehicle IoU vs. its latency; decide in week 3.
- **Evaluation trap:** RELLIS-3D splits are by sequence, and sequence 00004 is entirely in both models'
  training splits. IoU measured on its replay is training-set accuracy. Sequence 00001 is the only one
  with no training frames (`docs/measurements/split_by_sequence.json`). **Its bag is downloaded and
  converted (Oct 8):** `data/bags/ros2/20200213_trail_1`, 232 s. Use 00001 for week 2 IoU and 00004
  for bring-up and the demo.
- Camera-lidar stamp offset differs per recording (median −22.8 ms on 00004, −41.2 ms on 00001; no
  hardware sync), so week 2 fusion must motion-compensate each pair. A constant offset is wrong.
- Week 2 requirements already known: a height-above-ground obstacle check (the lidar model cannot
  output `object`); a debayer colour-order check for the `bayer_rggb8` images.

## Working conventions
- Python for training and evaluation; C++ for the runtime tracker and inference nodes.
- Small, reviewable commits with descriptive messages.
- Keep `docs/week1_checklist.md` (and later weekly checklists) updated as tasks complete.
- Measurements go in `docs/measurements/*.json`, written by the script that measured them.
  Quote numbers from there, never from memory.
- At the end of each session, update the "Current state" section above.
