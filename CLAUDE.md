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
  - GPU / scheduler: <FILL: e.g. SLURM with H100 nodes, partition name, conda vs. container>
  - Data root: <FILL: path with ~40 GB+ free>
- **Jetson (deployment target):** ROS 2 runtime, TensorRT engines, C++ nodes, latency measurement.
  - Model / JetPack: <FILL> → this determines the ROS 2 distro and TensorRT version.
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
- Done: repo skeleton, `perception_msgs` contract v0.1, interface contract doc, ADR-0001, week 1 checklist.
- Not started: environment setup, data download, any model work.
- Open: Jetson model/JetPack; grid extent and resolution (initial 40 m × 40 m @ 0.2 m);
  RELLIS class → traversability cost table; ADR-0002 (late vs. early fusion).

## Working conventions
- Python for training and evaluation; C++ for the runtime tracker and inference nodes.
- Small, reviewable commits with descriptive messages.
- Keep `docs/week1_checklist.md` (and later weekly checklists) updated as tasks complete.
- At the end of each session, update the "Current state" section above.
