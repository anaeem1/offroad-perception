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
    Shared NFS volume, ~95 % full; this project uses ~150 GB. (`/home` was 99 % full until it was expanded to 10 TB ~Oct 18.)
- **Jetson (deployment target):** ROS 2 runtime, TensorRT engines, C++ nodes, latency measurement.
  - **Orin NX 8 GB** (not an Orin Nano, as first assumed), L4T R36.3.0 = **JetPack 6.0**, TensorRT 8.6.2,
    CUDA 12.2, **ROS 2 Humble already installed** (colcon present), Docker present. Power mode
    MAXN on Oct 17. The board also runs other people's work (Isaac ROS, nomad_node), so stop it before timing.
    **Internet since ~Oct 18** (it was offline before). Files still move by tarball (VS Code download →
    `~/parklabs/CAT` on the Jetson). A Claude Code session runs there: hand it self-contained prompts with
    a fixed reply format. Its clock was at 1970, so check NTP sync first, and run `conda deactivate`
    before any ROS 2 work. See `docs/jetson_setup.md`.
  - ONNX ships at **opset 17** (ADR-0006).
  - TensorRT engines must be built **on the Jetson**, not on the server.
  - Keep exported ONNX opsets compatible with the Jetson's TensorRT version.

## Timeline
| Week | Dates | Exit criterion |
|---|---|---|
| 1 | Oct 5–11 | Both seg models export to ONNX; lidar-on-camera projection overlay verified |
| 2 | Oct 12–18 → **Oct 18–21** | Fusion → BEV grid + tracking run on replayed bag; per-class IoU and ID-switch counts recorded |
| 3 | Oct 19–25 → **Oct 22–26** | C++/TensorRT nodes on Jetson; README latency table filled with measured numbers; planner consumes the contract |
| 4 | Oct 26–29 | ODD, hazard analysis/FMEA, ADRs complete; README reads as an architecture document; demo video |

**Re-plan (Oct 18):** week 2 started a week late. The owner chose **no scope cuts** and extra hours instead.
Week 2 is a Python reference pipeline on the server (also the golden reference for C++ parity tests);
week 3 is the C++/TensorRT runtime on the Jetson, including the C++ tracker, the planner consumer, and
bringing both models under budget. The Oct 26 feature stop is unchanged.

## Current state
*Updated 2026-10-22.*
- **Weeks 1–2 complete.** Week 2 (00001 test): wedge mIoU lidar 30.9 → fused 34.5 (≥10k-pt classes 53.7 → 59.1);
  lethal cells 88.8 % lethal / 1.7 % free; tracking MOTA 0.59, IDF1 0.56, 41 ID switches. C++ core matches the
  Python reference in all 9 stages (`cpp_parity.json`).
- **Week 3 (checklist: `docs/week3_checklist.md`).** Run 1 on the Jetson (`jetson_replay_run1.txt`): builds and
  runs, but the lidar path took 150 ms per scan; 203 of 319 scans, age ~720 ms, SAFE_STOP throughout. Diagnosis:
  receive delay 570 ms = 123 ms sensor lag recorded in the bag + ~447 ms queueing (subscription depth 5).
  Fixes since: ADR-0010 (SegFormer 480×768 FP16 20 ms; SalsaNext FP16 49 ms, a recorded overrun of its 20 ms
  budget), CPU stage rework, ADR-0011 (sensor queues depth 1, newest wins; odometry in its own callback group),
  planner watchdog (contract rule 8: stop when grids stop; found while designing the H-6 test).
- **INT8 lidar closed:** 10 calibration variants, best 38.72 vs 41.44 FP32 (bar ≤ 2 mIoU loss), vehicle IoU
  22.0 → 15.2 (ADR-0010). DLA and SegFormer INT8 rejected too. No model lever left; accept the overrun.
- **Waiting on the Jetson:** `CAT/jetson_bundle6.tar.gz` (commit b3d93f0; replaces bundle 5, which was never
  run): run 2 with the deployed config, timing of the best INT8 engine, run 3 = lidar dropped for 3 s (H-6).
  Results come back as a tarball. Then: `split_receive_delay.py`, `compare_jetson_outputs.py`, README latency
  table, `jetson_replay_run2.txt`, H-6 evidence.
- Records corrected 2026-10-22 after the board agent's review: the latency sweep ran with nomad_node and Isaac
  ROS still running (single-model times still match run 1); its concurrent rows are two processes, not the
  runtime's one-process/two-stream setup.
- **Open / needs a decision:** ADR-0005 output-age bound 150 → 250 ms (the deployed params already use 250;
  the contract text still says 150). Optional private GitHub repo (the Jetson would need a deploy key).
- Week 4 (Oct 26–29): ODD and hazard analysis are drafted (update with run 2/3 evidence); README as an
  architecture document; demo video. Feature stop Oct 26.
- Evaluation notes: RELLIS-3D splits are by sequence (00004 is all training data; use 00001); mIoU over
  "present" classes is dominated by tiny-support classes, so report the ≥10k-point mean alongside.

## Working conventions
- Python for training and evaluation; C++ for the runtime tracker and inference nodes.
- Small, reviewable commits with descriptive messages.
- Keep `docs/week1_checklist.md` (and later weekly checklists) updated as tasks complete.
- Measurements go in `docs/measurements/*.json`, written by the script that measured them.
  Quote numbers from there, never from memory.
- At the end of each session, update the "Current state" section above.
