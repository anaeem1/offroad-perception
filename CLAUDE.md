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

**Re-plan (Oct 24, ADR-0012):** owner decision: finish on the server first (robustness study, demo video,
README as an architecture document); the second Jetson run moves after the write-up. The deadline stands.

## Current state
*Updated 2026-10-24.*
- **Weeks 1–2 complete; week 3 per ADR-0012.** Held-out 00001 test: wedge mIoU 30.9 → 34.5 fused; lethal cells
  88.8 % lethal / 1.7 % free; MOTA 0.59, IDF1 0.56, 41 IDSW; contract-following planner (offline port,
  `pipeline/planner.py`) moves in 84 % of scans. C++ core = Python reference in all 9 stages.
- **Robustness study done** (`robustness_00001_test.json`, `docs/figures/robustness.png`, 18 runs via
  `scripts/slurm/robustness.sbatch`, faults in `pipeline/faults.py`): lidar faults raise lethal-reported-free
  1.67 % → 3.14 % (50 % returns lost) / 4.60 % (32 rings) while the planner moves as often and the health
  density floor (0.2) fires only at ≥ 75 % loss → hazard H-14 (RPN 60). Camera faults only remove the camera's
  in-view gain (4° drift: 34.5 → 30.9 = lidar-only); grid and tracking unchanged.
- **Demo:** `<data root>/videos/demo_00001_test.mp4` (105 s, 1080p, 33 MB; not in git), GIF + still in
  `docs/figures/`. Render data: `<data root>/runs/replay_00001_demo/render/`.
- **README rewritten** as the architecture document (at a glance, architecture, contract, results, robustness,
  safety case, deployment status, findings, decision log, limitations, reproduce).
- **Server env fix:** `env/activate.sh` puts cuDNN 9 on LD_LIBRARY_PATH; before Oct 24 every replay ran ORT on
  the CPU. Full test-block replay now ~5 min on GPU with `--default-only`.
- **Jetson (deferred, ADR-0012):** run 1 measured (150 ms/scan, SAFE_STOP held). `CAT/jetson_bundle6.tar.gz`
  (commit b3d93f0) is ready: run 2 + INT8 timing + lidar-dropout test (H-6). INT8 lidar closed (ADR-0010).
- **ADR-0005 accepted (2026-10-24):** output age ≤ 250 ms from the sweep-start stamp; contract v1.0,
  `perception_msgs` 1.0.0 (no message definition changed).
- **Public repo:** github.com/anaeem1/offroad-perception (pushed 2026-10-25). Push from this server with the
  repo-only deploy key `~/.ssh/offroad_perception_deploy` (set as `core.sshCommand` in the repo config):
  `git push`. Owner chose to publish everything as is (server paths, CLAUDE.md, author email).
- **Open / needs the owner:** YouTube link for the demo video (owner uploads `CAT/demo_00001_test.mp4`,
  unlisted), then it goes into the README under the GIF.
- **Next:** hosting + links in the README; week 4 polish (Oct 26–29); Jetson run 2 when the owner has board time.
- Evaluation notes: RELLIS-3D splits are by sequence (00004 is all training data; use 00001); mIoU over
  "present" classes is dominated by tiny-support classes, so report the ≥10k-point mean alongside.

## Working conventions
- Python for training and evaluation; C++ for the runtime tracker and inference nodes.
- Small, reviewable commits with descriptive messages.
- Keep `docs/week1_checklist.md` (and later weekly checklists) updated as tasks complete.
- Measurements go in `docs/measurements/*.json`, written by the script that measured them.
  Quote numbers from there, never from memory.
- At the end of each session, update the "Current state" section above.
