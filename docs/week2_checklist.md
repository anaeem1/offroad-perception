# Week 2 (Oct 18–21, re-planned): fusion → BEV → tracking, Python reference on the server

Runs offline on the ROS 2 bags via `rosbags` (no ROS on the server). Metrics on held-out sequence 00001:
settings chosen on its val block (scans 1047–2318), reported on its test block (0–1046). This code is also
the golden reference for the C++ runtime (`docs/measurements/cpp_parity.json`: all 9 stages identical).

## Data plumbing
- [x] Align bag 00001 lidar scans with the labelled scans: exact xyz hash, all 2,319 matched (`alignment_00001.json`)
- [x] Debayer: OpenCV `BayerBG2RGB` (mean error 3 vs dataset JPEGs; `BayerRG2RGB` 22); remission = intensity/65535

## Fusion
- [x] Camera frame matched to the lidar's sweep of the camera FOV (6 ms median), not the scan stamp (contract fixed)
- [x] Per-pair odometry motion compensation (no measurable effect at 1.4 m/s; kept for higher speeds)
- [x] Late fusion in ontology space; log-linear, α 0.5 (chosen on val) — ADR-0002
- [x] Pitch-residual A/B: −0.5° correction adopted, +2.2 wedge mIoU on test (`fusion_00001.json`)

## BEV traversability grid
- [x] Extent/resolution from measured coverage (ADR-0007); 5 s static grid memory for the lidar blind zone (ADR-0008)
- [x] Grid from fused semantics + cost table + height check; 255 = UNKNOWN; ego filter for self-returns
- [x] Confidence and `source_flags` per the contract (v0.2 adds bit 4 = remembered)

## Clustering + tracking
- [x] Connected components of obstacle points (+ void points in height-check cells)
- [x] Kalman + Hungarian tracker in the odom frame; TENTATIVE / CONFIRMED / COASTING; existence probability
- [x] Pseudo-GT tracks from labels, with don't-care rules (ADR-0009); label "trails" found in RELLIS-3D

## Metrics (exit criterion)
- [x] Per-class IoU on 00001 test: points (lidar-only vs fused, all vs camera wedge) and BEV cells
- [x] Tracking: 41 ID switches, MOTA 0.59, IDF1 0.56 on 66 objects (`tracking_00001_test.json`)

**Exit criterion:** met on Oct 18. Fusion → BEV grid + tracking run on a replayed bag; per-class IoU and
ID-switch counts recorded.
