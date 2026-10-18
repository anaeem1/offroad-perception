# Week 2 (Oct 18–21, re-planned): fusion → BEV → tracking, Python reference on the server

Runs offline on the ROS 2 bags via `rosbags` (no ROS on the server). Metrics on held-out sequence
00001 only; 00004 is for bring-up. This code is also the golden reference for the week 3 C++ nodes.

## Data plumbing
- [ ] Align bag 00001 lidar scans with the labelled SemanticKITTI scans (verified by content, not index)
- [ ] Camera frames: debayer `bayer_rggb8` and confirm colour order against the dataset JPEGs

## Fusion
- [ ] Per-pair camera↔lidar time offset with odometry motion compensation (constant offset is wrong)
- [ ] Late fusion of class probabilities inside the camera wedge (ADR-0002); lidar-only outside
- [ ] Pitch-residual A/B: fused IoU with nominal vs −0.5° extrinsic

## BEV traversability grid
- [ ] Decide extent/resolution from the observed-cell fraction per range band (single scan vs accumulated)
- [ ] Grid from fused semantics + cost table + height-above-ground check (generic obstacles); 255 = UNKNOWN
- [ ] Confidence and `source_flags` per cell, per the contract

## Clustering + tracking
- [ ] Cluster non-traversable obstacle-class points → obstacle candidates
- [ ] Tracker: Kalman filter + Hungarian in an ego-motion-compensated frame; TENTATIVE/CONFIRMED/COASTING
- [ ] Pseudo-ground-truth tracks from labels + poses (RELLIS has no boxes, ADR-0001)

## Metrics (exit criterion)
- [ ] Per-class IoU on 00001: lidar-only vs fused (points), and BEV cell IoU
- [ ] Tracking: ID switches, MOTA/IDF1 against pseudo-GT; record method limits

**Exit criterion:** fusion → BEV grid + tracking run on a replayed bag; per-class IoU and ID-switch
counts recorded.
