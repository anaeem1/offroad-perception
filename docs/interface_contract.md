# Perception Interface Contract v0.1

## Frames
- `base_link`: machine body frame (REP-103: x forward, y left, z up). All outputs are in this frame.
- Sensor frames as recorded in RELLIS-3D: lidar `ouster1/os1_lidar`, camera `pylon_camera`.
  `base_link` to both is available from `/tf_static`. The camera-lidar extrinsic comes from the RELLIS-3D
  calibration files (per sequence), which agree with `/tf_static` to 0.11° / 3.6 cm.

## Timing
- `header.stamp` is the **measurement time** of the source lidar scan, never the publish time. For the
  Ouster driver used here, that is the start of the 100 ms sweep (measured; per-point `t` spans the sweep).
- Camera frames are matched to lidar scans within ±50 ms; frames outside the window are dropped and counted
  as camera DEGRADED. Measured on bag 00004: camera stamps lead lidar stamps by a median 22.8 ms, and
  99.9 % of scans have a camera frame inside the window.
- Consumers may assume outputs are no older than 150 ms when `/perception/health.mode == NOMINAL`.
  **Under review (ADR-0005, proposed):** lidar scans arrive a median 127 ms after their stamp, so this
  bound cannot hold. The proposed bound is 250 ms.

## Semantics consumers must respect
1. `cost == 255` means UNKNOWN. It must never be treated as free space.
2. Obstacles are gated on `existence_probability`, not on presence in the array.
3. `STATE_COASTING` tracks are predictions only; consumers should inflate margins.
4. `MODE_SAFE_STOP` overrides all other outputs.
5. `semantic_class` fields carry **RELLIS-3D ontology ids** (0–34, see `config/class_costs.yaml`),
   never a model's internal class index. The lidar and camera models have different class sets.
6. The camera covers only a 37.7° forward wedge (ADR-0002). In `MODE_CAMERA_ONLY`, every cell outside
   that wedge is UNKNOWN (255). The mode means a narrow field of view, not just reduced range.

## Versioning
Breaking changes bump the major version and get an ADR. Additive fields bump the minor version.

## Open questions
| Question | Status |
|---|---|
| Grid extent and resolution (initial 40 m × 40 m, 0.2 m) | **Deferred to week 2, with a measurement:** fraction of 0.2 m cells observed per range band in a single scan vs. with odometry accumulation. Ground-ring spacing grows with range, so 0.2 m may be mostly UNKNOWN beyond ~15 m without accumulation. |
| RELLIS-3D class → traversability cost | **Resolved (v0.1):** `config/class_costs.yaml`, explained in `docs/traversability_cost_table.md`. Costs are judgment-based and get revisited with week 2 data. |
| Output age guarantee | **Open:** ADR-0005 (proposed), needs an owner decision. |
| Generic obstacles outside the camera wedge | **Requirement for week 2:** the lidar model cannot output `object` (ADR-0003), so the BEV stage needs a height-above-ground check. |
| ~0.5° pitch residual in the camera-lidar extrinsic | **Week 2 A/B:** fused IoU with and without a −0.5° correction (`docs/measurements/projection_check.json`). |
| Camera debayering (`bayer_rggb8`) | **Week 3 check:** the runtime debayer must reproduce the training images' colour order (compare a bag frame to its dataset JPEG). |
