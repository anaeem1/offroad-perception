# Perception Interface Contract v0.2

## Frames
- `base_link`: machine body frame (REP-103: x forward, y left, z up). All outputs are in this frame.
- Sensor frames as recorded in RELLIS-3D: lidar `ouster1/os1_lidar`, camera `pylon_camera`.
  `base_link` to both is available from `/tf_static`. The camera-lidar extrinsic comes from the RELLIS-3D
  calibration files (per sequence), which agree with `/tf_static` to 0.11° / 3.6 cm.

## Timing
- `header.stamp` is the **measurement time** of the source lidar scan, never the publish time. For the
  Ouster driver used here, that is the start of the 100 ms sweep (measured; per-point `t` spans the sweep).
- Camera frames are matched to the **time the lidar swept the camera's field of view** (t_wedge: median
  measurement time of the lidar points that project into the image), within ±50 ms. Frames outside the
  window are dropped and counted as camera DEGRADED. *Clarified 2026-10-18:* v0.1 said "within ±50 ms of
  the lidar scan", i.e. of `header.stamp`. But the sweep starts behind the machine, so the lidar passes
  the camera's view ~55–60 ms after the stamp. Measured on bag 00001: the best-matching frame is 6–10 ms
  (median) from t_wedge and 100 % of scans have one within the window, but **0 %** are within ±50 ms of
  the stamp. A stamp-based rule would silently fuse ~100 ms stale images
  (`docs/measurements/camera_matching_00001.json`). The sensors are not hardware synchronized, so the
  offset changes between recordings. Each pair is motion-compensated with odometry instead of assuming a
  constant offset.
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
6. **Grid memory (v0.2, ADR-0008):** cells the current sweep cannot see (notably the lidar's ~3 m blind
   zone) may carry *static* evidence up to 5 s older than `header.stamp`, with confidence scaled down by age.
   `source_flags` bit 4 marks that remembered evidence is present. People and vehicles are never remembered.
   Cells with no observation in the last 5 s are UNKNOWN.
7. The camera covers only a 37.7° forward wedge (ADR-0002). In `MODE_CAMERA_ONLY`, every cell outside
   that wedge is UNKNOWN (255). The mode means a narrow field of view, not just reduced range.

## Versioning
Breaking changes bump the major version and get an ADR. Additive fields bump the minor version.
- v0.2 (2026-10-18): grid memory semantics and `source_flags` bit 4 (additive, ADR-0008); camera matching
  rule clarified (match to the lidar's sweep of the camera FOV).

## Open questions
| Question | Status |
|---|---|
| Grid extent and resolution | **Resolved:** 40 m × 40 m @ 0.2 m (ADR-0007), with 5 s static memory (ADR-0008). |
| RELLIS-3D class → traversability cost | **Resolved (v0.1):** `config/class_costs.yaml`, explained in `docs/traversability_cost_table.md`. Costs are judgment-based and get revisited with week 2 data. |
| Output age guarantee | **Open:** ADR-0005 (proposed), needs an owner decision. |
| Generic obstacles outside the camera wedge | **Resolved:** height-above-ground check raises cells to lethal (ADR-0003, ADR-0007). |
| ~0.5° pitch residual in the camera-lidar extrinsic | **Resolved:** −0.5° correction adopted; +2.2 wedge mIoU on held-out test (`docs/measurements/fusion_00001.json`). |
| Camera debayering (`bayer_rggb8`) | **Resolved:** OpenCV `BayerBG2RGB` (mean error 3 vs the dataset JPEGs; `BayerRG2RGB` gives 22). |
