# Perception Interface Contract v0.1

## Frames
- `base_link`: machine body frame (REP-103: x forward, y left, z up). All outputs are in this frame.
- `os1_lidar`, `pylon_camera`: sensor frames; extrinsics from RELLIS-3D calibration files, published via `/tf_static`.

## Timing
- `header.stamp` is the **measurement time** of the source lidar scan, never the publish time.
- Camera frames are matched to lidar scans within ±50 ms; frames outside the window are dropped and counted as camera DEGRADED.
- Consumers may assume outputs are no older than 150 ms when `/perception/health.mode == NOMINAL`.

## Semantics consumers must respect
1. `cost == 255` means UNKNOWN. It must never be treated as free space.
2. Obstacles are gated on `existence_probability`, not on presence in the array.
3. `STATE_COASTING` tracks are predictions only; consumers should inflate margins.
4. `MODE_SAFE_STOP` overrides all other outputs.

## Versioning
Breaking changes bump the major version and get an ADR. Additive fields bump the minor version.

## Open questions (resolve by end of week 2)
- Grid extent and resolution (initial: 40 m × 40 m, 0.2 m)
- Class grouping: map 20 RELLIS-3D classes → traversability cost table
