# ADR-0002: Late (decision-level) fusion of camera and lidar semantics
- **Status:** accepted
- **Date:** 2026-10-05
## Context
- Both modalities have semantic labels in RELLIS-3D, but there is no paired multi-modal
  training setup and no budget for one (four weeks, off-the-shelf models preferred).
- The camera sees a narrow forward wedge. From the RELLIS-3D intrinsics (fx = 2813.6 px, 1920 px wide)
  its horizontal FOV is **37.7°**, i.e. **10.5 %** of the lidar's 360° azimuth; vertical FOV is 24.1°.
  Most of the BEV grid will only ever have lidar evidence.
- The contract defines `MODE_LIDAR_ONLY` and `MODE_CAMERA_ONLY`, so each modality must be able to
  produce output on its own when the other fails.
- Target is an Orin NX with a ~100 ms end-to-end budget (camera seg 30 ms, lidar seg 20 ms).
## Options considered
1. **Early / raw fusion**: paint RGB onto points and train one lidar network on the joint input.
   Needs retraining on paired data. The joint input exists for only ~10 % of points, so the network
   sees two input distributions. A camera fault silently corrupts the lidar path.
2. **Feature-level fusion** (BEVFusion-style): highest accuracy ceiling, but needs heavy training
   and custom ops that are hard to fit in TensorRT on an Orin NX. A research detour, out of scope.
3. **Sequential** (PointPainting): camera seg output is painted onto points before lidar seg.
   Puts camera latency and camera failures in series with the lidar path.
4. **Late / decision-level**: each modality segments independently. Lidar points that project into
   the image take a fused class distribution; points outside the image keep the lidar result.
## Decision
Option 4. Lidar is the primary 360° source. The camera refines classes inside its 37.7° wedge.
`source_flags` on each grid cell records which modalities contributed.
## Consequences
- Easier: the models are trained, evaluated, exported and swapped independently. Degraded modes fall
  out naturally, since each branch already runs alone. Per-modality and fused IoU can be measured
  separately, which shows what fusion actually buys.
- Harder: no cross-modal feature learning, so the accuracy ceiling is lower than option 2.
  Fusion quality depends on extrinsic calibration and time alignment. The ±50 ms match window means
  0.25 m of motion at 5 m/s, so projection either motion-compensates or bounds speed.
- Contract impact: `MODE_CAMERA_ONLY` means a ~38° forward wedge of perception, not just "reduced
  range". The interface contract must say so (see interface_contract.md).
- Monitor: projection alignment (overlay check per sequence; extrinsics differ between RELLIS-3D
  sequences), camera-lidar time offset, and disagreement rate between modalities inside the wedge.
## Evidence
- FOV numbers: computed from `Rellis-3D/*/camera_info.txt` (identical across sequences).
- To be added in week 2: lidar-only vs camera-only vs fused per-class IoU inside the camera wedge.
