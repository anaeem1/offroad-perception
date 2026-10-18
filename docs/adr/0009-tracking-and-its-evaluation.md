# ADR-0009: Obstacle tracking design and how it is evaluated without ground-truth tracks
- **Status:** accepted
- **Date:** 2026-10-18
## Context
- The contract needs stable track IDs, existence probabilities and COASTING predictions
  (`perception_msgs/Obstacle`). RELLIS-3D has no boxes and no track labels (ADR-0001), so "ID switches"
  need a defensible reference.
- First evaluation runs on 00001 test exposed three problems with any naive setup:
  1. **Self-returns.** 55 % of false-positive track-frames were within 1.5 m of `base_link`: returns from
     parts mounted on the Warthog, labelled `person` by the lidar model (`ego_footprint_00001.json`).
  2. **Label trails.** RELLIS-3D's map-based lidar annotation marks ground where people walked as `person`.
     In the camera wedge these points sit 9 cm above the ground with no person in the image, and both
     models independently say grass (`diag_person_wedge_00001.json`). 11 % of evaluated person clusters
     were such flat trails.
  3. **Fragments.** Distant or partial objects produce tiny clusters that split and merge, which inflates
     "unique objects" (436 reference objects in 105 s).
## Options considered
1. Score against raw labelled clusters. Penalises the tracker for label artifacts and self-hits; the numbers
   would describe the dataset, not the system.
2. Hand-label tracks. No time; does not scale.
3. **Pseudo-GT from labelled clusters with explicit don't-care rules (KITTI-style), plus an ego filter in the
   pipeline itself.**
## Decision
Option 3.
- **Pipeline:** an ego box (x ∈ [-0.9, 0.9], y ∈ [-0.8, 0.8], z ∈ [-0.5, 2.0] m in `base_link`, measured:
  every point inside is unlabelled) removes self-returns before the grid and clustering. Tracker: constant-
  velocity Kalman filter in the odom frame + Hungarian assignment (Mahalanobis gate χ²₂ 99 %), TENTATIVE →
  CONFIRMED after 3 hits in 5 frames, COASTING on misses, deleted after 5 misses; existence probability
  rises 30 % of the gap on a hit and falls 30 % on a miss.
- **Reference:** obstacle-class clusters from the ground-truth labels (same clustering code), linked by
  nearest neighbour (same class, ≤ 1 m, gaps ≤ 5 frames). Deliberately not our Kalman tracker, so the
  reference does not share its biases.
- **Don't-care:** reference objects with < 30 points, beyond 20 m (the grid edge), or with vertical extent
  < 0.5 m (label trails). Tracks that match only don't-care objects are neither hits nor false positives.
- **Metrics:** CLEAR-MOT and IDF1 (py-motmetrics), 1 m match radius, at existence gates 0 / 0.5 / 0.8.
  Tracker parameters are tuned on the 00001 val block; numbers are reported on the test block.
## Consequences
- The metrics measure how well tracks built from *predicted* semantics follow objects that perfect
  semantics would produce. They do not certify absolute tracker correctness, and odometry drift affects
  both sides equally.
- The ego box is machine-specific configuration (`config/sensors.yaml`); a different machine needs it
  re-measured with `pipeline/ego_footprint.py`.
- The label-trail finding is a dataset limitation worth stating wherever person IoU is quoted.
## Evidence
- `docs/measurements/tracking_00001_test.json` (results), `ego_footprint_00001.json`,
  `diag_person_wedge_00001.json`, `cpp_parity.json` (C++ tracker identical to the reference).
