# ADR-0001: Semantic fusion + traversability instead of 3D box detection
- **Status:** accepted
- **Date:** 2026-10-04
## Context
RELLIS-3D provides per-point lidar and per-pixel camera semantic labels but no 3D bounding boxes.
Off-road and construction autonomy is dominated by "where can the machine drive", and much of the
relevant terrain (mud, rubble, puddles, tall grass) is not box-shaped.
## Options considered
1. Train a 3D box detector on pseudo-labels derived from semantic clusters.
2. Semantic segmentation on both modalities, late fusion into a BEV traversability grid,
   plus clustering of non-traversable points for discrete obstacle tracking.
## Decision
Option 2.
## Consequences
- Supervision matches the available ground truth; no pseudo-label noise.
- Traversability is the primary output; obstacles are secondary and coarser (cluster-level boxes).
- Tracking quality depends on clustering stability; this must be monitored as a risk.
## Evidence
To be filled with per-class IoU and track ID-switch counts in week 2.
