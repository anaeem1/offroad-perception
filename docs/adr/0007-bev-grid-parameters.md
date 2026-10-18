# ADR-0007: BEV grid is 40 m × 40 m at 0.2 m, built from one lidar sweep
- **Status:** accepted
- **Date:** 2026-10-18
## Context
- The contract left grid extent and resolution open (initial guess 40 m × 40 m @ 0.2 m).
- The Ouster's ground rings spread out with range, so far cells get few or no returns. Measured on the
  sequence 00001 val block (`docs/measurements/grid_coverage_00001.json`), as the fraction of cells with
  at least one return:

  | Range | 1 sweep @ 0.2 m | 5 sweeps (0.5 s) @ 0.2 m | 1 sweep @ 0.4 m |
  |---|---|---|---|
  | 0–5 m | 0.46 | 0.54 | 0.52 |
  | 5–10 m | 0.73 | 0.92 | 0.91 |
  | 10–15 m | 0.33 | 0.64 | 0.59 |
  | 15–20 m | 0.13 | 0.34 | 0.29 |
  | 20–30 m | ≤ 0.05 | ≤ 0.15 | ≤ 0.12 |

  0–5 m stays low in every setting: that is the lidar's blind circle around the machine, not resolution.
- RELLIS-3D driving speed is a median 1.4 m/s. Stopping distance is a few metres, so 15–20 m of
  forward reach is enough for this ODD. A faster machine would need more (see Consequences).
## Options considered
1. **40 m × 40 m @ 0.2 m, single sweep.** Every cell describes the same 100 ms sweep as `header.stamp`.
2. **Same grid, accumulating the last 5 sweeps with odometry.** Nearly doubles coverage at 10–20 m.
   But moving objects smear (a person at 1.5 m/s moves 0.75 m in 0.5 s), the stamp no longer bounds
   the data's age, and the C++ node needs a motion-compensated ring buffer.
3. **0.4 m cells.** Similar coverage gain from a single sweep, but a 0.4 m cell hides thin obstacles
   (poles, people seen side-on) inside mixed cells.
## Decision
Option 1. Centred on `base_link`; 200 × 200 cells; height band ±3 m; height check: lethal when the
highest point within 2.5 m of the local ground (lowest point in a 1 m window) rises more than 0.3 m above
it, except for grass and bush, whose semantic cost already reflects height.
## Consequences
- Beyond ~15 m most cells are UNKNOWN (255). That is honest, and the contract already forbids treating
  it as free. The planner has to plan with UNKNOWN as a cost, not as free space.
- If a faster machine needs more reach, option 2 for *static* classes only (dynamic classes from the
  current sweep) is the upgrade path. It is a change to the grid builder, not to the contract.
- Ground estimation by a local minimum is simple and can misread steep slopes as steps. It is monitored
  through the traversability confusion matrix in the replay metrics.
## Evidence
- `docs/measurements/grid_coverage_00001.json` (coverage); BEV cell IoU and the traversability confusion
  in the week 2 replay metrics.
