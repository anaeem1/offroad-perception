# ADR-0005: Output age guarantee must account for lidar sweep time
- **Status:** proposed (needs owner decision: it weakens a consumer guarantee in contract v0.1)
- **Date:** 2026-10-06
## Context
- Contract v0.1: `header.stamp` = lidar measurement time, and "consumers may assume outputs are no
  older than 150 ms when NOMINAL".
- Measured on RELLIS-3D bag 00004 (`docs/measurements/bag_00004_inspection.json`): the Ouster driver
  stamps each scan at the **start** of the 100 ms sweep (per-point `t` spans the sweep), and the scan
  reaches subscribers a median **127 ms** after its stamp (p95 137 ms).
- Our processing budget is 100 ms end-to-end (README). So output age at publish is about
  127 + 100 ≈ 227 ms. The 150 ms guarantee cannot hold even with zero processing time.
## Options considered
1. **Keep stamp = sweep start; raise the guarantee to 250 ms.** Honest, and consistent with the stamp
   rule. Consumers predict forward by the age they observe.
2. **Re-stamp outputs at sweep end** (stamp + 100 ms). Age drops by 100 ms on paper, but the stamp no
   longer matches when most points were measured. It hides latency instead of removing it.
3. **Keep 150 ms; measure age from message arrival instead of the stamp.** Misleading: consumers care
   about the age of the world state, not of our processing.
## Decision (proposed)
Option 1. Contract v1.0 states: `header.stamp` is the sweep-start time of the source scan, and outputs are
no older than 250 ms after `header.stamp` in NOMINAL. `PerceptionHealth.end_to_end_latency_ms`
reports publish time minus `header.stamp`, and exceeding 250 ms counts as DEGRADED.
## Consequences
- Consumers (planner) must forward-predict obstacles by the observed age; at 5 m/s, 250 ms is 1.25 m.
  The planner's safety margins must be sized for this. That is the point of making it explicit.
- Latency reporting splits into sensor/driver latency (~127 ms, outside our control on replay) and
  our pipeline latency (the 100 ms budget). Both get measured on the Jetson in week 3.
- Changing a guarantee is breaking under the contract's versioning rule: the version goes from 0.1 to 1.0.
## Evidence
- `docs/measurements/bag_00004_inspection.json` → `receive_minus_stamp_ms`, `pointcloud.fields` (per-point `t`).
