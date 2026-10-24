# ADR-0005: Output age guarantee must account for lidar sweep time
- **Status:** accepted (owner decision, 2026-10-24): contract v1.0
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
## Decision
Option 1. Contract v1.0 states: `header.stamp` is the sweep-start time of the source scan, and outputs are
no older than 250 ms after `header.stamp` in NOMINAL. `PerceptionHealth.end_to_end_latency_ms`
reports publish time minus `header.stamp`. Exceeding 250 ms forces MODE_SAFE_STOP, as implemented in
`health_monitor_node.cpp` and assumed by hazard H-2. (The first draft said "DEGRADED", which the health
message has no mode for.)
## Consequences
- Consumers (planner) must forward-predict obstacles by the observed age; at the ODD's 2 m/s, 250 ms is 0.5 m
  (1.25 m at 5 m/s).
  The planner's safety margins must be sized for this. That is the point of making it explicit.
- Latency reporting splits into sensor/driver latency (~127 ms, outside our control on replay) and
  our pipeline latency (the 100 ms budget). Both get measured on the Jetson in week 3.
- Changing a guarantee is breaking under the contract's versioning rule: the version goes from 0.1 to 1.0.
## Jetson run 1 (2026-10-19)
- The deployed params already run the proposed 250 ms bound (health monitor and planner); the contract text
  still says 150 ms until this ADR is decided.
- The bound was not what stopped the planner. Output age was ~720 ms, so it held SAFE_STOP for all 203
  decisions under either bound. Of the 570 ms median receive delay, 123 ms is sensor lag recorded in the
  bag and ~447 ms is the scan waiting for our node, which processed a scan in 150 ms against a 100 ms period.
  The bound only matters once the lidar path keeps up with 10 Hz (ADR-0010).
- The sensor lag on the replayed clip is stable: median 123.45 ms, max 130.41 ms. So under a 150 ms bound,
  the whole pipeline (receive to publish) would have ~25 ms, which is less than the lidar model alone (49 ms).
## Evidence
- `docs/measurements/bag_00004_inspection.json` → `receive_minus_stamp_ms`, `pointcloud.fields` (per-point `t`).
- `docs/measurements/clip_00001_s100-400_lidar_lag.json` (lag on the replayed clip),
  `docs/measurements/jetson_replay_run1.txt` (on-device ages).
