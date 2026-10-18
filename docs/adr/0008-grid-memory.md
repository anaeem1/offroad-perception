# ADR-0008: The BEV grid remembers static evidence for 5 s (supersedes the single-sweep part of ADR-0007)
- **Status:** accepted
- **Date:** 2026-10-18
## Context
- ADR-0007 chose a 40 m × 40 m @ 0.2 m grid built from one sweep and listed temporal accumulation as an
  optional upgrade.
- Writing the contract's consumer (`planner_consumer`) exposed the flaw. The lidar never sees the ground within
  ~3 m of the machine, so in a single-sweep grid the forward corridor (|y| < 0.9 m, from the vehicle front)
  is **100 % UNKNOWN at 0.9–3 m** on both 00001 blocks. The contract says UNKNOWN is never free, so a
  compliant planner always sees the corridor blocked 0.9 m ahead and **can never move**.
## Options considered
1. **Treat the blind zone specially in the planner.** Breaks the contract's core rule, and pushes a
   perception limitation onto every consumer.
2. **Accumulate points from the last N sweeps.** ~65k points per sweep × N; heavy on the Orin, and moving
   objects smear.
3. **Grid memory keyed by odom-frame cells.** After building the current grid, cells the sweep did not observe
   take the value remembered from earlier sweeps if it is no older than a maximum age, with confidence
   decayed linearly by age. Dynamic classes (person, vehicle) are never remembered: they come only from the
   current sweep and the tracker.
## Decision
Option 3 with a **5 s** maximum age. The age was chosen on the 00001 val block as the shortest age that
brings every corridor band within 3 points of the 8 s plateau:

| Corridor UNKNOWN fraction (val) | 0.9–2 m | 2–3 m | 3–4 m | 4–6 m |
|---|---|---|---|---|
| single sweep | 1.00 | 0.99 | 0.82 | 0.06 |
| memory 2 s | 0.47 | 0.13 | 0.08 | 0.00 |
| memory 3 s | 0.17 | 0.12 | 0.07 | 0.00 |
| **memory 5 s** | **0.13** | **0.10** | **0.06** | **0.00** |
| memory 8 s | 0.11 | 0.08 | 0.04 | 0.00 |

On the held-out test block, 5 s leaves 0.09 / 0.08 / 0.06 UNKNOWN in the first three bands (single sweep:
1.00 / 1.00 / 0.85).
## Consequences
- **Contract v0.2 (additive):** grid cells may carry static evidence up to 5 s older than `header.stamp`.
  Their confidence is scaled down by age, and the grid-level `source_flags` gets bit 4 when any cell was
  remembered. Cells never observed, or observed more than 5 s ago, are still UNKNOWN.
- **Known hazard, for the week 4 hazard analysis:** after standing still for more than 5 s, the near field
  expires back to UNKNOWN, and something could enter the blind zone unseen. The tracker carries people that
  were seen walking in (COASTING), but a static object placed while stopped is invisible. Starting from rest
  needs an explicit policy.
- The C++ `GridMemory` (hash map of odom cells) matches the Python reference exactly over 60 consecutive scans
  (`docs/measurements/cpp_parity.json`: 0 of 2.4 M cells differ).
## Evidence
- `docs/measurements/replay_00001_summary.json` (corridor tables, both blocks), `cpp_parity.json`.
