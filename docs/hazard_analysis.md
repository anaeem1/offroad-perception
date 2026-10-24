# Hazard analysis (perception FMEA)

Scope: failures of the perception stack that can make the downstream planner act unsafely (move into
something) or unavailably (stop needlessly). Ratings are qualitative: **S**everity (1 minor … 5 injury),
**O**ccurrence (1 rare … 5 frequent, from measurements where they exist), **D**etection by the system at
runtime (1 always detected … 5 undetected). RPN = S × O × D. Every measured number links to its source.

| ID | Failure mode | Cause | Effect on consumer | Runtime detection | Mitigation in this design | S | O | D | RPN | Evidence / residual |
|---|---|---|---|---|---|---|---|---|---|---|
| H-1 | Lethal ground reported free | Segmentation error; obstacle class the lidar model cannot output (generic `object`) | Plans through an obstacle | None directly | Height-above-ground check raises cells to lethal regardless of class; UNKNOWN never free; ODD speed ≤ 2 m/s | 5 | 2 | 5 | 50 | 1.7 % of truly lethal cells reported free (`replay_00001_summary.json`) |
| H-2 | Stale output treated as current | Pipeline slower than the 10 Hz sensor; queueing | Plans on a world 0.5–1 s old | Health monitor: output age vs bound → SAFE_STOP | Age = now − sweep-start stamp, checked every 100 ms; contract forbids using outputs in SAFE_STOP | 4 | 3 | 1 | 12 | Jetson run 1: age ~720 ms, health held SAFE_STOP throughout (`jetson_replay_run1.txt`): detection worked. Throughput fix (ADR-0010 engines, CPU work) not yet re-measured on the Jetson |
| H-3 | Object in the near-field blind zone after standstill | Lidar blind within ~3 m; grid memory expires after 5 s | Starts moving into an unseen object | Cells revert to UNKNOWN (consumer must not drive) | Memory while moving (ADR-0008); tracked people coast into the zone | 5 | 2 | 3 | 30 | Residual: a static object placed while stopped is invisible. Needs a start-from-rest policy or near-field sensor (out of scope) |
| H-4 | Camera semantics misplaced | Extrinsic drift; residual pitch error | Wrong class inside the camera wedge | Not online (offline label-agreement check only) | −0.5° pitch correction (measured); lidar is primary, camera only refines | 3 | 2 | 4 | 24 | Agreement peaks 0.5° below nominal in all sequences (`projection_check.json`). Fault study: 4° yaw drift drops in-view mIoU 34.5 → 30.9, exactly lidar-only; grid safety (1.67 → 1.71 % lethal reported free) and tracking unchanged (`robustness_00001_test.json`). Recommend an online camera–lidar disagreement monitor |
| H-5 | Camera frame mismatched in time | No hardware sync; dropped frames | Fusion with a stale image | Frame must be within ±50 ms of t_wedge, else camera DEGRADED → LIDAR_ONLY | Match to the lidar's sweep of the camera FOV (contract v0.2) | 3 | 1 | 1 | 3 | 100 % of scans matched, 6 ms median (`camera_matching_00001.json`). Fault study: a camera **clock** offset is not detected (the matcher still reports 6 ms, since the stamps agree with themselves), but costs little at these speeds: 200 ms → in-view mIoU 34.5 → 32.2, grid and tracking unchanged (`robustness_00001_test.json`) |
| H-6 | Lidar loss | Sensor or driver fault | No geometry at all | Lidar age > 250 ms → FAILED → SAFE_STOP; consumer watchdog: no grid for 250 ms → stop | No camera-only geometry is claimed; SAFE_STOP instead. Found in review (2026-10-22): the reference planner decided once per grid, so with no grids its last command would have stood; it now has a watchdog (contract rule 8) | 5 | 1 | 1 | 5 | `health_monitor_node.cpp`, `planner_consumer_node.cpp`. Fault-injection replay (lidar dropped for 3 s) pending on the Jetson |
| H-7 | Phantom obstacles / tracks from the machine itself | Mast and antennas inside the lidar FOV | Spurious "person" tracks next to the machine; needless stops | — | Ego filter box, measured from data | 2 | 1 | 2 | 4 | Before the filter, 55 % of false-positive track-frames were self-returns (ADR-0009) |
| H-8 | Needless stop (free reported lethal) | Vegetation height, segmentation error | Availability loss, no safety loss | — | Grass/bush exempt from the height check; cost table | 2 | 2 | 5 | 20 | 0.8 % of free cells reported lethal (`replay_00001_summary.json`) |
| H-9 | Vehicle not tracked | Lidar model segments vehicles poorly | Planner lacks the motion of a moving vehicle | — | Vehicles still become lethal cells via semantics or the height check | 4 | 3 | 5 | 60 | Vehicle tracking recall 0.20 (`tracking_00001_test.json`). **Highest RPN.** Needs vehicle-capable training data before any deployment with traffic |
| H-10 | Track identity swap | Close objects; occlusion | Wrong velocity attached to an obstacle | — | Existence gating; COASTING margins doubled in the planner | 3 | 2 | 5 | 30 | 41 ID switches over 105 s / 66 objects (`tracking_00001_test.json`) |
| H-11 | Silent camera colour error | Wrong debayer pattern | Camera semantics degrade without any error | — | Pattern measured against dataset images (`BayerBG2RGB`) | 3 | 1 | 5 | 15 | Wrong pattern: error 22 vs 3 (`alignment_00001.json`). Recommend a startup self-test |
| H-12 | Deep water or mud judged traversable | Puddle/water/mud confusion; depth unobservable | Immobilization or worse | — | Water lethal, mud 140, puddle 80 in the cost table | 4 | 2 | 5 | 40 | Lidar water IoU 0.0 (rare class); outside ODD: water crossings |
| H-13 | Out-of-domain deployment | Different site, machine, weather | All rates above unknown | — | ODD explicitly excludes it (`odd.md`) | 5 | — | 5 | — | Single-dataset validation |
| H-14 | Lidar partly blinded | Rain, dust, mud on the window: returns lost or noisy | Lethal ground reported free more often, while the planner keeps moving as often as before | Point density < 20 % of 131k points → lidar DEGRADED (→ `MODE_LIDAR_ONLY` speed cap); nominal density here is 56 %, so it fires only at ≥ 75 % loss | None beyond the density check | 5 | 3 | 4 | 60 | Fault study: lethal reported free 1.67 % → 2.23 / 3.14 / 4.74 % at 25 / 50 / 75 % returns lost; planner moving 84 → 85–86 %; MOTA 0.59 → 0.53 at 50 % (`robustness_00001_test.json`). **Ties the top RPN.** Recommend a density check relative to a rolling nominal, and a degraded mode that is not named LIDAR_ONLY |
| H-15 | Lidar with fewer rings than the model was trained on | Sensor substitution on a different machine | Semantics and tracking collapse | Not detected at 32 rings (density 28 %); flagged at 16 (14 %) | ODD requires a 64-ring sensor | 5 | 2 | 4 | 40 | Fault study: 32 rings → lethal reported free 4.60 %, MOTA 0.50; 16 rings → 8.53 %, MOTA −0.08 (`robustness_00001_test.json`). Needs retraining per sensor |

## Highest risks and what would retire them
1. **H-9 vehicle tracking (RPN 60):** retrain or fine-tune the lidar model with vehicle-rich data; until then,
   the ODD excludes mixed traffic.
1. **H-14 partly blinded lidar (RPN 60):** the fault study shows safety degrading while availability does not,
   so neither the planner nor the health monitor gets a signal. A density check against a rolling nominal (here
   56 % valid returns) with a DEGRADED threshold around −25 % would catch it at the 25–50 % loss where lethal
   cells reported free already rise 1.3–1.9×.
2. **H-1 lethal-as-free (RPN 50):** the height check covers geometry; semantics-only misses on flat hazards
   (mud, holes) remain. Validate on site-specific data; consider a negative-obstacle (hole) check.
3. **H-12 water/mud (RPN 40)** and **H-3 standstill blind zone (RPN 30):** both need either operating rules
   (ODD) or additional sensing.

## Verification status
- Measured on held-out data: H-1, H-5, H-7, H-8, H-9, H-10, H-11 (numbers above).
- Tested by fault injection on held-out data (`robustness_00001_test.json`, `pipeline/faults.py`): H-4 (extrinsic
  drift), H-5 (camera clock offset), H-14 (returns lost, range noise), H-15 (fewer rings), camera exposure.
- Demonstrated on the target hardware: H-2 (health monitor forced SAFE_STOP when output was stale).
- Not yet tested by fault injection: H-6 (total lidar loss, a runtime timing behaviour). H-6 test built: `scripts/jetson/make_dropout_clip.py`
  removes the lidar topic for 3 s from the replay clip; the replay report lists each lidar status change with
  the data age at that moment, and the planner's watchdog decisions.
