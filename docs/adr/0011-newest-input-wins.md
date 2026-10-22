# ADR-0011: Sensor input queues hold one message; a node that falls behind drops stale input
- **Status:** accepted
- **Date:** 2026-10-22
## Context
- Jetson run 1 (`docs/measurements/jetson_replay_run1.txt`): the lidar path took 150 ms per scan against a
  100 ms period. The cloud subscription kept 5 messages, so scans waited in a full queue: the median receive
  delay was 570 ms. Of that, 123 ms is sensor lag recorded in the bag (`clip_00001_s100-400_lidar_lag.json`)
  and ~447 ms is queueing, i.e. ~4.5 periods. Output age was ~720 ms.
- Under overload a queue does not raise throughput; it only adds delay. 116 of 319 scans were dropped anyway,
  once the queue was full.
- A planner gains nothing from an old scan when a newer one is waiting. The contract (ADR-0005) bounds
  output age, not the fraction of scans processed.
## Options considered
1. **Keep depth 5.** Under overload, age grows to about depth × period plus processing time. Rejected: that
   is what run 1 measured.
2. **Depth 1: the newest message replaces a waiting one.** Under overload, age is the sensor lag, plus up
   to one period of waiting, plus processing time. Throughput is 1 / processing time; skipped scans are
   simply never processed. No extra code.
3. **Depth 5, but skip scans older than a threshold in the callback.** Same effect as option 2, with a tuning
   parameter and more code.
## Decision
Option 2 for every sensor-rate input of the processing chain: the lidar cloud and camera image in
`segmentation_fusion_node`, and the fused points in `bev_tracker_node`. Low-rate state stays queued:
odometry (depth 100, needed for deskew), camera info, the health and planner inputs.
Odometry also moves to its own callback group in `segmentation_fusion_node`, so its updates never wait
behind a lidar callback (they shared one group before).
## Consequences
- Age under overload is bounded: about 123 ms + U(0, 100) ms + processing time, instead of growing with the
  queue. For 150 ms processing that is ~275–375 ms rather than ~720 ms, which is still over the 250 ms bound.
  Depth 1 bounds the waiting time; only faster processing (ADR-0010) brings the age under the bound.
- Dropped scans become a measured quantity: the processed fraction (CSV rows / clip scans) is reported by
  `replay_on_jetson.sh` and `pipeline/split_receive_delay.py`.
- The tracker sees irregular time steps when scans are dropped. The Kalman filter already predicts by the
  actual dt between scans (`tracker.cpp`), so this needs no change. A longer gap widens the predicted
  covariance and loosens the gate for that step; the Jetson-vs-reference comparison shows whether tracking
  changes.
- Python reference replay processes every scan offline; it has no queue, so parity is unaffected.
## Evidence
- `docs/measurements/jetson_replay_run1.txt` (receive delay, rows processed vs clip scans),
  `docs/measurements/clip_00001_s100-400_lidar_lag.json` (sensor lag recorded on the replayed clip).
- To verify: Jetson run 2 reports the receive-delay split (sensor lag vs queueing) and the processed fraction.
