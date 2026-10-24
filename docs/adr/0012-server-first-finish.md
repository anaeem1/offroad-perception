# ADR-0012: Finish on the server first; the second on-device replay moves after the write-up
- **Status:** accepted (owner decision, 2026-10-24)
- **Date:** 2026-10-24
## Context
- Six days to the application deadline (Oct 30) and two to the feature stop (Oct 26). Each Jetson round trip
  (bundle → board agent → results) has taken one to two days of owner time.
- The week 3 exit criterion needs on-device evidence (README latency table from a run that keeps up with 10 Hz,
  planner consuming the contract on the board). Bundle 6 is ready, but run 2 has not happened.
- The server has four idle H100s today. Two things the write-up lacks can be built and measured there:
  how the system degrades under sensor faults, and a demo video of the full pipeline on held-out data.
- While preparing the robustness study, it turned out that ONNX Runtime had been running every replay on the
  CPU (cuDNN 9 not on the loader path). Fixed in `env/activate.sh`; replays now take ~5 min instead of ~26.
## Options considered
1. **Keep the plan:** wait for run 2, then fill the latency table. The write-up and video compete for the same
   last days.
2. **Server first, Jetson after:** run the robustness study and the demo on the server now, write the README as
   an architecture document, and slot run 2 in when the owner has time with the board. On-device results are
   reported as they stand (run 1 measured, run 2 ready to run).
3. **Drop on-device work from the write-up.** Throws away measured evidence (run 1, the sweep, ADR-0010/0011).
## Decision
Option 2. The week 3 criterion "C++/TensorRT nodes on the Jetson" is met by run 1 (builds and runs end to end
on the board); "README latency table filled" and "planner consumes the contract" are met offline for now
(planner rules ported to the reference replay, `pipeline/planner.py`) and on the board once run 2 is in.
The robustness study and demo are evaluation of the existing system, not new features, so they fit before the
Oct 26 feature stop.
## Consequences
- The README states plainly which numbers are on-device (run 1, latency sweep) and which are offline.
- The end-to-end latency on the board under the ADR-0010/0011 configuration stays unmeasured until run 2;
  that is the first item to do after the write-up.
- All replay timings measured before 2026-10-24 (`timing_ms_median_server` in run metrics) were CPU timings.
  They were never quoted as performance claims; the Jetson numbers come from the board.
## Evidence
- `docs/measurements/jetson_replay_run1.txt`, `jetson_latency_sweep.txt` (on-device so far).
- `docs/measurements/robustness_00001_test.json` (fault-injection study).
