# ADR-0010: Deploy SegFormer at 480×768 FP16 and SalsaNext FP16; INT8 and DLA rejected for now
- **Status:** accepted; INT8 lidar re-opened only if a calibration variant passes (see Evidence)
- **Date:** 2026-10-20
## Context
- First on-device replay (`docs/measurements/jetson_replay_run1.txt`): with SalsaNext FP16 and SegFormer
  640×1024 FP16, the lidar path took 150 ms per scan against a 100 ms frame period. 203 of 319 scans were
  processed, output was ~720 ms old, and health stayed SAFE_STOP. The two engines share the GPU at
  ~88 % duty (lidar 47 ms + camera 41 ms per 100 ms) and slow each other down.
- Latency sweep on the same board (`docs/measurements/jetson_latency_sweep.txt`, MAXN, clocks locked; other
  people's workloads were still running, but single-model times match run 1's in-pipeline GPU times with
  them stopped). The concurrent rows run two processes, which the GPU time-slices; the runtime uses one
  process with two streams, so those rows show contention, not runtime numbers:

  | Engine | Median GPU time |
  |---|---|
  | SalsaNext FP16 | 49.1 ms |
  | SalsaNext INT8 (uncalibrated speed bound) | 25.8 ms |
  | SalsaNext FP16 on DLA (GPU fallback allowed) | 143.7 ms |
  | SegFormer 640×1024 FP16 | 40.5 ms |
  | SegFormer 480×768 FP16 | 20.0 ms |
  | SegFormer 640×1024 INT8 (bound) | 36.3 ms |
  | Both at once, GPU + GPU, 640 | lidar 111.5 / camera 91.6 ms |
  | Both at once, DLA + GPU, 480 | lidar 201.9 / camera 36.0 ms |
## Options considered
1. **DLA for SalsaNext** so it runs beside SegFormer. Measured 3× slower alone and ~4× slower concurrently.
   11 layers are unsupported on DLA (DepthToSpace, Shuffle, ArgMax/TopK, Cast), 20 convs fail DLA
   validation, and the 16-subgraph-per-core limit is hit. The DLA↔GPU transitions dominate. Rejected.
2. **INT8 for SegFormer.** Only 10 % faster (transformer: attention and LayerNorm stay in higher
   precision). Not worth the calibration and accuracy risk. Rejected.
3. **SegFormer at 480×768 FP16.** 2.0× faster; costs 1.7 test mIoU (42.75 → 41.08,
   `camera_onnx_*_test.json`). Inside the camera wedge, every lidar point still samples a camera cell.
4. **SalsaNext INT8 via explicit quantization (QDQ ONNX).** 1.9× faster (speed bound). Calibrated on
   129 val scans with ONNX Runtime (MinMax, symmetric, per-channel, Conv only, float biases). Entropy
   calibration ran out of memory at 96 and 180 GB. TensorRT builds a QDQ model directly as INT8, and
   ONNX Runtime runs the same file on the server, so the accuracy cost is measured before deployment.
5. Higher TensorRT builder optimization level (5): the build was killed by the out-of-memory killer on the
   8 GB board.
## Decision
Option 3: SegFormer 480×768 FP16. SalsaNext stays **FP16**: option 4 failed its accuracy check. With
MinMax calibration over all Conv layers, lidar test mIoU fell from 41.68 to **16.39**, and only 60 % of
point labels agreed with FP32 (`lidar_int8_vs_fp32.json`). The lidar context runs on the highest-priority
CUDA stream (critical path). GPU work per frame drops from ~89 ms to ~69 ms. Whether that keeps up with
10 Hz is measured on the Jetson, not assumed.
## Consequences
- **Lidar seg is over its 20 ms stage budget** (49 ms FP16). Camera seg at 480×768 (20 ms) frees 10 ms of
  the 50 ms model budget, which is not enough. The end-to-end bound (ADR-0005) and the measured replay
  decide whether this is acceptable. Recorded as a known budget overrun, not hidden.
- INT8 is still the only lever left for the lidar model. Calibration variants are being tested: moving-
  average MinMax, the first/last conv kept in FP16, and percentile. Adopting one needs ≤ 2 mIoU loss on test.
- Accuracy cost is explicit: −1.7 camera mIoU (lidar unchanged at FP16). Fusion and grid metrics
  should be re-measured with the deployed engines' outputs (Jetson-vs-reference comparison).
- DLA stays unused. A different lidar network designed for DLA would change that, but that is a research
  detour outside this project's scope.
## Evidence
- `docs/measurements/jetson_latency_sweep.txt`, `jetson_replay_run1.txt`, `camera_onnx_640x1024_test.json`,
  `camera_onnx_480x768_test.json`, `lidar_int8_vs_fp32.json` (INT8 vs FP32 on the same test scans).
