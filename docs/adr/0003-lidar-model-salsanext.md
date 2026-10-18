# ADR-0003: Lidar segmentation uses the released RELLIS-3D SalsaNext checkpoint, unmodified
- **Status:** accepted
- **Date:** 2026-10-06
## Context
- The lidar branch needs per-point semantics in a 20 ms budget on an Orin NX, built with TensorRT.
- RELLIS-3D publishes a SalsaNext checkpoint trained on this exact sensor (Ouster OS1-64, 64×2048)
  and dataset, with its training configs. The code is MIT-licensed.
- Four weeks total, with the guardrail "prefer off-the-shelf; no research detours".
## Options considered
1. **Released SalsaNext checkpoint as-is.** A range-image CNN with only Conv, BatchNorm, LeakyReLU,
   AvgPool, PixelShuffle and Softmax, all native TensorRT layers.
2. **Retrain SalsaNext or a lighter range net** (e.g. FIDNet, CENet). Possibly faster or better, but
   costs GPU days and a training pipeline we would have to validate.
3. **Point / sparse-voxel networks** (KPConv, Cylinder3D, MinkUNet). KPConv is the other published
   RELLIS-3D baseline at 18.6 mIoU. Sparse convolutions need custom TensorRT plugins.
## Decision
Option 1. Evaluate it ourselves, export it to ONNX unchanged, and retrain only if the measured
accuracy or Jetson latency fails its budget.
## Consequences
- **Class set is narrower than the camera's.** Training folded dirt, sky, object, asphalt and building
  into void (the lidar labels contain almost none: e.g. 2 points of `object` in the whole set).
  So the lidar branch cannot say "object". Outside the camera's 37.7° wedge, a generic obstacle is
  visible only geometrically. The BEV stage therefore must apply a height-above-ground check that
  can raise cost on its own (cost table rule 1). This is a requirement for week 2, not an option.
- **Preprocessing is frozen by the checkpoint.** The spherical projection uses fov ±22.5° (the sensor's
  vertical FOV is narrower) and specific means/stds. The C++ node must reproduce
  `ml/lidar/range_projection.py` exactly; changing the projection means retraining.
  The remission channel is the PointCloud2 `intensity` field divided by 65535, bit-exact
  (`pipeline/rellis.py`). The labelled scans have identical xyz in identical point order to the bag
  clouds (`docs/measurements/alignment_*.json`).
- **KNN post-processing.** The published number uses a GPU KNN refinement (unfold + top-k) that has
  no TensorRT equivalent. Measured: it adds only +0.6 mIoU overall, but the gain is concentrated in
  the tracked obstacle classes (barrier +4.0, vehicle +2.5, pole +1.6, person +1.3 IoU). So the
  choice is a latency trade-off, decided in week 3 against the measured Orin NX cost of a CUDA KNN
  kernel (0.8 ms in PyTorch on an H100). Until then, claimed accuracy is the CNN-only number.
- **Licensing.** The code is MIT, but the RELLIS-3D data (and so these weights) is CC BY-NC-SA 3.0.
  That's fine for this portfolio; a production system would retrain on owned data.
## Evidence
- Test split (3,343 scans), `docs/measurements/lidar_salsanext_test.json`:

  | | mIoU, 14 classes | Published convention (÷15) |
  |---|---|---|
  | CNN only (deployed path) | **41.79** | 39.00 |
  | CNN + KNN | 42.41 | 39.58 |
  | RELLIS-3D README (CNN + KNN) | — | 40.20 |

  The published mean divides the 14 class IoUs by 15, so it reads 2.8 points lower than the
  14-class mean; under its own convention we reproduce it within 0.6. The upstream and strict IoU
  conventions agree, so the model does not emit void on labeled points. Weak classes: water 0.0,
  fence 12.6, mud 10.3, rubble 14.1 (all rare).
- No retraining was needed: the reproduced number matches the published one.
- ONNX export and PyTorch parity (opset 16 and 17): `docs/measurements/onnx_salsanext_rellis_op*.json`
- Jetson latency: week 3.
