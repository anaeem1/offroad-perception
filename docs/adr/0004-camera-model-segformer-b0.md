# ADR-0004: Camera segmentation is SegFormer-B0, fine-tuned on RELLIS-3D at 640×1024
- **Status:** accepted (resolution provisional until measured on the Orin Nano)
- **Date:** 2026-10-06
## Context
- Camera seg budget: 30 ms on an Orin Nano (no DLA; GPU shared with lidar seg).
- RELLIS-3D images are 1920×1200. The camera sees only a 37.7° forward wedge (ADR-0002), so camera
  accuracy matters inside that wedge only.
- The decided architecture names SegFormer. The open questions were size, resolution and initialization.
- Published RELLIS-3D image baselines: HRNet+OCR 48.83 and GSCNN 50.13 mIoU (18 classes). Both are far
  too heavy for the budget.
## Options considered
1. **SegFormer-B0** (3.7 M params) vs **B1/B2** (2-7× the compute). B0 is the only one with a
   realistic chance at 30 ms on an Orin Nano at a useful resolution.
2. **Input 640×1024** (exact 1.6 aspect, divisible by 32) vs **480×768** vs full resolution.
3. **Init from Cityscapes-finetuned B0** (full decoder pretrained on driving scenes) vs ImageNet-only
   `mit-b0` encoder.
## Decision
B0, 640×1024, initialized from `nvidia/segformer-b0-finetuned-cityscapes-1024-1024` with a new
19-class head. Void is ignored in training and scoring, so the model never predicts it, and "don't
know" is carried by confidence. 8k iterations, batch 16, AdamW, poly LR.
## Consequences
- The ONNX graph takes RGB 0-255 and normalizes internally, so the C++ node only resizes.
- Outputs are probabilities at 1/4 input resolution (160×256). Fusion samples them directly; one
  output cell is 7.5 px of the original image, well below the ~25 px calibration uncertainty
  found in the projection check.
- If the Orin Nano misses 30 ms, the first lever is a 480×768 export (a flag, no retraining needed;
  SegFormer has no positional encodings). Its accuracy cost must be measured before switching.
- Licensing: SegFormer weights are under NVIDIA's non-commercial license, and the Cityscapes init adds
  Cityscapes' terms. Fine for a portfolio; a product would need another backbone or a commercial license.
## Evidence
- Test-split per-class IoU: `docs/measurements/camera_segformer_b0_test.json`
- ONNX export and parity: `docs/measurements/onnx_segformer_b0_640x1024_op*.json`
- Jetson latency: week 3.
