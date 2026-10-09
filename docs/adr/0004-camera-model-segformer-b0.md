# ADR-0004: Camera segmentation is SegFormer-B0, fine-tuned on RELLIS-3D at 640×1024
- **Status:** accepted (resolution provisional until measured on the Orin NX)
- **Date:** 2026-10-06
## Context
- Camera seg budget: 30 ms on an Orin NX (GPU shared with lidar seg).
- RELLIS-3D images are 1920×1200. The camera sees only a 37.7° forward wedge (ADR-0002), so camera
  accuracy matters inside that wedge only.
- The decided architecture names SegFormer. The open questions were size, resolution and initialization.
- Published RELLIS-3D image baselines: HRNet+OCR 48.83 and GSCNN 50.13 mIoU (18 classes). Both are far
  too heavy for the budget.
## Options considered
1. **SegFormer-B0** (3.7 M params) vs **B1/B2** (2-7× the compute). B0 is the only one with a
   realistic chance at 30 ms on an Orin NX at a useful resolution.
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
- If the Orin NX misses 30 ms, the first lever is a 480×768 export (a flag, no retraining needed;
  SegFormer has no positional encodings). Its accuracy cost must be measured before switching.
- Licensing: SegFormer weights are under NVIDIA's non-commercial license, and the Cityscapes init adds
  Cityscapes' terms. Fine for a portfolio; a product would need another backbone or a commercial license.
## Evidence
- Test split (1,672 images), `docs/measurements/camera_segformer_b0_test.json`. 15.5 min on 1× H100.

  | | mIoU, 19 classes | 18 published classes |
  |---|---|---|
  | SegFormer-B0 640×1024 (ours) | **42.7** | 45.1 |
  | HRNet+OCR full-res (RELLIS-3D README) | 48.83 (÷19) | 51.6 |

  Strong where it matters most: person 89.2 (HRNet+OCR 89.5), sky 97.1, grass 86.3, concrete 81.7.
  Weak on thin or rare classes: pole 1.7, log 1.5, building 5.6, water 8.0, asphalt 20.7, barrier
  25.3, fence 26.9. Thin obstacles therefore must not rely on the camera; lidar pole IoU is 47.9
  (ADR-0003). Best val mIoU 49.15 (iteration 4000); the val-test gap is unexplained so far.
- ONNX export and parity: `docs/measurements/onnx_segformer_b0_640x1024_op*.json`
- Jetson latency: week 3.
