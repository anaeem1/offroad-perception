# ADR-0006: Ship ONNX models at opset 17 for TensorRT 8.6 (JetPack 6.0)
- **Status:** accepted
- **Date:** 2026-10-09
## Context
- The deployment Jetson is an Orin NX 8 GB on L4T R36.3.0 (JetPack 6.0), TensorRT 8.6.2
  (`docs/measurements/jetson_info.txt`).
- Both models were exported at opset 16 and 17 with identical PyTorch parity. They differ in one place:
  at opset 17, SegFormer's LayerNorm is a single `LayerNormalization` node; at opset 16 it is split
  into ReduceMean/Sub/Pow/Sqrt/Div (`docs/measurements/onnx_*_op*.json`).
- Toy models covering both op sets parse and build as FP16 engines at both opsets on the Jetson
  (`docs/measurements/jetson_trt_toy.txt`).
## Options considered
1. **Opset 16.** Works, but TensorRT 8.6 then sees LayerNorm as five separate ops.
2. **Opset 17.** TensorRT 8.6 imports `LayerNormalization` as its native normalization layer, which is
   the form it recommends for FP16 accuracy (the split form can overflow in FP16).
3. **Opset ≥ 18.** Not supported by TensorRT 8.6's parser, and our pinned exporter does not target it.
## Decision
Opset 17 for both models. The opset 16 files stay in the data root as a fallback, not as a shipping
artifact.
## Consequences
- The server exporter stays on torch 2.5.x (TorchScript exporter); recent torch defaults to the
  dynamo exporter at opset 18+.
- Moving the Jetson to JetPack 6.1+ (TensorRT 10.3) would allow newer opsets, but needs no change:
  opset 17 stays supported.
- To check in week 3: SegFormer FP16 accuracy on the Jetson against the PyTorch FP32 test mIoU.
## Evidence
- `docs/measurements/jetson_trt_toy.txt`: all four toy models PASS; toy latency is the same at both opsets.
