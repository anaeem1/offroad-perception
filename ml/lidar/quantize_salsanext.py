"""Explicit INT8 quantization of SalsaNext for TensorRT: a QDQ ONNX model, calibrated on val scans.

TensorRT 8.6 builds QDQ models directly as INT8 (no calibrator on the Jetson), and ONNX Runtime can run the
same file on the server, so the accuracy cost is measured before deploying. Conv layers only, symmetric,
per-channel weights: the form TensorRT fuses; everything else stays FP16.

  python -m ml.lidar.quantize_salsanext --calib-scans 200
  -> <data root>/onnx/salsanext_rellis_op17_int8qdq.onnx
"""
import argparse
from pathlib import Path

import numpy as np
import yaml
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType,
                                      quant_pre_process, quantize_static)

from ml.lidar.eval_salsanext import RUN_DIR, RellisLidar
from ml.lidar.range_projection import SensorConfig
from pipeline.models import ONNX_DIR


class ValScans(CalibrationDataReader):
    def __init__(self, n):
        data = yaml.safe_load(open(RUN_DIR / "data_cfg.yaml"))
        self.ds = RellisLidar("val", SensorConfig.from_arch_cfg(RUN_DIR / "arch_cfg.yaml"), data["learning_map"])
        self.idx = iter(np.linspace(0, len(self.ds) - 1, n).astype(int).tolist())

    def get_next(self):
        i = next(self.idx, None)
        return None if i is None else {"input": self.ds[i][0][None].numpy()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=ONNX_DIR / "salsanext_rellis_op17.onnx")
    ap.add_argument("--out", type=Path, default=ONNX_DIR / "salsanext_rellis_op17_int8qdq.onnx")
    ap.add_argument("--calib-scans", type=int, default=200)
    ap.add_argument("--method", default="Entropy", choices=["Entropy", "Percentile", "MinMax"])
    args = ap.parse_args()
    pre = args.out.with_suffix(".pre.onnx")
    quant_pre_process(str(args.src), str(pre), skip_symbolic_shape=True)
    quantize_static(str(pre), str(args.out), ValScans(args.calib_scans), quant_format=QuantFormat.QDQ,
                    op_types_to_quantize=["Conv"], per_channel=True, activation_type=QuantType.QInt8,
                    weight_type=QuantType.QInt8, calibrate_method=getattr(CalibrationMethod, args.method),
                    extra_options={"ActivationSymmetric": True, "WeightSymmetric": True})
    pre.unlink(missing_ok=True)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
