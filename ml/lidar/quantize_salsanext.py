"""Explicit INT8 quantization of SalsaNext for TensorRT: a QDQ ONNX model, calibrated on val scans.

TensorRT 8.6 builds QDQ models directly as INT8 (no calibrator on the Jetson), and ONNX Runtime can run the
same file on the server, so the accuracy cost is measured before deploying. Conv layers only, symmetric,
per-channel weights: the form TensorRT fuses; everything else stays FP16.

  python -m ml.lidar.quantize_salsanext --calib-scans 200
  -> <data root>/onnx/salsanext_rellis_op17_int8qdq.onnx
"""
import argparse
import json
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
    ap.add_argument("--calib-scans", type=int, default=129)  # not a multiple of --flush-every: ORT 1.20 raises "No data is collected" otherwise
    ap.add_argument("--method", default="Entropy", choices=["Entropy", "Percentile", "MinMax"])
    ap.add_argument("--flush-every", type=int, default=4)
    ap.add_argument("--exclude", nargs="*", default=[], help="Conv node names kept in FP16 (sensitive layers)")
    ap.add_argument("--exclude-worst", type=int, default=0,
                    help="also keep in FP16 the Convs reading the K lowest-SQNR activations (quant_sensitivity.py)")
    ap.add_argument("--sensitivity", type=Path, default=Path("docs/measurements/lidar_int8_sensitivity.json"))
    ap.add_argument("--moving-average", action="store_true", help="MinMax: moving-average ranges, damping outliers")
    ap.add_argument("--percentile", type=float, default=99.999)
    args = ap.parse_args()
    exclude = list(args.exclude)
    if args.exclude_worst:
        ranked = json.loads(args.sensitivity.read_text())["activations"]
        exclude += [n for r in ranked[:args.exclude_worst] for n in r["conv_nodes"] if n not in exclude]
        print("FP16 (excluded):", exclude)
    pre = args.out.with_suffix(".pre.onnx")
    quant_pre_process(str(args.src), str(pre), skip_symbolic_shape=True)
    quantize_static(str(pre), str(args.out), ValScans(args.calib_scans), quant_format=QuantFormat.QDQ,
                    op_types_to_quantize=["Conv"], nodes_to_exclude=exclude, per_channel=True,
                    activation_type=QuantType.QInt8,
                    weight_type=QuantType.QInt8, calibrate_method=getattr(CalibrationMethod, args.method),
                    extra_options={"ActivationSymmetric": True, "WeightSymmetric": True,
                                   "QuantizeBias": False,  # TensorRT 8.6 parses INT8 Q/DQ only, not INT32 bias
                                   # SalsaNext keeps ~1-2 GB of full-resolution activations per scan; flush the
                                   # calibration statistics every few scans instead of holding all of them.
                                   "CalibMaxIntermediateOutputs": args.flush_every,
                                   "CalibMovingAverage": args.moving_average,
                                   "CalibPercentile": args.percentile})
    pre.unlink(missing_ok=True)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
