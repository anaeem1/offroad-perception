"""Per-layer INT8 sensitivity of a QDQ SalsaNext: which quantized activations lose the most signal.

For every activation the QDQ model quantizes, the SQNR (dB) of the INT8 round trip alone ("local", the error
this layer adds) and against the float model at the same tensor ("accumulated", everything upstream too);
for every Conv weight, the SQNR of its quantization. The Convs reading the lowest-local-SQNR activations are
the candidates to keep in FP16 (`quantize_salsanext --exclude-worst K`).

  python -m ml.lidar.quant_sensitivity --qdq <data root>/onnx/salsanext_int8_pct.onnx --scans 4 \
      --out docs/measurements/lidar_int8_sensitivity.json
"""
import argparse
import datetime
import json
import os
import tempfile
from pathlib import Path

import onnx
import onnxruntime as ort
from onnxruntime.quantization import quant_pre_process
from onnxruntime.quantization.qdq_loss_debug import (collect_activations, compute_activation_error,
                                                     compute_weight_error, create_activation_matching,
                                                     create_weight_matching,
                                                     modify_model_output_intermediate_tensors)

from ml.lidar.quantize_salsanext import ValScans
from pipeline.models import ONNX_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qdq", type=Path, required=True)
    ap.add_argument("--src", type=Path, default=ONNX_DIR / "salsanext_rellis_op17.onnx")
    ap.add_argument("--scans", type=int, default=4)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = int(os.environ.get("SLURM_CPUS_PER_TASK", "0"))
    with tempfile.TemporaryDirectory() as tmp:
        # The QDQ model was built from the pre-processed graph; compare against the same graph so names match.
        pre, aug_f, aug_q = (str(Path(tmp) / n) for n in ("pre.onnx", "aug_float.onnx", "aug_qdq.onnx"))
        quant_pre_process(str(args.src), pre, skip_symbolic_shape=True)
        modify_model_output_intermediate_tensors(pre, aug_f, op_types_for_saving=["Conv"])
        modify_model_output_intermediate_tensors(str(args.qdq), aug_q, op_types_for_saving=["QuantizeLinear", "DequantizeLinear"])
        float_acts = collect_activations(aug_f, ValScans(args.scans), session_options=opts)
        qdq_acts = collect_activations(aug_q, ValScans(args.scans), session_options=opts)
        act_err = compute_activation_error(create_activation_matching(qdq_acts, float_acts))
        w_err = compute_weight_error(create_weight_matching(pre, str(args.qdq)))
        convs = [n for n in onnx.load(pre).graph.node if n.op_type == "Conv"]
    readers = {t: [n.name for n in convs if n.input[0] == t] for t in act_err}
    ranked = sorted(({"tensor": t, "conv_nodes": readers[t], "local_sqnr_db": round(float(e["qdq_err"]), 2),
                      "accumulated_sqnr_db": round(float(e.get("xmodel_err", float("nan"))), 2),
                      "weight_sqnr_db": [round(float(w_err[n.input[1]]), 2) for n in convs if n.name in readers[t] and n.input[1] in w_err]}
                     for t, e in act_err.items() if readers[t]), key=lambda r: r["local_sqnr_db"])
    out = {"date": datetime.date.today().isoformat(), "qdq_model": args.qdq.name, "float_model": args.src.name,
           "val_scans": args.scans, "metric": "SQNR dB = 10 log10(signal power / error power); lower is worse",
           "activations": ranked}
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    for r in ranked[:10]:
        print(f"{r['local_sqnr_db']:7.2f} {r['accumulated_sqnr_db']:7.2f}  {r['tensor']}  -> {r['conv_nodes']}")


if __name__ == "__main__":
    main()
