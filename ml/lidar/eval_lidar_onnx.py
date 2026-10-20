"""Score exported lidar ONNX models (e.g. FP32 vs INT8 QDQ) on the same test scans, CNN-only path (what the
runtime deploys), upstream IoU convention as in eval_salsanext.

  python -m ml.lidar.eval_lidar_onnx --onnx a.onnx b.onnx --every 4 --out docs/measurements/lidar_int8_vs_fp32.json
"""
import argparse
import datetime
import json
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import yaml

from ml.lidar.eval_salsanext import RUN_DIR, RellisLidar
from ml.lidar.range_projection import SensorConfig
from ml.metrics import ConfusionMatrix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=Path, nargs="+", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--every", type=int, default=4)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    data = yaml.safe_load(open(RUN_DIR / "data_cfg.yaml"))
    names = {i: data["labels"][raw] for i, raw in data["learning_map_inv"].items()}
    n_cls = len(names)
    ds = RellisLidar(args.split, SensorConfig.from_arch_cfg(RUN_DIR / "arch_cfg.yaml"), data["learning_map"])
    picks = list(range(0, len(ds), args.every))
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = int(os.environ.get("SLURM_CPUS_PER_TASK", "0"))
    sessions = {p.name: ort.InferenceSession(str(p), sess_options=opts, providers=["CPUExecutionProvider"]) for p in args.onnx}
    cms = {k: ConfusionMatrix(n_cls, device="cpu") for k in sessions}
    agree = {k: [] for k in sessions}
    ref = args.onnx[0].name
    for i in picks:
        proj, _, px, py, _, gt = ds[i]
        labels = {}
        for k, s in sessions.items():
            lab = s.run(["labels"], {"input": proj[None].numpy()})[0][0]
            pred = torch.from_numpy(lab[py.numpy(), px.numpy()].astype(np.int64))
            cms[k].update(pred, gt)
            labels[k] = pred
        for k in sessions:
            agree[k].append(float((labels[k] == labels[ref]).float().mean()))
    out = {"date": datetime.date.today().isoformat(), "split": f"pt_{args.split}.lst", "scans": len(picks),
           "every": args.every, "path": "CNN only (no KNN), upstream IoU convention, 14 classes",
           "models": {}}
    for k, cm in cms.items():
        iou = cm.iou("upstream")
        out["models"][k] = {"miou": round(100 * float(np.mean(list(iou.values()))), 2),
                            "per_class_iou": {names[c]: round(100 * v, 2) for c, v in iou.items()},
                            f"point_label_agreement_with_{ref}": round(float(np.mean(agree[k])), 4)}
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: (v["miou"], v[f"point_label_agreement_with_{ref}"]) for k, v in out["models"].items()}))


if __name__ == "__main__":
    main()
