"""Score an exported SegFormer ONNX model (the deployment artifact) on a RELLIS-3D image split.
Used to compare input resolutions on equal terms: same script, same scoring at full label resolution.

  python -m ml.camera.eval_segformer_onnx --onnx <data root>/onnx/segformer_b0_480x768_op17.onnx \
      --out docs/measurements/camera_onnx_480x768_test.json
"""
import argparse
import datetime
import json
import os
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F

from ml.camera.train_segformer import IGNORE, RellisImages, class_table, summarize
from ml.metrics import ConfusionMatrix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = int(os.environ.get("SLURM_CPUS_PER_TASK", "0"))
    sess = ort.InferenceSession(str(args.onnx), sess_options=opts, providers=["CPUExecutionProvider"])
    _, _, H, W = sess.get_inputs()[0].shape
    ids, names = class_table()
    n = len(ids)
    ds = RellisImages(args.split, ids, (H, W), limit=args.limit)
    cm = ConfusionMatrix(n + 1, ignore=(n,), device="cpu")
    for img_rel, lab_rel in ds.items:
        rgb = cv2.cvtColor(cv2.imread(str(ds.base / img_rel)), cv2.COLOR_BGR2RGB)
        x = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1)[None].astype(np.float32)
        probs = torch.from_numpy(sess.run(None, {"input": x})[0])
        lab = torch.from_numpy(ds.lut[cv2.imread(str(ds.base / lab_rel), cv2.IMREAD_GRAYSCALE)].astype(np.int64))
        pred = F.interpolate(probs, size=lab.shape, mode="bilinear", align_corners=False).argmax(1)[0]
        cm.update(pred, torch.where(lab == IGNORE, n, lab))
    out = {"onnx": str(args.onnx), "input_hw": [H, W], "split": args.split, "n_images": len(ds),
           "date": datetime.date.today().isoformat(), "scored_at": "full label resolution, probs upsampled",
           "test": summarize(cm, names)}
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(args.onnx.name, "mIoU", out["test"]["miou"])


if __name__ == "__main__":
    main()
