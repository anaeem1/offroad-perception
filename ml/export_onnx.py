"""Export the deployment models to ONNX (static shapes, batch 1) and check parity against PyTorch.

  python -m ml.export_onnx lidar  --opset 17
  python -m ml.export_onnx camera --opset 17 --ckpt <run_dir>/best

Each export writes <data root>/onnx/<name>_op<opset>.onnx and docs/measurements/onnx_<name>_op<opset>.json
(file hash, ops, I/O shapes, and ONNX Runtime vs PyTorch agreement on real test samples).

Output contract of both graphs (consumed by the fusion stage):
  probs  float32 (1, C, h, w)  softmax class probabilities
  labels int32   (1, h, w)     argmax of probs
Lidar:  input (1, 5, 64, 2048) normalized range image (ml.lidar.range_projection.project); h, w = 64, 2048.
Camera: input (1, 3, H, W) RGB float in 0..255 (normalization is inside the graph); h, w = H/4, W/4.
"""
import argparse
import collections
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from onnxsim import simplify

from ml.lidar.eval_salsanext import DATA_ROOT, RUN_DIR, RellisLidar, load_model
from ml.lidar.range_projection import SensorConfig

ONNX_DIR = DATA_ROOT / "onnx"


class LidarDeploy(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        probs = self.net(x)  # SalsaNext already ends in softmax
        return probs, probs.argmax(1).to(torch.int32)


class CameraDeploy(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255)

    def forward(self, rgb):
        probs = self.model(pixel_values=(rgb - self.mean) / self.std).logits.softmax(1)
        return probs, probs.argmax(1).to(torch.int32)


def lidar_samples(n):
    import yaml
    data = yaml.safe_load(open(RUN_DIR / "data_cfg.yaml"))
    ds = RellisLidar("test", SensorConfig.from_arch_cfg(RUN_DIR / "arch_cfg.yaml"), data["learning_map"])
    picks = np.linspace(0, len(ds) - 1, n).astype(int)
    return [ds[int(i)][0][None].numpy() for i in picks], [ds.items[int(i)][0] for i in picks]


def camera_samples(n, size_hw):
    import cv2
    from ml.camera.train_segformer import RellisImages, class_table
    ds = RellisImages("test", class_table()[0], size_hw)
    picks = np.linspace(0, len(ds) - 1, n).astype(int)
    out = []
    for i in picks:
        img = cv2.cvtColor(cv2.imread(str(ds.base / ds.items[int(i)][0])), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (size_hw[1], size_hw[0]), interpolation=cv2.INTER_LINEAR)
        out.append(img.transpose(2, 0, 1)[None].astype(np.float32))
    return out, [ds.items[int(i)][0] for i in picks]


def export(module, example, path, opset):
    module.eval()
    torch.onnx.export(module, example, path, opset_version=opset, input_names=["input"],
                      output_names=["probs", "labels"], dynamo=False)
    model, ok = simplify(onnx.load(path))
    assert ok, "onnxsim could not validate the simplified model"
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return model


def parity(module, path, samples):
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    max_diff, agree = [], []
    with torch.no_grad():
        for x in samples:
            p_ref, l_ref = (t.numpy() for t in module(torch.from_numpy(x)))
            p, l = sess.run(None, {"input": x})
            max_diff.append(float(np.abs(p - p_ref).max()))
            agree.append(float((l == l_ref).mean()))
    return {"samples": len(samples), "max_abs_prob_diff": max(max_diff), "min_label_agreement": min(agree)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=["lidar", "camera"])
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--ckpt", type=Path, help="camera: fine-tuned HF checkpoint dir")
    ap.add_argument("--size", type=int, nargs=2, default=[640, 1024], metavar=("H", "W"))
    ap.add_argument("--samples", type=int, default=3)
    args = ap.parse_args()
    torch.set_grad_enabled(False)

    if args.model == "lidar":
        name, source = "salsanext_rellis", f"{RUN_DIR.name}/SalsaNext_valid_best"
        module = LidarDeploy(load_model(RUN_DIR / "SalsaNext_valid_best", 15))
        samples, sample_ids = lidar_samples(args.samples)
    else:
        from transformers import SegformerForSemanticSegmentation
        assert args.ckpt, "--ckpt is required for the camera model"
        name, source = f"segformer_b0_{args.size[0]}x{args.size[1]}", str(args.ckpt)
        module = CameraDeploy(SegformerForSemanticSegmentation.from_pretrained(args.ckpt))
        samples, sample_ids = camera_samples(args.samples, args.size)

    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    path = ONNX_DIR / f"{name}_op{args.opset}.onnx"
    model = export(module.eval(), torch.from_numpy(samples[0]), path, args.opset)
    shapes = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim] for v in [*model.graph.input, *model.graph.output]}
    report = {
        "model": name, "source_checkpoint": source, "date": datetime.date.today().isoformat(),
        "onnx_file": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_mb": round(path.stat().st_size / 2**20, 2), "opset": args.opset, "io_shapes": shapes,
        "ops": dict(sorted(collections.Counter(n.op_type for n in model.graph.node).items())),
        "parity_vs_pytorch_cpu": {**parity(module, path, samples), "sample_ids": sample_ids},
        "env": {"torch": torch.__version__, "onnx": onnx.__version__, "onnxruntime": ort.__version__},
    }
    out = Path("docs/measurements") / f"onnx_{name}_op{args.opset}.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("onnx_file", "size_mb", "io_shapes", "parity_vs_pytorch_cpu")}, indent=1))


if __name__ == "__main__":
    main()
