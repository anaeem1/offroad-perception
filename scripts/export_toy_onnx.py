"""Export small ONNX models that exercise the ops our real models need, at opset 16 and 17.

Run on the server; copy the outputs to the Jetson and run scripts/jetson/check_trt.sh.
Whichever opset parses and builds on the Jetson's TensorRT is the one we export the real models at.

  toy_camera_*: Conv, LayerNorm, GELU(erf), MatMul attention, bilinear Resize, ArgMax  (SegFormer)
  toy_lidar_*:  Conv, BatchNorm, LeakyReLU, AvgPool, PixelShuffle(DepthToSpace), ArgMax (SalsaNext)
"""
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class ToyCamera(nn.Module):
    def __init__(self, c=32, n_cls=20):
        super().__init__()
        self.embed = nn.Conv2d(3, c, 7, stride=4, padding=3)
        self.norm = nn.LayerNorm(c)
        self.qkv = nn.Linear(c, 3 * c)
        self.mlp = nn.Sequential(nn.Linear(c, 2 * c), nn.GELU(), nn.Linear(2 * c, c))
        self.head = nn.Conv2d(c, n_cls, 1)

    def forward(self, x):
        h, w = x.shape[-2:]
        f = self.embed(x)
        b, c, fh, fw = f.shape
        t = f.flatten(2).transpose(1, 2)
        q, k, v = self.qkv(self.norm(t)).chunk(3, dim=-1)
        t = t + torch.softmax(q @ k.transpose(1, 2) / c**0.5, dim=-1) @ v
        t = t + self.mlp(self.norm(t))
        f = t.transpose(1, 2).reshape(b, c, fh, fw)
        logits = F.interpolate(self.head(f), size=(h, w), mode="bilinear", align_corners=False)
        return logits.argmax(dim=1).to(torch.int32)


class ToyLidar(nn.Module):
    def __init__(self, c=32, n_cls=20):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(5, c, 3, padding=1), nn.BatchNorm2d(c), nn.LeakyReLU(0.01),
                                 nn.AvgPool2d(2))
        self.up = nn.PixelShuffle(2)
        self.head = nn.Conv2d(c // 4, n_cls, 1)

    def forward(self, x):
        return self.head(self.up(self.enc(x))).argmax(dim=1).to(torch.int32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--opsets", type=int, nargs="+", default=[16, 17])
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cases = {"camera": (ToyCamera(), torch.randn(1, 3, 256, 384)),
             "lidar": (ToyLidar(), torch.randn(1, 5, 64, 2048))}
    for name, (model, x) in cases.items():
        model.eval()
        for opset in args.opsets:
            path = args.out / f"toy_{name}_op{opset}.onnx"
            torch.onnx.export(model, x, path, opset_version=opset, input_names=["input"],
                              output_names=["labels"], dynamo=False)
            print("wrote", path)


if __name__ == "__main__":
    main()
