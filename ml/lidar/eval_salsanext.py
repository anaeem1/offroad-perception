"""Evaluate the RELLIS-3D-released SalsaNext checkpoint on a lidar split; write per-class IoU as JSON.

Scores every scan twice from one forward pass: CNN output unprojected directly to points, and with the
upstream KNN post-processing (which the published RELLIS-3D number used and the Jetson runtime may not).

  source env/activate.sh
  python -m ml.lidar.eval_salsanext --split test --out docs/measurements/lidar_salsanext_test.json
"""
import argparse
import datetime
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from ml.lidar.knn import KNN
from ml.lidar.range_projection import SensorConfig, project
from ml.lidar.salsanext import SalsaNext
from ml.metrics import ConfusionMatrix

DATA_ROOT = Path(os.environ.get("OFFROAD_DATA_ROOT", "/usace_share/Share_Abdullah/offroad_perception"))
RUN_DIR = DATA_ROOT / "weights/salsanext_rellis/2020-10-14-17:37rellis"
PUBLISHED_MIOU = 40.20  # RELLIS-3D README benchmark table


class RellisLidar(Dataset):
    def __init__(self, split, cfg: SensorConfig, learning_map, limit=None):
        lst = DATA_ROOT / "rellis3d/meta" / f"pt_{split}.lst"
        self.items = [line.split() for line in open(lst) if line.strip()][:limit]
        self.base = DATA_ROOT / "rellis3d/Rellis-3D"
        self.cfg = cfg
        self.lut = np.zeros(1 << 16, np.int64)  # raw ids not in the map fall to 0 (void, ignored)
        for raw, train_id in learning_map.items():
            self.lut[raw] = train_id

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        scan_rel, label_rel = self.items[i]
        scan = np.fromfile(self.base / scan_rel, np.float32).reshape(-1, 4)
        raw = np.fromfile(self.base / label_rel, np.uint32) & 0xFFFF
        assert len(raw) == len(scan), f"{scan_rel}: {len(scan)} points vs {len(raw)} labels"
        proj, proj_range, px, py, depth = project(scan[:, :3].copy(), scan[:, 3].copy(), self.cfg)
        return (torch.from_numpy(proj), torch.from_numpy(proj_range), torch.from_numpy(px).long(),
                torch.from_numpy(py).long(), torch.from_numpy(depth), torch.from_numpy(self.lut[raw]))


def load_model(ckpt_path, n_classes):
    model = SalsaNext(n_classes)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
    state = {k.removeprefix("module."): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    return model


def summarize(cm: ConfusionMatrix, names):
    out = {}
    for conv in ("upstream", "strict"):
        iou = cm.iou(conv)
        out[conv] = {"per_class_iou": {names[c]: round(100 * v, 2) for c, v in iou.items()},
                     "miou": round(100 * float(np.mean(list(iou.values()))), 2),
                     # The published table averages its 14 class IoUs over 15 entries; reproduce that too.
                     "miou_published_convention": round(100 * sum(iou.values()) / (len(iou) + 1), 2)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--ckpt", default="SalsaNext_valid_best")
    ap.add_argument("--limit", type=int, default=None, help="evaluate only the first N scans (smoke test)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cuda", help="cpu only for smoke tests")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    arch = yaml.safe_load(open(RUN_DIR / "arch_cfg.yaml"))
    data = yaml.safe_load(open(RUN_DIR / "data_cfg.yaml"))
    n_classes = len(data["learning_map_inv"])
    names = {i: data["labels"][raw] for i, raw in data["learning_map_inv"].items()}
    cfg = SensorConfig.from_arch_cfg(RUN_DIR / "arch_cfg.yaml")

    ds = RellisLidar(args.split, cfg, data["learning_map"], args.limit)
    device = torch.device(args.device)
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    loader = DataLoader(ds, batch_size=1, num_workers=args.workers, pin_memory=True)
    model = load_model(RUN_DIR / args.ckpt, n_classes).to(device).eval()
    knn = KNN(arch["post"]["KNN"]["params"], n_classes)
    cm_cnn, cm_knn = ConfusionMatrix(n_classes, device=device), ConfusionMatrix(n_classes, device=device)
    t_cnn, t_knn = [], []

    with torch.no_grad():
        for proj, proj_range, px, py, depth, gt in loader:
            proj, proj_range, px, py, depth, gt = (t.to(device, non_blocking=True)[0] for t in
                                                   (proj, proj_range, px, py, depth, gt))
            sync(); t0 = time.perf_counter()
            argmax = model(proj[None])[0].argmax(0)
            sync(); t1 = time.perf_counter()
            pred_knn = knn(proj_range, depth, argmax, px, py)
            sync(); t2 = time.perf_counter()
            cm_cnn.update(argmax[py, px], gt)
            cm_knn.update(pred_knn, gt)
            t_cnn.append(1e3 * (t1 - t0)); t_knn.append(1e3 * (t2 - t1))

    result = {
        "what": "SalsaNext, RELLIS-3D released checkpoint, Ouster OS1-64 SemanticKITTI-format scans",
        "checkpoint": f"{RUN_DIR.name}/{args.ckpt}",
        "split": f"pt_{args.split}.lst", "n_scans": len(ds), "date": datetime.date.today().isoformat(),
        "classes_scored": len(names) - 1, "ignored": ["void (incl. dirt, sky, object, asphalt, building)"],
        "published_miou": PUBLISHED_MIOU,
        "cnn_only": summarize(cm_cnn, names),
        "cnn_plus_knn": summarize(cm_knn, names),
        "support_points": {names[c]: n for c, n in cm_cnn.support().items()},
        "timing_ms": {"device": torch.cuda.get_device_name() if device.type == "cuda" else "cpu", "note": "server-side PyTorch FP32, not a Jetson number",
                      "cnn_median": round(float(np.median(t_cnn)), 2), "knn_median": round(float(np.median(t_knn)), 2)},
        "env": {"torch": torch.__version__},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    for k in ("cnn_only", "cnn_plus_knn"):
        print(k, {c: result[k][c]["miou"] for c in ("upstream", "strict")})
    print("wrote", args.out)


if __name__ == "__main__":
    main()
