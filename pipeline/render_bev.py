"""Render a saved BEV grid from a replay run: predicted cost | ground-truth cost | predicted classes.

  python -m pipeline.render_bev --run <data root>/runs/replay_00001_test --scan 500 --out docs/figures/bev_grid.png
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from pipeline.rellis import DATA_ROOT

UNKNOWN_RGB = (40, 40, 48)


def cost_rgb(cost):
    """0 free -> green, 254 lethal -> red, 255 unknown -> dark grey."""
    c = cost.astype(np.float32) / 254.0
    rgb = np.stack([255 * np.clip(2 * c, 0, 1), 255 * np.clip(2 - 2 * c, 0, 1), np.full_like(c, 40)], -1)
    rgb[cost == 255] = UNKNOWN_RGB
    return rgb.astype(np.uint8)


def class_rgb(cls, flags):
    colors = yaml.safe_load(open(DATA_ROOT / "rellis3d/meta/Rellis_3D_ontology/ontology.yaml"))[1]
    lut = np.zeros((256, 3), np.uint8)
    for k, v in colors.items():
        lut[int(k)] = v
    rgb = lut[cls]
    rgb[flags == 0] = UNKNOWN_RGB
    return rgb


def panel(rgb, title, scale):
    img = cv2.resize(rgb[::-1], None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)  # x right, y up
    img = np.ascontiguousarray(cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE))  # forward (+x) points up
    c = img.shape[0] // 2
    cv2.drawMarker(img, (c, c), (255, 255, 255), cv2.MARKER_TRIANGLE_UP, 14, 2)  # base_link, facing forward
    bar = np.full((44, img.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, title, (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    return np.vstack([bar, img])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--scan", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    grids = np.load(args.run / "grids.npz")
    key = min(grids.files, key=lambda k: abs(int(k) - args.scan))
    cost, sem, flags, cost_gt, _ = grids[key]
    metrics = json.loads((args.run / "metrics.json").read_text())
    extent, mem = metrics["grid"]["extent_m"], metrics.get("memory_max_age_s", 0)
    scale = max(1, 600 // cost.shape[0])
    pred_title = f"predicted cost ({mem:g} s memory)" if mem else "predicted cost"
    row = np.hstack([panel(cost_rgb(cost), pred_title, scale), np.full((44 + cost.shape[0] * scale, 8, 3), 255, np.uint8),
                     panel(cost_rgb(cost_gt), "ground truth (this sweep only)", scale), np.full((44 + cost.shape[0] * scale, 8, 3), 255, np.uint8),
                     panel(class_rgb(sem, flags), "predicted class", scale)])
    foot = np.full((40, row.shape[1], 3), 255, np.uint8)
    cv2.putText(foot, f"scan {key}, {extent:.0f} m x {extent:.0f} m, forward = up. green free -> red lethal, grey unknown",
                (8, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), cv2.cvtColor(np.vstack([row, foot]), cv2.COLOR_RGB2BGR))
    print("wrote", args.out, "from scan", key)


if __name__ == "__main__":
    main()
