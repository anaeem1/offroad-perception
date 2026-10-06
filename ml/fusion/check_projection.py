"""Calibration sanity check: project lidar onto camera images, save an overlay figure, and measure
lidar-label vs image-label agreement at the projected pixels for the nominal extrinsic and for small
rotations of it. If the extrinsic is right, agreement peaks at zero offset.

  python -m ml.fusion.check_projection --figure-frame 00004/000104 \
      --fig docs/figures/projection_overlay.jpg --out docs/measurements/projection_check.json
"""
import argparse
import datetime
import json
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

from ml.fusion.projection import DATA_ROOT, load_rellis_calib, perturb, project_points

BASE = DATA_ROOT / "rellis3d/Rellis-3D"
OFFSETS_DEG = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]


def annotated_frames():
    """{seq: {frame_index: (image_rel, image_label_rel)}} for frames with an image label and a lidar scan+label."""
    frames = defaultdict(dict)
    for split in ("train", "val", "test"):
        for line in open(DATA_ROOT / f"rellis3d/meta/{split}.lst"):
            if not line.strip():
                continue
            img, lab = line.split()
            seq = img.split("/")[0]
            idx = int(re.search(r"frame(\d+)-", img).group(1))
            if (BASE / seq / "os1_cloud_node_kitti_bin" / f"{idx:06d}.bin").exists() and \
               (BASE / seq / "os1_cloud_node_semantickitti_label_id" / f"{idx:06d}.label").exists():
                frames[seq][idx] = (img, lab)
    return frames


def load_frame(seq, idx, img_rel, lab_rel):
    scan = np.fromfile(BASE / seq / "os1_cloud_node_kitti_bin" / f"{idx:06d}.bin", np.float32).reshape(-1, 4)
    lid = (np.fromfile(BASE / seq / "os1_cloud_node_semantickitti_label_id" / f"{idx:06d}.label", np.uint32)
           & 0xFFFF).astype(np.int32)
    img = cv2.imread(str(BASE / img_rel), cv2.IMREAD_COLOR)
    imlab = cv2.imread(str(BASE / lab_rel), cv2.IMREAD_GRAYSCALE)
    return scan[:, :3], lid, img, imlab


def agreement(xyz, lid, imlab, cam, T):
    uv, idx, _ = project_points(xyz, cam, T)
    u, v = uv[:, 0].astype(int), uv[:, 1].astype(int)
    a, b = lid[idx], imlab[v, u].astype(np.int32)
    valid = (a != 0) & (b != 0)
    return int((a[valid] == b[valid]).sum()), int(valid.sum())


def render(xyz, lid, img, cam, T, colors):
    uv, idx, depth = project_points(xyz, cam, T)
    gray = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    by_range, by_class = gray.copy(), gray.copy()
    rng = cv2.applyColorMap((np.clip(depth / 40.0, 0, 1) * 255).astype(np.uint8)[:, None],
                            cv2.COLORMAP_TURBO)[:, 0]
    for (x, y), c_r, cls in zip(uv.astype(int), rng, lid[idx]):
        cv2.circle(by_range, (x, y), 3, tuple(int(c) for c in c_r), -1)
        if cls != 0:
            r, g, b = colors.get(int(cls), (255, 255, 255))
            cv2.circle(by_class, (x, y), 3, (b, g, r), -1)
    for panel, text in ((by_range, "lidar range, 0-40 m"), (by_class, "lidar ground-truth class")):
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 4)
        cv2.rectangle(panel, (0, 0), (tw + 60, th + 60), (0, 0, 0), -1)
        cv2.putText(panel, text, (30, th + 30), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 4, cv2.LINE_AA)
    out = np.vstack([by_range, by_class])
    return cv2.resize(out, (1280, int(out.shape[0] * 1280 / out.shape[1])), interpolation=cv2.INTER_AREA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figure-frame", default="00004/000104", help="seq/scan_index for the README figure")
    ap.add_argument("--frames-per-seq", type=int, default=40)
    ap.add_argument("--fig", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    ontology = yaml.safe_load(open(DATA_ROOT / "rellis3d/meta/Rellis_3D_ontology/ontology.yaml"))
    colors = {int(k): tuple(v) for k, v in ontology[1].items()}
    frames = annotated_frames()

    seq, idx = args.figure_frame.split("/")
    idx = int(idx)
    if idx not in frames[seq]:  # fall back to the nearest annotated frame in that sequence
        idx = min(frames[seq], key=lambda k: abs(k - idx))
    cam, T = load_rellis_calib(seq)
    xyz, lid, img, _ = load_frame(seq, idx, *frames[seq][idx])
    args.fig.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.fig), render(xyz, lid, img, cam, T, colors), [cv2.IMWRITE_JPEG_QUALITY, 88])

    results = {}
    for s in sorted(frames):
        cam, T = load_rellis_calib(s)
        keys = sorted(frames[s])
        sample = [keys[i] for i in np.linspace(0, len(keys) - 1, min(args.frames_per_seq, len(keys))).astype(int)]
        tallies = {f"{axis}{off:+.1f}": [0, 0] for axis in ("yaw", "pitch") for off in OFFSETS_DEG}
        for k in sample:
            xyz, lid, _, imlab = load_frame(s, k, *frames[s][k])
            for axis in ("yaw", "pitch"):
                for off in OFFSETS_DEG:
                    Tp = perturb(T, **{f"{axis}_deg": off})
                    agree, n = agreement(xyz, lid, imlab, cam, Tp)
                    tallies[f"{axis}{off:+.1f}"][0] += agree
                    tallies[f"{axis}{off:+.1f}"][1] += n
        curve = {k: round(100 * a / max(n, 1), 2) for k, (a, n) in tallies.items()}
        best = {axis: max(OFFSETS_DEG, key=lambda o: curve[f"{axis}{o:+.1f}"]) for axis in ("yaw", "pitch")}
        results[s] = {"frames": len(sample), "labeled_point_pairs_nominal": tallies["yaw+0.0"][1],
                      "agreement_pct": curve, "best_offset_deg": best,
                      "nominal_is_peak": all(v == 0.0 for v in best.values())}
        print(s, "nominal", curve["yaw+0.0"], "best", best)

    out = {"what": "lidar GT class == image GT class at projected pixel (both non-void), per extrinsic offset",
           "date": datetime.date.today().isoformat(), "figure": str(args.fig), "figure_frame": f"{seq}/{idx:06d}",
           "distortion_source": "RELLIS-3D utils/lidar2img.ipynb", "sequences": results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print("wrote", args.fig, args.out)


if __name__ == "__main__":
    main()
