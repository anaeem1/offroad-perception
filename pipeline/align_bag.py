"""Align a converted RELLIS-3D bag with the labelled dataset, in one pass over the bag:
  - labelled scan k  <-> bag lidar message, by exact hash of the xyz bytes (point order is identical)
  - dataset image k  <-> bag camera message, by header stamp (filenames carry the stamp in ms)
  - debayer check: which OpenCV Bayer code reproduces the dataset JPEGs from bayer_rggb8

  python -m pipeline.align_bag --seq 00001
Writes <data root>/rellis3d/alignment/<seq>.json (full maps) and docs/measurements/alignment_<seq>.json.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from rosbags.highlevel import AnyReader

from pipeline.rellis import (ALIGN_DIR, BAGS, CAMERA_TOPIC, LIDAR_TOPIC, decode_bayer, decode_cloud,
                             image_frames, scan_paths, stamp_of)

BAYER_CODES = {"BayerRG2RGB": cv2.COLOR_BayerRG2RGB, "BayerBG2RGB": cv2.COLOR_BayerBG2RGB,
               "BayerGR2RGB": cv2.COLOR_BayerGR2RGB, "BayerGB2RGB": cv2.COLOR_BayerGB2RGB}


def xyz_hash(xyz):
    return hashlib.blake2b(np.ascontiguousarray(xyz, np.float32).tobytes(), digest_size=16).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True, choices=sorted(BAGS))
    ap.add_argument("--debayer-samples", type=int, default=8)
    args = ap.parse_args()

    bins = sorted(int(p.stem) for p in scan_paths(args.seq, 0)[0].parent.glob("*.bin"))
    bin_hash = {xyz_hash(np.fromfile(scan_paths(args.seq, k)[0], np.float32).reshape(-1, 4)[:, :3]): k for k in bins}
    frames = image_frames(args.seq)
    by_ms = {round(s * 1000): k for k, (s, _) in frames.items()}  # filenames carry the stamp truncated to ms
    probe = set(sorted(frames)[i] for i in np.linspace(0, len(frames) - 1, args.debayer_samples).astype(int))

    lidar, camera, raw_for_debayer = [], [], {}
    with AnyReader([BAGS[args.seq]]) as reader:
        conns = [c for c in reader.connections if c.topic in (LIDAR_TOPIC, CAMERA_TOPIC)]
        n_l = n_c = 0
        for conn, _, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)
            if conn.topic == LIDAR_TOPIC:
                k = bin_hash.get(xyz_hash(decode_cloud(msg)["xyz"]))
                lidar.append({"msg": n_l, "stamp": stamp_of(msg), "scan": k})
                n_l += 1
            else:
                st = stamp_of(msg)
                k = by_ms.get(int(np.floor(st * 1000 + 1e-6)))
                camera.append({"msg": n_c, "stamp": st, "frame": k})
                if k in probe:
                    raw_for_debayer[k] = decode_bayer(msg).copy()
                n_c += 1

    # Debayer: mean absolute error against the dataset JPEG for each OpenCV code.
    err = {name: [] for name in BAYER_CODES}
    for k, mosaic in raw_for_debayer.items():
        ref = cv2.cvtColor(cv2.imread(str(frames[k][1])), cv2.COLOR_BGR2RGB).astype(np.float32)
        for name, code in BAYER_CODES.items():
            err[name].append(float(np.abs(cv2.cvtColor(mosaic, code).astype(np.float32) - ref).mean()))
    debayer = {name: round(float(np.mean(v)), 2) for name, v in err.items()}

    matched = [r for r in lidar if r["scan"] is not None]
    offsets = sorted({r["msg"] - r["scan"] for r in matched})
    cam_matched = [r for r in camera if r["frame"] is not None]
    ALIGN_DIR.mkdir(parents=True, exist_ok=True)
    (ALIGN_DIR / f"{args.seq}.json").write_text(json.dumps({"lidar": lidar, "camera": camera}) + "\n")
    summary = {
        "seq": args.seq, "bag": BAGS[args.seq].name, "date": datetime.date.today().isoformat(),
        "lidar": {"bag_messages": len(lidar), "labelled_scans": len(bins), "matched_by_xyz_hash": len(matched),
                  "unmatched_labelled_scans": sorted(set(bins) - {r["scan"] for r in matched}),
                  "msg_minus_scan_offsets": offsets},
        "camera": {"bag_messages": len(camera), "dataset_images": len(frames), "matched_by_stamp_ms": len(cam_matched)},
        "debayer_mean_abs_error_vs_dataset_jpeg": debayer, "debayer_samples": len(raw_for_debayer),
        "debayer_best": min(debayer, key=debayer.get) if debayer else None,
        "full_maps": str(ALIGN_DIR / f"{args.seq}.json"),
    }
    out = Path("docs/measurements") / f"alignment_{args.seq}.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
