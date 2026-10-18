"""Measure the ego vehicle's own footprint in lidar data: while the machine moves, the world slides through
base_link, but returns from the vehicle itself stay put. Cells occupied in most moving scans = self hits.

  python -m pipeline.ego_footprint --seq 00001 --scans 1047:2319 --out docs/measurements/ego_footprint_00001.json
"""
import argparse
import datetime
import json
from pathlib import Path

import numpy as np

from pipeline.extract_odom import load_odom
from pipeline.geometry import T_root_frame, apply, tf_graph
from pipeline.rellis import ALIGN_DIR, load_scan_labels, scan_paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--scans", required=True)
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--half", type=float, default=3.0, help="half-size of the examined square around base_link (m)")
    ap.add_argument("--res", type=float, default=0.1)
    ap.add_argument("--persist", type=float, default=0.5, help="fraction of moving scans a self cell must be hit in")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    lo, hi = (int(v) for v in args.scans.split(":"))
    stamps = {r["scan"]: r["stamp"] for r in json.loads((ALIGN_DIR / f"{args.seq}.json").read_text())["lidar"]
              if r["scan"] is not None}
    insp = json.loads(Path(f"docs/measurements/bag_{args.seq}_inspection.json").read_text())
    T_bl = T_root_frame(tf_graph(insp["tf_static"]), insp["pointcloud"]["frame_id"])
    odom = load_odom(args.seq)
    n = int(2 * args.half / args.res)
    hits = np.zeros((n, n))
    zmax = np.full((n, n), -np.inf)
    labels = {}
    used = 0
    for k in range(lo, hi, args.every):
        t = stamps[k]
        v = np.linalg.norm(odom.at(t + 0.1)[:2, 3] - odom.at(t - 0.1)[:2, 3]) / 0.2
        if v < 0.5:
            continue
        xyz = np.fromfile(scan_paths(args.seq, k)[0], np.float32).reshape(-1, 4)[:, :3]
        ok = np.abs(xyz).sum(1) > 0
        b = apply(T_bl, xyz[ok].astype(np.float64))
        gt = load_scan_labels(args.seq, k)[ok]
        ij = np.floor((b[:, :2] + args.half) / args.res).astype(int)
        inside = (ij >= 0).all(1) & (ij < n).all(1)
        occ = np.zeros((n, n), bool)
        occ[ij[inside, 1], ij[inside, 0]] = True
        np.maximum.at(zmax, (ij[inside, 1], ij[inside, 0]), b[inside, 2])
        hits += occ
        for c in np.unique(gt[inside]):
            labels[int(c)] = labels.get(int(c), 0) + int((gt[inside] == c).sum())
        used += 1
    frac = hits / max(used, 1)
    self_cells = frac >= args.persist
    jj, ii = np.nonzero(self_cells)
    xs = (ii + 0.5) * args.res - args.half
    ys = (jj + 0.5) * args.res - args.half
    box = {"x_min": round(float(xs.min() - args.res / 2), 2), "x_max": round(float(xs.max() + args.res / 2), 2),
           "y_min": round(float(ys.min() - args.res / 2), 2), "y_max": round(float(ys.max() + args.res / 2), 2),
           "z_max": round(float(zmax[self_cells].max()), 2)} if len(xs) else None
    out = {"seq": args.seq, "scans": args.scans, "moving_scans_used": used, "date": datetime.date.today().isoformat(),
           "cell_m": args.res, "persistence_threshold": args.persist, "self_cells": int(self_cells.sum()),
           "self_box_base_link": box,
           "gt_labels_near_vehicle_points": labels}
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
