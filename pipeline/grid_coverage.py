"""How much of a BEV grid can the lidar actually observe? Fraction of cells with >= 1 return per range
band, per resolution, for a single scan and for the last K scans accumulated with odometry.
Decides grid extent/resolution and whether temporal accumulation is needed (val block only).

  python -m pipeline.grid_coverage --seq 00001 --scans 1047:2319 --out docs/measurements/grid_coverage_00001.json
"""
import argparse
import datetime
import json
from pathlib import Path

import numpy as np

from pipeline.extract_odom import load_odom
from pipeline.geometry import T_root_frame, apply, inv, tf_graph
from pipeline.rellis import ALIGN_DIR, scan_paths

BANDS = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 30)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--scans", required=True)
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--resolutions", type=float, nargs="+", default=[0.1, 0.2, 0.4])
    ap.add_argument("--accumulate", type=int, nargs="+", default=[1, 5, 10])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    lo, hi = (int(v) for v in args.scans.split(":"))
    stamps = {r["scan"]: r["stamp"] for r in json.loads((ALIGN_DIR / f"{args.seq}.json").read_text())["lidar"]
              if r["scan"] is not None}
    insp = json.loads(Path(f"docs/measurements/bag_{args.seq}_inspection.json").read_text())
    T_bl = T_root_frame(tf_graph(insp["tf_static"]), insp["pointcloud"]["frame_id"])
    odom = load_odom(args.seq)
    extent = 2 * max(b[1] for b in BANDS)

    def points_in_base_of(k, ref):
        xyz = np.fromfile(scan_paths(args.seq, k)[0], np.float32).reshape(-1, 4)[:, :3]
        xyz = xyz[np.abs(xyz).sum(1) > 0]
        T = inv(odom.at(stamps[ref])) @ odom.at(stamps[k]) @ T_bl  # base@ref <- lidar@k
        return apply(T, xyz)

    acc = {(r, K): np.zeros((len(BANDS), 2)) for r in args.resolutions for K in args.accumulate}
    for ref in range(lo + max(args.accumulate), hi, args.every):
        clouds = [points_in_base_of(k, ref) for k in range(ref - max(args.accumulate) + 1, ref + 1)]
        for K in args.accumulate:
            pts = np.concatenate(clouds[-K:])
            pts = pts[(pts[:, 2] > -3) & (pts[:, 2] < 3)]
            for r in args.resolutions:
                n = int(extent / r)
                ij = np.floor((pts[:, :2] + extent / 2) / r).astype(int)
                ok = (ij >= 0).all(1) & (ij < n).all(1)
                occ = np.zeros((n, n), bool)
                occ[ij[ok, 1], ij[ok, 0]] = True
                c = (np.arange(n) + 0.5) * r - extent / 2
                dist = np.hypot(*np.meshgrid(c, c))
                for b, (a, z) in enumerate(BANDS):
                    band = (dist >= a) & (dist < z)
                    acc[(r, K)][b] += (occ[band].sum(), band.sum())

    out = {"seq": args.seq, "scans": args.scans, "every": args.every, "date": datetime.date.today().isoformat(),
           "what": "fraction of BEV cells with >= 1 lidar return, by range band from base_link",
           "observed_fraction": {f"res={r}|scans={K}": {f"{a}-{z}m": round(float(v[b, 0] / v[b, 1]), 3)
                                                        for b, (a, z) in enumerate(BANDS)}
                                 for (r, K), v in acc.items()}}
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    for k, v in out["observed_fraction"].items():
        print(f"{k:16s}", "  ".join(f"{b}:{f:.2f}" for b, f in v.items()))


if __name__ == "__main__":
    main()
