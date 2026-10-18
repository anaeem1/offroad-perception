"""Regenerate pseudo-ground-truth obstacle clusters for a replay run, with their extents, from the labelled
scans (no models needed). Extents let the tracking evaluation drop flat "objects": RELLIS-3D's map-based
lidar labels mark ground where people walked as `person` (docs/measurements/diag_person_wedge_00001.json).

Point times: the Ouster cloud (and so the .bin, same order) is stored column-major, 64 beams per column
with one firing time per column, so point i fires at stamp + (i // 64) * sweep / 2048.

  python -m pipeline.gt_objects --run <data root>/runs/replay_00001_test_final
"""
import argparse
import json
from pathlib import Path

import numpy as np

from pipeline import fusion as F
from pipeline.bev import GridParams, load_costs, outside_ego
from pipeline.cluster import cluster
from pipeline.extract_odom import load_odom
from pipeline.geometry import T_root_frame, apply, deskew, tf_graph
from pipeline.rellis import load_scan_labels, scan_paths

SWEEP_S, COLUMNS, BEAMS = 0.1, 2048, 64


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    args = ap.parse_args()
    seq = json.loads((args.run / "metrics.json").read_text())["seq"]
    frames = [json.loads(line) for line in open(args.run / "frames.jsonl")]
    odom = load_odom(seq)
    insp = json.loads(Path(f"docs/measurements/bag_{seq}_inspection.json").read_text())
    T_bl = T_root_frame(tf_graph(insp["tf_static"]), insp["pointcloud"]["frame_id"])
    _, obstacle_lut = load_costs()
    gp = GridParams()
    onto_lut = np.array([F.ONTO_INDEX.get(k, 0) for k in range(1 << 16)])
    col_t = np.repeat(np.arange(COLUMNS) * SWEEP_S / COLUMNS, BEAMS)
    out = open(args.run / "gt_objects_v2.jsonl", "w")
    n_obj = 0
    for f in frames:
        scan, stamp = f["scan"], f["t"]
        xyz = np.fromfile(scan_paths(seq, scan)[0], np.float32).reshape(-1, 4)[:, :3]
        gt = load_scan_labels(seq, scan)
        vi = np.flatnonzero(np.abs(xyz).sum(1) > 0)
        base = apply(T_bl, deskew(xyz[vi], stamp + col_t[vi], odom, T_bl, stamp))
        keep = outside_ego(base)
        vi, base = vi[keep], base[keep]
        onehot = np.zeros((len(vi), len(F.ONTOLOGY_IDS)), np.float32)
        onehot[np.arange(len(vi)), onto_lut[gt[vi]]] = 1.0
        T_ob = odom.at(stamp)
        for d in cluster(base, onehot, np.zeros((gp.n, gp.n), bool), gp, obstacle_lut):
            xo = apply(T_ob, d.xyz[None])[0]
            out.write(json.dumps({"scan": scan, "t": stamp, "x": float(xo[0]), "y": float(xo[1]), "cls": d.semantic_class,
                                  "n": d.n_points, "range": float(np.hypot(*d.xyz[:2])),
                                  "dims": [round(float(v), 3) for v in d.dims]}) + "\n")
            n_obj += 1
    print(f"{n_obj} GT objects over {len(frames)} scans -> {args.run / 'gt_objects_v2.jsonl'}")


if __name__ == "__main__":
    main()
