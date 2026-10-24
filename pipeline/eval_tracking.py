"""Tracking metrics (CLEAR-MOT / IDF1) for a replay run, against pseudo-ground-truth tracks.

RELLIS-3D has no box or track labels (ADR-0001). Pseudo-GT: obstacle clusters computed from the
ground-truth point labels (same clustering code), linked across frames in the odom frame by a simple
nearest-neighbour linker (same class, <= link_m, gaps up to max_gap frames). This is deliberately not
our Kalman tracker, so the reference does not share its biases. What this measures: how well tracks built
from *predicted* semantics follow objects that perfect semantics would produce. It does not measure
absolute tracker correctness, and odometry drift affects both sides equally.

  python -m pipeline.eval_tracking --run <data root>/runs/replay_00001_test --out docs/measurements/tracking_00001_test.json
"""
import argparse
import collections
import datetime
import json
from pathlib import Path

import motmetrics as mm
import numpy as np
from scipy.optimize import linear_sum_assignment

ONTO_NAMES = {5: "pole", 8: "vehicle", 9: "object", 15: "log", 17: "person", 27: "barrier", 0: "generic"}


def ego_xy(seq):
    from pipeline.extract_odom import load_odom
    odom = load_odom(seq)
    return lambda t: odom.at(t)[:2, 3]


def read_jsonl(path):
    return [json.loads(line) for line in open(path)]


def link_gt(gt_rows, link_m=1.0, max_gap=5):
    """Assign pseudo-GT ids to per-frame GT clusters. Returns {scan: [(id, x, y, cls)]}."""
    by_scan = collections.defaultdict(list)
    for r in gt_rows:
        by_scan[r["scan"]].append(r)
    active, next_id, out = {}, 1, {}
    for scan in sorted(by_scan):
        dets = by_scan[scan]
        ids = list(active)
        cost = np.full((len(ids), len(dets)), 1e6)
        for i, oid in enumerate(ids):
            o = active[oid]
            for j, d in enumerate(dets):
                dist = np.hypot(d["x"] - o["x"], d["y"] - o["y"])
                if d["cls"] == o["cls"] and dist <= link_m:
                    cost[i, j] = dist
        rows, cols = linear_sum_assignment(cost) if len(ids) and len(dets) else ([], [])
        assigned = {}
        for i, j in zip(rows, cols):
            if cost[i, j] < 1e6:
                assigned[j] = ids[i]
        frame = []
        for j, d in enumerate(dets):
            oid = assigned.get(j)
            if oid is None:
                oid, next_id = next_id, next_id + 1
            active[oid] = {"x": d["x"], "y": d["y"], "cls": d["cls"], "last": scan}
            frame.append((oid, d["x"], d["y"], d["cls"]))
        active = {k: v for k, v in active.items() if scan - v["last"] <= max_gap}
        out[scan] = frame
    return out


def evaluate(gt_by_scan, hyp_rows, scans, match_m, existence_gate, classes=None, ignore=None, ego=None, max_range=None):
    """ignore(scan) -> list of (x, y) don't-care GT positions. Hypotheses that match no evaluated GT but lie
    within match_m of a don't-care object, or beyond max_range of the machine, are not counted (KITTI-style)."""
    acc = mm.MOTAccumulator(auto_id=True)
    hyp_by_scan = collections.defaultdict(list)
    for r in hyp_rows:
        if r["existence"] >= existence_gate and r["cls"] != 0 and (classes is None or r["cls"] in classes):
            hyp_by_scan[r["scan"]].append(r)
    for scan in scans:
        g = [o for o in gt_by_scan.get(scan, []) if classes is None or o[3] in classes]
        h = hyp_by_scan.get(scan, [])
        gp = np.array([[o[1], o[2]] for o in g]).reshape(-1, 2)
        hp = np.array([[r["x"], r["y"]] for r in h]).reshape(-1, 2)
        if len(hp) and (ignore is not None or max_range is not None):
            matched = np.zeros(len(hp), bool)
            if len(gp):
                dmat = np.linalg.norm(gp[:, None] - hp[None], axis=2)
                rr, cc = linear_sum_assignment(np.where(dmat <= match_m, dmat, 1e6))
                matched[[c for r_, c in zip(rr, cc) if dmat[r_, c] <= match_m]] = True
            drop = np.zeros(len(hp), bool)
            dc = np.array(ignore(scan) if ignore else []).reshape(-1, 2)
            if len(dc):
                drop |= (np.linalg.norm(dc[:, None] - hp[None], axis=2) <= match_m).any(0)
            if max_range is not None:
                drop |= np.linalg.norm(hp - ego(h[0]["t"]), axis=1) > max_range
            keep = matched | ~drop
            h, hp = [x for x, k in zip(h, keep) if k], hp[keep]
        d = mm.distances.norm2squared_matrix(gp, hp, max_d2=match_m ** 2)
        acc.update([o[0] for o in g], [r["id"] for r in h], np.sqrt(d))
    names = ["num_frames", "num_unique_objects", "mostly_tracked", "mostly_lost", "num_false_positives",
             "num_misses", "num_switches", "num_fragmentations", "mota", "motp", "idf1", "precision", "recall"]
    s = mm.metrics.create().compute(acc, metrics=names, name="all")
    return {k: (round(float(s[k].iloc[0]), 4) if isinstance(s[k].iloc[0], float) else int(s[k].iloc[0])) for k in names}


def split_gt(run, seq, min_gt_points, max_range, gt_file="gt_objects.jsonl", min_height=0.0):
    """Pseudo-GT objects to evaluate vs don't-care: too few points, beyond max_range of the machine, or
    flatter than min_height (label trails on the ground, not objects)."""
    rows = read_jsonl(run / gt_file)
    ego = ego_xy(seq)
    for r in rows:
        r["range"] = r.get("range", float(np.linalg.norm(np.array([r["x"], r["y"]]) - ego(r["t"]))))
        r["care"] = r["n"] >= min_gt_points and r["range"] <= max_range and r.get("dims", [0, 0, 1e9])[2] >= min_height
    linked = link_gt(rows)  # link all, so an object keeps its identity while it is briefly sparse
    care = {(r["scan"], round(r["x"], 6), round(r["y"], 6)): r["care"] for r in rows}
    gt, dont = {}, collections.defaultdict(list)
    for scan, objs in linked.items():
        gt[scan] = [o for o in objs if care[(scan, round(o[1], 6), round(o[2], 6))]]
        dont[scan] = [(o[1], o[2]) for o in objs if not care[(scan, round(o[1], 6), round(o[2], 6))]]
    return gt, (lambda scan: dont.get(scan, [])), ego


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--tracks", default="tracks.jsonl", help="tracks file inside the run dir")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--match-m", type=float, default=1.0)
    ap.add_argument("--gates", type=float, nargs="+", default=[0.0, 0.5, 0.8])
    ap.add_argument("--min-gt-points", type=int, default=30)
    ap.add_argument("--max-range", type=float, default=20.0)
    ap.add_argument("--gt-file", default="gt_objects.jsonl")
    ap.add_argument("--min-gt-height", type=float, default=0.0, help="vertical extent (m) an evaluated GT object needs")
    args = ap.parse_args()
    seq = json.loads((args.run / "metrics.json").read_text())["seq"]
    frames = read_jsonl(args.run / "frames.jsonl")
    scans = [f["scan"] for f in frames]
    gt, ignore, ego = split_gt(args.run, seq, args.min_gt_points, args.max_range, args.gt_file, args.min_gt_height)
    hyp = read_jsonl(args.run / args.tracks)
    gt_classes = collections.Counter(o[3] for objs in gt.values() for o in objs)
    out = {
        "run": str(args.run), "date": datetime.date.today().isoformat(), "match_m": args.match_m,
        "tracks_file": args.tracks,
        "evaluated_gt": (f">= {args.min_gt_points} points, <= {args.max_range} m from the machine, vertical extent >= "
                         f"{args.min_gt_height} m ({args.gt_file}); others are don't-care"),
        "pseudo_gt": {"method": "GT-label clusters linked by nearest neighbour, same class, <=1.0 m, gap<=5 frames",
                      "unique_objects": len({o[0] for objs in gt.values() for o in objs}),
                      "detections_per_class": {ONTO_NAMES.get(c, str(c)): n for c, n in gt_classes.items()}},
        "hypotheses": {"generic_class_tracks_excluded": len({r["id"] for r in hyp if r["cls"] == 0}),
                       "tracks": len({r["id"] for r in hyp if r["cls"] != 0})},
        "by_existence_gate": {str(g): evaluate(gt, hyp, scans, args.match_m, g, None, ignore, ego, args.max_range)
                              for g in args.gates},
        "by_class_gate_0.5": {ONTO_NAMES[c]: evaluate(gt, hyp, scans, args.match_m, 0.5, {c}, ignore, ego, args.max_range)
                              for c in (17, 27, 8, 5) if gt_classes.get(c)},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    gate = str(args.gates[-1])
    g = out["by_existence_gate"][gate]
    print(f"gate {gate}: MOTA {g['mota']} IDF1 {g['idf1']} IDSW {g['num_switches']} FP {g['num_false_positives']} "
          f"FN {g['num_misses']} objects {g['num_unique_objects']} -> {args.out}")


if __name__ == "__main__":
    main()
