"""Re-run the tracker offline on a replay's stored detections (detections.jsonl), so tracker parameters
can be tuned on the val block in seconds instead of re-running the whole replay.

  python -m pipeline.track_offline --run <data root>/runs/replay_00001_val_final --min-points 15 --out tracks_mp15.jsonl
"""
import argparse
import collections
import json
from pathlib import Path

import numpy as np

from pipeline.tracker import COASTING, CONFIRMED, Tracker, TrackerParams


def run(run_dir: Path, min_points: int, params: TrackerParams):
    frames = [json.loads(line) for line in open(run_dir / "frames.jsonl")]
    dets = collections.defaultdict(list)
    for line in open(run_dir / "detections.jsonl"):
        d = json.loads(line)
        if d["n"] >= min_points:
            dets[d["scan"]].append(d)
    tracker, rows = Tracker(params), []
    for f in frames:
        inputs = [(np.array([d["x"], d["y"], d["z"]]), np.array(d["dims"]), d["cls"], d["conf"]) for d in dets[f["scan"]]]
        for tr in tracker.step(f["t"], inputs):
            if tr.state in (CONFIRMED, COASTING):
                rows.append({"scan": f["scan"], "t": f["t"], "id": tr.id, "state": tr.state, "x": float(tr.x[0]),
                             "y": float(tr.x[1]), "vx": float(tr.x[2]), "vy": float(tr.x[3]),
                             "cls": int(tr.semantic_class), "existence": round(tr.existence, 3)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--min-points", type=int, default=5)
    ap.add_argument("--out", required=True, help="file name inside the run dir")
    args = ap.parse_args()
    rows = run(args.run, args.min_points, TrackerParams())
    with open(args.run / args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{len(rows)} track rows, {len({r['id'] for r in rows})} tracks -> {args.run / args.out}")


if __name__ == "__main__":
    main()
