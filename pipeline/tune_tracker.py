"""Tune the tracker's detection threshold on the val block, report the chosen setting on the test block.
Runs offline from the replays' stored detections (pipeline/track_offline.py), so each setting takes seconds.

  python -m pipeline.tune_tracker --val <run> --test <run> --out docs/measurements/tracking_00001_test.json
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from pipeline.track_offline import run as track
from pipeline.tracker import TrackerParams


def evaluate(run_dir, tracks_name, out_json, gates):
    cmd = [sys.executable, "-m", "pipeline.eval_tracking", "--run", str(run_dir), "--tracks", tracks_name,
           "--gt-file", "gt_objects_v2.jsonl", "--min-gt-height", "0.5", "--out", str(out_json),
           "--gates", *[str(g) for g in gates]]
    subprocess.run(cmd, check=True, capture_output=True)
    return json.loads(Path(out_json).read_text())


def write_tracks(run_dir, min_points):
    name = f"tracks_mp{min_points}.jsonl"
    with open(run_dir / name, "w") as f:
        for r in track(run_dir, min_points, TrackerParams()):
            f.write(json.dumps(r) + "\n")
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", type=Path, required=True)
    ap.add_argument("--test", type=Path, required=True)
    ap.add_argument("--min-points", type=int, nargs="+", default=[5, 15, 30, 50])
    ap.add_argument("--gates", type=float, nargs="+", default=[0.5, 0.8])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    val_table = {}
    for mp in args.min_points:
        res = evaluate(args.val, write_tracks(args.val, mp), args.val / f"tracking_mp{mp}.json", args.gates)
        for g in args.gates:
            v = res["by_existence_gate"][str(g)]
            val_table[f"min_points={mp}|gate={g}"] = {k: v[k] for k in ("mota", "idf1", "num_switches",
                                                                         "num_false_positives", "num_misses", "recall", "precision")}
    best = max(val_table, key=lambda k: val_table[k]["mota"])
    mp, gate = int(best.split("|")[0].split("=")[1]), float(best.split("|")[1].split("=")[1])
    test = evaluate(args.test, write_tracks(args.test, mp), args.test / f"tracking_mp{mp}.json", [0.0, 0.5, 0.8])
    out = {"selection": {"criterion": "MOTA on the 00001 val block", "chosen": best, "val_table": val_table},
           "test": test, "test_at_chosen_gate": test["by_existence_gate"][str(gate)]}
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print("chosen", best)
    print("test", json.dumps(out["test_at_chosen_gate"]))


if __name__ == "__main__":
    main()
