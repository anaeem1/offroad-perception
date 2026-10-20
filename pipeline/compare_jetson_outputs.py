"""Compare the contract outputs recorded on the Jetson (C++ nodes, TensorRT FP16) against the Python reference
(ONNX Runtime FP32) on the same clip. Not expected to be bit-identical (different inference precision and
live callback timing), so this reports agreement rates.

  python -m pipeline.compare_jetson_outputs --outputs <jetson replay_out/outputs> \
      --ref <data root>/runs/replay_00001_clip_ref --out docs/measurements/jetson_vs_reference_00001.json
"""
import argparse
import datetime
import json
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_types_from_msg, get_typestore

REPO = Path(__file__).resolve().parents[1]


def typestore_with_contract():
    ts = get_typestore(Stores.ROS2_HUMBLE)
    types = {}
    for f in sorted((REPO / "src/perception_msgs/msg").glob("*.msg")):
        types.update(get_types_from_msg(f.read_text(), f"perception_msgs/msg/{f.stem}"))
    ts.register(types)
    return ts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", type=Path, required=True, help="ros2 bag recorded on the Jetson")
    ap.add_argument("--ref", type=Path, required=True, help="reference replay run with --save-grids 1")
    ap.add_argument("--warmup-s", type=float, default=5.0, help="skip while the grid memories converge")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    frames = [json.loads(line) for line in open(args.ref / "frames.jsonl")]
    stamp_to_scan = {round(f["t"], 3): f["scan"] for f in frames}
    grids = np.load(args.ref / "grids.npz")
    t_start = frames[0]["t"] + args.warmup_s

    ts = typestore_with_contract()
    per = []
    n_grid = n_obs = 0
    obstacles, health_modes = [], []
    with AnyReader([args.outputs], default_typestore=ts) as reader:
        for conn, _, raw in reader.messages():
            msg = reader.deserialize(raw, conn.msgtype)
            if conn.topic == "/perception/health":
                health_modes.append(int(msg.mode))
                continue
            stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            if conn.topic == "/perception/obstacles":
                n_obs += 1
                obstacles.append(sum(1 for o in msg.obstacles if o.existence_probability >= 0.8))
                continue
            if conn.topic != "/perception/traversability":
                continue
            n_grid += 1
            scan = stamp_to_scan.get(round(stamp, 3))
            if scan is None or stamp < t_start or str(scan) not in grids.files:
                continue
            ref_cost, ref_cls = grids[str(scan)][0], grids[str(scan)][1]
            cost = np.asarray(msg.cost, np.uint8).reshape(msg.height, msg.width)
            cls = np.asarray(msg.semantic_class, np.uint8).reshape(msg.height, msg.width)
            both = (cost != 255) & (ref_cost != 255)
            lj, lr = cost == 254, ref_cost == 254
            per.append({"scan": scan, "cost_identical": float((cost == ref_cost).mean()),
                        "unknown_agree": float(((cost == 255) == (ref_cost == 255)).mean()),
                        "class_identical_where_both_known": float((cls[both] == ref_cls[both]).mean()) if both.any() else None,
                        "lethal_iou": float((lj & lr).sum() / max(1, (lj | lr).sum())),
                        "lethal_jetson_only": int((lj & ~lr).sum()), "lethal_ref_only": int((~lj & lr).sum())})
    agg = {k: round(float(np.mean([p[k] for p in per if p[k] is not None])), 4)
           for k in ("cost_identical", "unknown_agree", "class_identical_where_both_known", "lethal_iou")}
    out = {"date": datetime.date.today().isoformat(), "jetson_outputs": str(args.outputs), "reference": str(args.ref),
           "grids_received": n_grid, "grids_compared": len(per), "warmup_skipped_s": args.warmup_s,
           "agreement_mean": agg,
           "lethal_cells_mean": {"jetson_only": float(np.mean([p["lethal_jetson_only"] for p in per])),
                                 "reference_only": float(np.mean([p["lethal_ref_only"] for p in per]))},
           "obstacle_msgs": n_obs, "confirmed_obstacles_per_msg_mean": float(np.mean(obstacles)) if obstacles else None,
           "health_mode_counts": {str(m): health_modes.count(m) for m in sorted(set(health_modes))}}
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
