"""Grid-level summary of the 00001 val/test replays: corridor coverage per memory age (ADR-0008), BEV cell IoU,
traversability safety rates, and camera matching.

  python -m pipeline.summarize_replay --val <run> --test <run> --out docs/measurements/replay_00001_summary.json
"""
import argparse
import datetime
import json
from pathlib import Path

import numpy as np


def block(run: Path, min_support: int):
    m = json.loads((run / "metrics.json").read_text())
    c = np.array(m["traversability_confusion_gt_rows_pred_cols"]["counts"])[:4, :4]  # GT observed this sweep
    frames = [json.loads(line) for line in open(run / "frames.jsonl")]
    w = np.array([f["cam_minus_wedge_ms"] for f in frames if f["cam_minus_wedge_ms"] is not None])
    bev = m["bev_cell_iou_default"]
    big = [v for k, v in bev["per_class_iou"].items() if bev["support"].get(k, 0) >= min_support]
    return {
        "run": str(run), "scans": m["scans"], "memory_max_age_s": m["memory_max_age_s"],
        "corridor_unknown_fraction": m["corridor_unknown_fraction"],
        "bev_cell_iou_single_sweep": {"miou_all_present": bev["miou"],
                                      f"miou_support_ge_{min_support}": round(float(np.mean(big)), 2),
                                      "per_class_iou": bev["per_class_iou"]},
        "traversability_vs_this_sweeps_labels": {
            "buckets": ["free<50", "low<128", "high<254", "lethal"],
            "gt_lethal_predicted_lethal": round(float(c[3, 3] / c[3].sum()), 4),
            "gt_lethal_predicted_free": round(float(c[3, 0] / c[3].sum()), 4),
            "gt_free_predicted_lethal": round(float(c[0, 3] / c[0].sum()), 4),
            "predicted_lethal_precision": round(float(c[3, 3] / c[:, 3].sum()), 4),
            "counts": c.tolist()},
        "camera_match": {"used_fraction": round(float(np.mean([f["camera_used"] for f in frames])), 4),
                         "cam_minus_wedge_ms_p5_p50_p95": [round(float(np.percentile(w, p)), 1) for p in (5, 50, 95)]},
        "remembered_cells_median": float(np.median([f.get("remembered_cells", 0) for f in frames])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", type=Path, required=True)
    ap.add_argument("--test", type=Path, required=True)
    ap.add_argument("--min-support", type=int, default=10000)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    out = {"date": datetime.date.today().isoformat(), "val": block(args.val, args.min_support),
           "test": block(args.test, args.min_support)}
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    t = out["test"]
    print(json.dumps({k: t[k] for k in ("traversability_vs_this_sweeps_labels", "camera_match", "remembered_cells_median")}, indent=1))
    print("bev test", t["bev_cell_iou_single_sweep"]["miou_all_present"], t["bev_cell_iou_single_sweep"][f"miou_support_ge_{args.min_support}"])


if __name__ == "__main__":
    main()
