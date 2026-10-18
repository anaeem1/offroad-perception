"""Summarize late-fusion results from the 00001 val/test replays: the val selection table and held-out test
numbers for the selected variant, under two mean conventions.

mIoU over every class present in the region is dominated by classes with a handful of points (00001
test has 22 barrier points), so the summary also reports the mean over classes with >= min_support
points. Per-class IoU and support are always reported alongside.

  python -m pipeline.summarize_fusion --val <run> --test <run> --out docs/measurements/fusion_00001.json
"""
import argparse
import datetime
import json
from pathlib import Path

import numpy as np


def means(entry, min_support):
    iou, sup = entry["per_class_iou"], entry["support"]
    big = [iou[c] for c in iou if sup.get(c, 0) >= min_support]
    return {"miou_all_present": entry["miou"], f"miou_support_ge_{min_support}": round(float(np.mean(big)), 2) if big else None,
            "classes_counted": len(big)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", type=Path, required=True)
    ap.add_argument("--test", type=Path, required=True)
    ap.add_argument("--min-support", type=int, default=10000)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    val = json.loads((args.val / "metrics.json").read_text())
    test = json.loads((args.test / "metrics.json").read_text())
    lidar_only = "alpha=1.0|linear|mc=on|pitch=0.0"
    selected = max(val["point_iou"]["wedge"], key=lambda k: val["point_iou"]["wedge"][k]["miou"])
    # Report the variant the pipeline actually ships (its default), plus the val argmax for transparency.
    shipped = test["default_variant"]
    table = sorted(((k, v["miou"]) for k, v in val["point_iou"]["wedge"].items()), key=lambda kv: -kv[1])
    out = {
        "date": datetime.date.today().isoformat(), "val_run": str(args.val), "test_run": str(args.test),
        "selection": {"criterion": "wedge mIoU on the 00001 val block (scans 1047-2318)",
                      "val_argmax": selected, "shipped": shipped,
                      "val_wedge_miou_top10": dict(table[:10]),
                      "val_wedge_miou_lidar_only": val["point_iou"]["wedge"][lidar_only]["miou"]},
        "test": {region: {name: {**means(test["point_iou"][region][key], args.min_support),
                                 "per_class_iou": test["point_iou"][region][key]["per_class_iou"],
                                 "support": test["point_iou"][region][key]["support"]}
                          for name, key in (("lidar_only", lidar_only), ("shipped_fusion", shipped),
                                            ("camera_only_same_extrinsic", shipped.replace("alpha=0.5", "alpha=0.0")))}
                 for region in ("wedge", "all")},
        "ablations_test_wedge": {k: test["point_iou"]["wedge"][k]["miou"] for k in (
            shipped, shipped.replace("pitch=-0.5", "pitch=0.0"), shipped.replace("mc=on", "mc=off"),
            shipped.replace("loglinear", "linear"))},
        "bev_cell_iou_test": {**means(test["bev_cell_iou_default"], args.min_support),
                              "per_class_iou": test["bev_cell_iou_default"]["per_class_iou"]},
        "traversability_confusion_test": test["traversability_confusion_gt_rows_pred_cols"],
    }
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    for region in ("wedge", "all"):
        print(region, {n: (v["miou_all_present"], v[f"miou_support_ge_{args.min_support}"]) for n, v in out["test"][region].items()})
    print("ablations", out["ablations_test_wedge"])


if __name__ == "__main__":
    main()
