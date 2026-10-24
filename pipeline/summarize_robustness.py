"""Summarize the robustness study (scripts/slurm/robustness.sbatch): for each injected sensor fault, what happens
to semantics, to the safety and availability of the grid, to tracking, and whether the health monitor notices.

  python -m pipeline.summarize_robustness --out docs/measurements/robustness_00001_test.json \
      --figure docs/figures/robustness.png
"""
import argparse
import datetime
import json
from pathlib import Path

import numpy as np

from pipeline.rellis import DATA_ROOT

# family: (label, x-axis label, [(run name, severity)])
FAMILIES = {
    "lidar_drop": ("Lidar returns lost", "fraction of returns lost",
                   [("baseline", 0), ("drop25", 0.25), ("drop50", 0.5), ("drop75", 0.75), ("drop90", 0.9)]),
    "beams": ("Lidar rings", "rings kept of 64", [("baseline", 64), ("beams32", 32), ("beams16", 16)]),
    "range_noise": ("Lidar range noise", "sigma (m)",
                    [("baseline", 0), ("noise03", 0.03), ("noise10", 0.1), ("noise30", 0.3)]),
    "cam_offset": ("Camera clock offset", "offset (ms)",
                   [("baseline", 0), ("camoff050", 50), ("camoff100", 100), ("camoff200", 200)]),
    "cam_yaw": ("Camera extrinsic drift", "yaw error (deg)",
                [("baseline", 0), ("camyaw1", 1), ("camyaw2", 2), ("camyaw4", 4)]),
    "cam_gain": ("Camera exposure", "brightness gain", [("camgain025", 0.25), ("baseline", 1), ("camgain3", 3)]),
}
POINTS = 64 * 2048
HEALTH_MIN_VALID = 0.2  # health_monitor lidar_min_valid_fraction (gen_ros_params.py)


def run_metrics(run: Path):
    m = json.loads((run / "metrics.json").read_text())
    v = m["default_variant"]
    pt = {r: m["point_iou"][r][v] for r in ("all", "wedge")}
    big = lambda s: round(float(np.mean([iou for c, iou in s["per_class_iou"].items() if s["support"][c] >= 10000])), 2)
    c = np.array(m["traversability_confusion_gt_rows_pred_cols"]["counts"])  # rows GT (clean sweep), cols predicted
    lethal_obs = c[3, :4].sum()  # GT-lethal cells that the prediction observed (README convention)
    frames = [json.loads(line) for line in open(run / "frames.jsonl")]
    trk = json.loads((run / "tracking.json").read_text())["by_existence_gate"]["0.8"]
    return {
        "point_miou_all": pt["all"]["miou"], "point_miou_all_ge10k": big(pt["all"]),
        "point_miou_wedge": pt["wedge"]["miou"],
        "lethal_predicted_lethal": round(float(c[3, 3] / lethal_obs), 4),
        "lethal_predicted_free": round(float(c[3, 0] / lethal_obs), 4),
        "lethal_predicted_unknown": round(float(c[3, 4] / c[3].sum()), 4),
        "free_predicted_lethal": round(float(c[0, 3] / c[0, :4].sum()), 4),
        "corridor_unknown_0.9-2m": m["corridor_unknown_fraction"].get("memory_5s|0.9-2m"),
        "planner_fraction_moving": m["planner_offline"]["fraction_moving"],
        "planner_reasons": m["planner_offline"]["reasons"],
        "tracking": {k: trk[k] for k in ("mota", "idf1", "num_switches", "recall", "precision")},
        "camera_matched_fraction": round(float(np.mean([f["camera_used"] for f in frames])), 4),
        "lidar_valid_fraction_median": round(float(np.median([f["n_valid"] for f in frames])) / POINTS, 4),
        "health_flags_lidar": bool(np.median([f["n_valid"] for f in frames]) / POINTS < HEALTH_MIN_VALID),
        "faults": m["faults"],
    }


def figure(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [("point_miou_all_ge10k", "point mIoU\n(classes ≥10k pts)", 1),
            ("lethal_predicted_free", "lethal cells\nreported free (%)", 100),
            ("planner_fraction_moving", "planner can move\n(% of scans)", 100),
            ("mota", "tracking MOTA", 1)]
    fams = [f for f in FAMILIES if all(n in res for n, _ in FAMILIES[f][2])]
    ink, ink2, series, grid_c, base_c = "#0b0b0b", "#52514e", "#2a78d6", "#e4e3df", "#8a8984"
    fig, axes = plt.subplots(len(rows), len(fams), figsize=(2.6 * len(fams), 2.1 * len(rows)), sharey="row",
                             squeeze=False, facecolor="#fcfcfb")
    for j, f in enumerate(fams):
        title, xlabel, pts = FAMILIES[f]
        pts = sorted(pts, key=lambda p: p[1])
        for i, (key, ylabel, scale) in enumerate(rows):
            ax = axes[i][j]
            ax.set_facecolor("#fcfcfb")
            val = lambda n: (res[n]["tracking"][key] if key == "mota" else res[n][key]) * scale
            xs, ys = [p[1] for p in pts], [val(n) for n, _ in pts]
            ax.axhline(val("baseline"), color=base_c, lw=1, ls=(0, (3, 3)), zorder=1)
            ax.plot(xs, ys, color=series, lw=2, marker="o", ms=5, mec="#fcfcfb", mew=1.5, zorder=3)
            ax.grid(axis="y", color=grid_c, lw=0.8)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                ax.spines[s].set_color(grid_c)
            ax.tick_params(colors=ink2, labelsize=8)
            ax.set_xticks(xs)
            if i == 0:
                ax.set_title(title, fontsize=10, color=ink, loc="left")
            if i == len(rows) - 1:
                ax.set_xlabel(xlabel, fontsize=8, color=ink2)
            if j == 0:
                ax.set_ylabel(ylabel, fontsize=8.5, color=ink2)
    fig.suptitle("Sensor faults injected into the replay of RELLIS-3D 00001 (held-out test block); "
                 "dashed line = no fault", fontsize=10.5, color=ink, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--figure", type=Path)
    args = ap.parse_args()
    names = sorted({n for _, _, pts in FAMILIES.values() for n, _ in pts})
    res, missing = {}, []
    for n in names:
        run = DATA_ROOT / f"runs/replay_00001_robust_{n}"
        if (run / "tracking.json").exists():
            res[n] = run_metrics(run)
        else:
            missing.append(n)
    out = {"date": datetime.date.today().isoformat(), "seq": "00001", "scans": "0:1047 (test block)",
           "method": "faults injected into the replayed sensor data (pipeline/faults.py); ground truth from the "
                     "clean data; same models, fusion, grid (5 s memory), tracker and offline planner as the baseline",
           "families": {f: {"label": FAMILIES[f][0], "x": FAMILIES[f][1], "runs": FAMILIES[f][2]} for f in FAMILIES},
           "runs": res, "missing": missing}
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(f"{len(res)} runs, missing {missing}")
    for n, r in res.items():
        print(f"{n:11s} mIoU {r['point_miou_all_ge10k']:5.1f}  wedge {r['point_miou_wedge']:5.1f}  "
              f"lethal->free {100 * r['lethal_predicted_free']:5.2f}%  lethal->unk {100 * r['lethal_predicted_unknown']:5.1f}%  "
              f"moving {100 * r['planner_fraction_moving']:5.1f}%  MOTA {r['tracking']['mota']:.3f}  "
              f"valid {r['lidar_valid_fraction_median']:.2f}  cam {r['camera_matched_fraction']:.2f}")
    if args.figure:
        figure(res, args.figure)
        print("wrote", args.figure)


if __name__ == "__main__":
    main()
