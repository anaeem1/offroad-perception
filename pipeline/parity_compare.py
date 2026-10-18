"""Compare C++ stage outputs (cpp_*.npy) with the Python reference (py_*.npy) and record the result.

  python -m pipeline.parity_compare --dump <data root>/parity/00001 --json docs/measurements/cpp_parity.json
"""
import argparse
import datetime
import json
from pathlib import Path

import numpy as np

from ml.lidar.range_projection import project  # noqa: F401  (documents which function is the reference)


def load(d, name):
    return np.load(d / f"{name}.npy")


def compare_scan(d):
    r = {}
    py, cpp = load(d, "py_range_input"), load(d, "cpp_range_input")
    px_diff = np.abs(py - cpp).max(0) > 1e-6
    r["range_input"] = {"max_abs": float(np.abs(py - cpp).max()), "pixels_differing": int(px_diff.sum()),
                        "pixels": int(px_diff.size)}
    r["deskew_cam_max_abs_m"] = float(np.abs(load(d, "py_deskew_cam") - load(d, "cpp_deskew_cam")).max())
    pi, ci = load(d, "py_hits_idx"), load(d, "cpp_hits_idx")
    common = np.intersect1d(pi, ci)
    pu = dict(zip(pi.tolist(), load(d, "py_hits_uv")))
    cu = dict(zip(ci.tolist(), load(d, "cpp_hits_uv")))
    r["camera_hits"] = {"python": int(len(pi)), "cpp": int(len(ci)), "only_one_side": int(len(np.setxor1d(pi, ci))),
                        "max_abs_px": float(max((np.abs(pu[k] - cu[k]).max() for k in common.tolist()), default=0))}
    pp, cp = load(d, "py_probs"), load(d, "cpp_probs")
    r["fusion"] = {"max_abs": float(np.abs(pp - cp).max()), "argmax_agree": float((pp.argmax(1) == cp.argmax(1)).mean())}
    r["xyz_base_max_abs_m"] = float(np.abs(load(d, "py_xyz_base") - load(d, "cpp_xyz_base")).max())
    g = {}
    for k in ("cost", "class", "flags", "geometric"):
        g[f"{k}_cells_differing"] = int((load(d, f"py_grid_{k}") != load(d, f"cpp_grid_{k}")).sum())
    conf = np.abs(load(d, "py_grid_conf").astype(int) - load(d, "cpp_grid_conf").astype(int))
    g["confidence_max_abs"], g["confidence_cells_differing"] = int(conf.max()), int((conf > 0).sum())
    r["grid"] = g
    pd, cd = load(d, "py_dets"), load(d, "cpp_dets")
    key = lambda a: a[np.lexsort((a[:, 1], a[:, 0]))] if len(a) else a
    pd, cd = key(pd), key(cd)
    same = len(pd) == len(cd)
    r["clusters"] = {"python": int(len(pd)), "cpp": int(len(cd)),
                     "max_abs": float(np.abs(pd - cd).max()) if same and len(pd) else (0.0 if same else None)}
    return r


def compare_tracker(d):
    py, cpp = load(d, "py_tracks"), load(d, "cpp_tracks")
    out = {"rows_python": int(len(py)), "rows_cpp": int(len(cpp))}
    if len(py) == len(cpp):
        out["id_state_class_identical"] = bool((py[:, [0, 1, 2, 8]] == cpp[:, [0, 1, 2, 8]]).all())
        out["state_max_abs"] = float(np.abs(py[:, 3:8] - cpp[:, 3:8]).max()) if len(py) else 0.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", type=Path, required=True)
    ap.add_argument("--json", type=Path, required=True)
    args = ap.parse_args()
    scans = {p.name: compare_scan(p) for p in sorted(args.dump.glob("scan_*"))}
    trk = compare_tracker(args.dump / "tracker")
    worst = lambda f: max(f(s) for s in scans.values())
    passed = {
        "range_input": worst(lambda s: s["range_input"]["pixels_differing"] / s["range_input"]["pixels"]) <= 1e-3,
        "deskew": worst(lambda s: s["deskew_cam_max_abs_m"]) < 1e-6,
        "camera_projection": worst(lambda s: s["camera_hits"]["only_one_side"]) == 0
                             and worst(lambda s: s["camera_hits"]["max_abs_px"]) < 1e-6,
        "fusion": worst(lambda s: s["fusion"]["max_abs"]) < 1e-5,
        "base_transform": worst(lambda s: s["xyz_base_max_abs_m"]) < 1e-6,
        "grid": worst(lambda s: s["grid"]["cost_cells_differing"] + s["grid"]["class_cells_differing"]
                      + s["grid"]["flags_cells_differing"] + s["grid"]["geometric_cells_differing"]) == 0
                and worst(lambda s: s["grid"]["confidence_max_abs"]) <= 1,
        "clustering": all(s["clusters"]["python"] == s["clusters"]["cpp"] and (s["clusters"]["max_abs"] or 0) < 1e-6
                          for s in scans.values()),
        "tracker": trk.get("id_state_class_identical", False) and trk.get("state_max_abs", 1) < 1e-6,
    }
    out = {"date": datetime.date.today().isoformat(), "dump": str(args.dump), "passed": passed,
           "all_passed": all(passed.values()), "per_scan": scans, "tracker": trk}
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"passed": passed, "tracker": trk}, indent=1))
    for name, s in scans.items():
        print(name, json.dumps(s))


if __name__ == "__main__":
    main()
