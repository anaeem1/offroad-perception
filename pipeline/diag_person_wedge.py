"""Diagnostic: why do ground-truth person points inside the camera wedge score ~0 IoU on 00001 test?
For the scans with the most such points: range, and what the lidar model / camera model / fusion say.

  python -m pipeline.diag_person_wedge --seq 00001 --scans 0:1047 --top 12 --out docs/measurements/diag_person_wedge_00001.json
"""
import argparse
import collections
import datetime
import json
from pathlib import Path

import cv2
import numpy as np

from ml.fusion.projection import load_rellis_calib, perturb, project_points
from pipeline import fusion as F
from pipeline.models import CameraSeg, LidarSeg
from pipeline.rellis import image_frames, load_scan_labels, scan_paths

PERSON = 17
NAMES = dict(zip(F.ONTOLOGY_IDS.tolist(), ["void", "dirt", "grass", "tree", "pole", "water", "sky", "vehicle", "object",
             "asphalt", "building", "log", "person", "fence", "bush", "concrete", "barrier", "puddle", "mud", "rubble"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--scans", required=True)
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    lo, hi = (int(v) for v in args.scans.split(":"))
    cam, T = load_rellis_calib(args.seq)
    T = perturb(T, pitch_deg=-0.5)
    counts = {}
    for k in range(lo, hi):
        xyz = np.fromfile(scan_paths(args.seq, k)[0], np.float32).reshape(-1, 4)[:, :3]
        gt = load_scan_labels(args.seq, k)
        sel = np.flatnonzero(gt == PERSON)
        if len(sel):
            _, i, _ = project_points(xyz[sel], cam, T)
            if len(i):
                counts[k] = len(i)
    total = sum(counts.values())
    top = sorted(counts, key=counts.get, reverse=True)[:args.top]
    lidar, camera = LidarSeg("cpu"), CameraSeg("cpu")
    frames = image_frames(args.seq)
    per_scan, agg = [], collections.Counter()
    for k in top:
        scan = np.fromfile(scan_paths(args.seq, k)[0], np.float32).reshape(-1, 4)
        xyz, rem = scan[:, :3], scan[:, 3]
        gt = load_scan_labels(args.seq, k)
        p_lid = lidar(xyz, rem)
        rgb = cv2.cvtColor(cv2.imread(str(frames[k][1])), cv2.COLOR_BGR2RGB)
        p_cam = camera(rgb)
        uv, idx, depth = project_points(xyz, cam, T)
        person = gt[idx] == PERSON
        pc = F.camera_to_ontology(F.sample_camera(p_cam, uv[person]), camera.class_ids)
        pl = F.lidar_to_ontology(p_lid[idx[person]], lidar.train_to_onto, lidar.bucket)
        lid = collections.Counter(NAMES[c] for c in F.lidar_labels(p_lid[idx[person]], lidar.train_to_onto))
        cmr = collections.Counter(NAMES[c] for c in F.ONTOLOGY_IDS[pc.argmax(1)])
        fus = collections.Counter(NAMES[c] for c in F.ONTOLOGY_IDS[F.fuse(pl, pc, 0.5, "loglinear").argmax(1)])
        # Does the camera see a person anywhere near the projected points (alignment vs recognition)?
        cam_lab = camera.class_ids[p_cam.argmax(0)]
        near = 0
        for u, v in uv[person]:
            cu, cv_ = int(u * cam_lab.shape[1] / 1920), int(v * cam_lab.shape[0] / 1200)
            win = cam_lab[max(0, cv_ - 3):cv_ + 4, max(0, cu - 3):cu + 4]  # +/- 3 cells = +/- 22 px
            near += int((win == PERSON).any())
        per_scan.append({"scan": k, "person_points_in_wedge": int(person.sum()),
                         "range_m_median": round(float(np.median(depth[person])), 1),
                         "lidar_says": dict(lid.most_common(3)), "camera_says": dict(cmr.most_common(3)),
                         "fused_says": dict(fus.most_common(3)),
                         "camera_person_within_22px": round(near / max(1, person.sum()), 3)})
        agg.update({"points": int(person.sum()), "lidar_person": lid["person"], "camera_person": cmr["person"],
                    "fused_person": fus["person"], "camera_person_nearby": near})
    out = {"seq": args.seq, "scans": args.scans, "date": datetime.date.today().isoformat(),
           "scans_with_person_in_wedge": len(counts), "person_points_in_wedge_total": int(total),
           "top_scans": per_scan,
           "top_scans_rates": {k: round(v / max(1, agg["points"]), 3) for k, v in agg.items() if k != "points"}}
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in ("scans_with_person_in_wedge", "person_points_in_wedge_total", "top_scans_rates")}))
    for s in per_scan:
        print(json.dumps(s))


if __name__ == "__main__":
    main()
