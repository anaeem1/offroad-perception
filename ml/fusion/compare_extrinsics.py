"""Cross-check the camera<-lidar extrinsic implied by the bag's /tf_static against the RELLIS-3D
calibration file, by rotation/translation difference and by lidar/image label agreement on sequence 00004.

  python -m ml.fusion.compare_extrinsics --inspection docs/measurements/bag_00004_inspection.json \
      --seq 00004 --out docs/measurements/extrinsics_tf_vs_calib.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from ml.fusion.check_projection import agreement, annotated_frames, load_frame
from ml.fusion.projection import load_rellis_calib
from pipeline.geometry import T_root_frame, tf_graph


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspection", type=Path, required=True)
    ap.add_argument("--seq", default="00004")
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    info = json.loads(args.inspection.read_text())
    g = tf_graph(info["tf_static"])
    lidar_frame, cam_frame = info["pointcloud"]["frame_id"], info["camera_info"]["frame_id"]
    T_tf = np.linalg.inv(T_root_frame(g, cam_frame)) @ T_root_frame(g, lidar_frame)
    cam, T_cal = load_rellis_calib(args.seq)

    dR = Rotation.from_matrix(T_tf[:3, :3] @ T_cal[:3, :3].T)
    yaw, pitch, roll = dR.as_euler("yxz", degrees=True)  # camera axes: y down (yaw), x right (pitch)
    frames = annotated_frames()[args.seq]
    keys = sorted(frames)
    sample = [keys[i] for i in np.linspace(0, len(keys) - 1, min(args.frames, len(keys))).astype(int)]
    tallies = {"calibration_file": [0, 0], "tf_static": [0, 0]}
    for k in sample:
        xyz, lid, _, imlab = load_frame(args.seq, k, *frames[k])
        for name, T in (("calibration_file", T_cal), ("tf_static", T_tf)):
            a, n = agreement(xyz, lid, imlab, cam, T)
            tallies[name][0] += a
            tallies[name][1] += n

    out = {
        "seq": args.seq, "frames": len(sample), "lidar_frame": lidar_frame, "camera_frame": cam_frame,
        "rotation_difference_deg": {"total": round(float(np.degrees(dR.magnitude())), 3),
                                    "yaw": round(float(yaw), 3), "pitch": round(float(pitch), 3),
                                    "roll": round(float(roll), 3)},
        "translation_difference_m": [round(float(v), 4) for v in T_tf[:3, 3] - T_cal[:3, 3]],
        "label_agreement_pct": {k: round(100 * a / max(n, 1), 2) for k, (a, n) in tallies.items()},
        "T_cam_lidar": {"calibration_file": np.round(T_cal, 6).tolist(), "tf_static": np.round(T_tf, 6).tolist()},
    }
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "T_cam_lidar"}, indent=1))


if __name__ == "__main__":
    main()
