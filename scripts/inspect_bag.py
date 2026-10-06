"""Inspect a converted RELLIS-3D ROS 2 bag: frame ids, TF tree, camera model, point layout, and the
camera-to-lidar timestamp offsets that the contract's +/-50 ms match window depends on. Also checks that
every topic deserializes with the stock ROS 2 Humble message definitions (i.e. replayable on the Jetson).

  python scripts/inspect_bag.py <ros2_bag_dir> --json docs/measurements/bag_00004_inspection.json
"""
import argparse
import json
import struct
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

LIDAR, CAMERA, INFO = "/os1_cloud_node/points", "/pylon_camera_node/image_raw", "/pylon_camera_node/camera_info"


def header_stamp(raw):
    """CDR: 4-byte encapsulation, then std_msgs/Header.stamp (int32 sec, uint32 nanosec)."""
    sec, nsec = struct.unpack_from("<iI", raw, 4)
    return sec + nsec * 1e-9


def field(msg, name):
    """Converted bags embed the ROS 1 definitions (CameraInfo D/K/P); Humble spells them d/k/p."""
    return getattr(msg, name) if hasattr(msg, name) else getattr(msg, name.upper())


def tf_edges(msg):
    return [(t.header.frame_id, t.child_frame_id,
             [round(v, 5) for v in (t.transform.translation.x, t.transform.translation.y, t.transform.translation.z)],
             [round(v, 6) for v in (t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z,
                                    t.transform.rotation.w)]) for t in msg.transforms]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bag", type=Path)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    humble = get_typestore(Stores.ROS2_HUMBLE)
    out = {"bag": str(args.bag), "humble_deserialization": {}}
    stamps = {LIDAR: [], CAMERA: []}
    lag = {LIDAR: [], CAMERA: []}  # bag receive time - header stamp
    dynamic_tf = set()

    with AnyReader([args.bag]) as reader:
        first = {}
        for conn, t_ns, raw in reader.messages():
            topic = conn.topic
            if topic not in first:
                first[topic] = True
                try:
                    humble.deserialize_cdr(raw, conn.msgtype)
                    out["humble_deserialization"][topic] = "ok"
                except Exception as e:  # report, don't stop: the point is to find incompatible topics
                    out["humble_deserialization"][topic] = f"FAILED: {e}"
                msg = reader.deserialize(raw, conn.msgtype)
                if topic == INFO:
                    out["camera_info"] = {"frame_id": msg.header.frame_id, "width": msg.width, "height": msg.height,
                                          "distortion_model": msg.distortion_model, "D": [float(v) for v in field(msg, "d")],
                                          "K": [float(v) for v in field(msg, "k")],
                                          "P": [float(v) for v in field(msg, "p")]}
                elif topic == CAMERA:
                    out["image"] = {"frame_id": msg.header.frame_id, "encoding": msg.encoding,
                                    "width": msg.width, "height": msg.height}
                elif topic == LIDAR:
                    out["pointcloud"] = {"frame_id": msg.header.frame_id, "height": msg.height, "width": msg.width,
                                         "point_step": msg.point_step, "is_dense": bool(msg.is_dense),
                                         "fields": [(f.name, f.offset, f.datatype) for f in msg.fields]}
                elif topic == "/tf_static":
                    out["tf_static"] = tf_edges(msg)
            if topic in stamps:
                s = header_stamp(raw)
                stamps[topic].append(s)
                lag[topic].append(t_ns * 1e-9 - s)
            elif topic == "/tf":
                for parent, child, *_ in tf_edges(reader.deserialize(raw, conn.msgtype)):
                    dynamic_tf.add((parent, child))

    lid, cam = np.array(stamps[LIDAR]), np.sort(np.array(stamps[CAMERA]))
    j = np.clip(np.searchsorted(cam, lid), 1, len(cam) - 1)
    nearest = np.where(np.abs(cam[j] - lid) < np.abs(cam[j - 1] - lid), cam[j], cam[j - 1])
    dt_ms = (nearest - lid) * 1e3
    pct = lambda a, q: round(float(np.percentile(a, q)), 2)
    out["tf_dynamic_edges"] = sorted(dynamic_tf)
    out["camera_minus_lidar_stamp_ms"] = {
        "n_lidar_scans": len(lid), "median": pct(dt_ms, 50), "p05": pct(dt_ms, 5), "p95": pct(dt_ms, 95),
        "max_abs": round(float(np.abs(dt_ms).max()), 2),
        "within_50ms_pct": round(100 * float((np.abs(dt_ms) <= 50).mean()), 2)}
    out["receive_minus_stamp_ms"] = {t: {"median": pct(np.array(v) * 1e3, 50), "p95": pct(np.array(v) * 1e3, 95)}
                                     for t, v in lag.items()}
    out["inter_scan_ms"] = {"lidar_median": pct(np.diff(np.sort(lid)) * 1e3, 50),
                            "camera_median": pct(np.diff(cam) * 1e3, 50),
                            "lidar_max": round(float(np.diff(np.sort(lid)).max() * 1e3), 2),
                            "camera_max": round(float(np.diff(cam).max() * 1e3), 2)}
    print(json.dumps(out, indent=1))
    if args.json:
        args.json.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
