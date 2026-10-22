#!/usr/bin/env python3
"""Fault injection for hazard H-6 (lidar loss): copy a window of a replay clip with the lidar topic removed
for a few seconds in the middle. Run ON the Jetson with ROS 2 sourced (uses rosbag2_py, outside conda):

  python3 make_dropout_clip.py <clip_dir> <out_dir> [--start 5 --end 25 --drop-at 12 --drop-for 3]

Times are seconds from the clip's first message. /tf_static is kept and moved to the window start.
Writes <out_dir> (mcap) and prints the dropout window in bag time for the analysis.
"""
import argparse

import rosbag2_py

LIDAR = "/os1_cloud_node/points"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("out")
    ap.add_argument("--start", type=float, default=5.0)
    ap.add_argument("--end", type=float, default=25.0)
    ap.add_argument("--drop-at", type=float, default=12.0)
    ap.add_argument("--drop-for", type=float, default=3.0)
    args = ap.parse_args()
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=args.clip, storage_id="mcap"), rosbag2_py.ConverterOptions("cdr", "cdr"))
    writer = rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=args.out, storage_id="mcap"), rosbag2_py.ConverterOptions("cdr", "cdr"))
    for meta in reader.get_all_topics_and_types():
        writer.create_topic(meta)
    t0 = None
    kept = dropped = 0
    last_before = first_after = None
    while reader.has_next():
        topic, data, t = reader.read_next()
        t0 = t if t0 is None else t0
        rel = (t - t0) * 1e-9
        if topic == "/tf_static":
            writer.write(topic, data, t0 + int(args.start * 1e9))
            continue
        if not args.start <= rel < args.end:
            continue
        if topic == LIDAR:
            if args.drop_at <= rel < args.drop_at + args.drop_for:
                dropped += 1
                continue
            if rel < args.drop_at:
                last_before = t
            elif first_after is None:
                first_after = t
        writer.write(topic, data, t)
        kept += 1
    del writer  # closes the bag
    print(f"kept {kept} messages, dropped {dropped} lidar scans")
    print(f"dropout (bag time, ns): last lidar before {last_before}, first lidar after {first_after}, "
          f"gap {(first_after - last_before) * 1e-6:.0f} ms")


if __name__ == "__main__":
    main()
