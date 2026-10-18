"""Cut a time window out of a converted ROS 2 bag (all topics, /tf_static always kept) for the Jetson replay.

  python -m pipeline.make_clip --seq 00001 --scans 100:700 --out <data root>/bags/clips/00001_s100-700
"""
import argparse
import json
from pathlib import Path

from rosbags.rosbag2 import StoragePlugin, Writer
from rosbags.highlevel import AnyReader

from pipeline.rellis import ALIGN_DIR, BAGS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--scans", required=True, help="labelled scan range a:b; the clip spans their stamps +/- 1 s")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    lo, hi = (int(v) for v in args.scans.split(":"))
    st = {r["scan"]: r["stamp"] for r in json.loads((ALIGN_DIR / f"{args.seq}.json").read_text())["lidar"] if r["scan"] is not None}
    t0, t1 = int((st[lo] - 1.0) * 1e9), int((st[hi - 1] + 1.0) * 1e9)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with AnyReader([BAGS[args.seq]]) as reader, Writer(args.out, version=8, storage_plugin=StoragePlugin.MCAP) as writer:
        conns = {}
        for c in reader.connections:
            conns[c.id] = writer.add_connection(c.topic, c.msgtype, typestore=reader.typestore,
                                                serialization_format="cdr", offered_qos_profiles=c.ext.offered_qos_profiles)
        for c, t, raw in reader.messages():
            if c.topic == "/tf_static":
                writer.write(conns[c.id], t0, raw)  # log it at the clip start so playback has no idle lead-in
                n += 1
            elif t0 <= t <= t1:
                writer.write(conns[c.id], t, raw)
                n += 1
            elif t > t1:
                break
    print(f"{n} messages, {(t1 - t0) / 1e9:.1f} s -> {args.out}")


if __name__ == "__main__":
    main()
