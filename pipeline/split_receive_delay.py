"""Split the lidar node's receive delay into the sensor lag recorded in the bag and our own queueing.

On replay the nodes run on the bag's clock, so `receive_delay_ms` (sim now - header.stamp at the callback)
is the recorded lag (bag log time - header.stamp: sweep time + driver, inherent to the data) plus the time
the scan waited for our node. The CSV alone cannot separate the two; the clip's log times can.
Note: /clock is published at 40 Hz, so sim now under-reads by 0-25 ms.

  python -m pipeline.split_receive_delay --clip <data root>/bags/clips/00001_s100-400 \
      [--csv <jetson replay_out>/segmentation_lidar.csv] --out docs/measurements/jetson_receive_delay_run1.json
"""
import argparse
import csv
import datetime
import json
import struct
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader

LIDAR_TOPIC = "/os1_cloud_node/points"


def stats(v):
    v = np.asarray(v, float)
    return {"n": int(v.size), "median": round(float(np.median(v)), 2), "p95": round(float(np.percentile(v, 95)), 2),
            "max": round(float(v.max()), 2)} if v.size else {"n": 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", type=Path, required=True)
    ap.add_argument("--csv", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    lag = {}  # stamp (ms, rounded) -> bag log time - header.stamp (ms)
    with AnyReader([args.clip]) as reader:
        conns = [c for c in reader.connections if c.topic == LIDAR_TOPIC]
        for _, t, raw in reader.messages(connections=conns):
            sec, nsec = struct.unpack_from("<iI", raw, 4)  # CDR: 4-byte encapsulation, then header.stamp
            stamp = sec + nsec * 1e-9
            lag[round(stamp * 1e3)] = t * 1e-6 - stamp * 1e3
    out = {"date": datetime.date.today().isoformat(), "clip": args.clip.name, "topic": LIDAR_TOPIC,
           "recorded_lag_ms": stats(list(lag.values()))}
    if args.csv:
        rows = list(csv.DictReader(open(args.csv)))
        rec, queue, unmatched = [], [], 0
        for r in rows:
            k = round(float(r["stamp"]) * 1e3)
            if k not in lag:
                unmatched += 1
                continue
            rec.append(lag[k])
            queue.append(float(r["receive_delay_ms"]) - lag[k])
        out.update({"csv": str(args.csv), "rows": len(rows), "unmatched_rows": unmatched,
                    "scans_in_clip": len(lag), "processed_fraction": round(len(rec) / len(lag), 3),
                    "receive_delay_ms": stats([float(r["receive_delay_ms"]) for r in rows]),
                    "recorded_lag_of_processed_ms": stats(rec), "queueing_ms": stats(queue)})
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
