"""Summarize a ROS 1 .bag or ROS 2 bag directory without ROS installed: topic, type, count, mean rate.

  python scripts/bag_info.py <bag_or_dir> [--json out.json]
"""
import argparse
import json
from pathlib import Path

from rosbags.highlevel import AnyReader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bag", type=Path)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    paths = sorted(args.bag.glob("*.bag")) if args.bag.is_dir() and not (args.bag / "metadata.yaml").exists() \
        else [args.bag]
    with AnyReader(paths) as reader:
        span_s = (reader.end_time - reader.start_time) / 1e9
        rows = []
        for topic in sorted({c.topic for c in reader.connections}):
            conns = [c for c in reader.connections if c.topic == topic]
            count = sum(c.msgcount for c in conns)
            rows.append({"topic": topic, "type": conns[0].msgtype, "count": count,
                         "rate_hz": round(count / span_s, 2) if span_s > 0 else None})
    print(f"{args.bag}  duration {span_s:.1f} s, {len(rows)} topics")
    for r in rows:
        print(f"  {r['topic']:45s} {r['type']:35s} {r['count']:7d}  {r['rate_hz']} Hz")
    if args.json:
        args.json.write_text(json.dumps({"bag": str(args.bag), "duration_s": round(span_s, 2), "topics": rows},
                                        indent=2) + "\n")


if __name__ == "__main__":
    main()
