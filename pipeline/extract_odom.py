"""Cache /odometry/filtered (odom -> base_link) from a converted bag as a small npz, so the replay
pipeline can interpolate ego pose at any point time without another pass over the bag.

  python -m pipeline.extract_odom --seq 00001   ->  <data root>/rellis3d/alignment/<seq>_odom.npz
"""
import argparse

import numpy as np
from rosbags.highlevel import AnyReader

from pipeline.geometry import PoseTrack
from pipeline.rellis import ALIGN_DIR, BAGS, ODOM_TOPIC, stamp_of


def load_odom(seq):
    d = np.load(ALIGN_DIR / f"{seq}_odom.npz")
    return PoseTrack(d["t"], d["p"], d["q"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True, choices=sorted(BAGS))
    args = ap.parse_args()
    t, p, q, frames = [], [], [], set()
    with AnyReader([BAGS[args.seq]]) as reader:
        conns = [c for c in reader.connections if c.topic == ODOM_TOPIC]
        for conn, _, raw in reader.messages(connections=conns):
            m = reader.deserialize(raw, conn.msgtype)
            pose = m.pose.pose
            t.append(stamp_of(m))
            p.append([pose.position.x, pose.position.y, pose.position.z])
            q.append([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
            frames.add((m.header.frame_id, m.child_frame_id))
    ALIGN_DIR.mkdir(parents=True, exist_ok=True)
    out = ALIGN_DIR / f"{args.seq}_odom.npz"
    np.savez(out, t=np.array(t), p=np.array(p), q=np.array(q))
    t = np.array(t)
    print(f"{len(t)} poses, frames {sorted(frames)}, span {t.max() - t.min():.1f} s, "
          f"median dt {1e3 * np.median(np.diff(np.sort(t))):.1f} ms, path length "
          f"{np.linalg.norm(np.diff(np.array(p)[np.argsort(t)], axis=0), axis=1).sum():.1f} m -> {out}")


if __name__ == "__main__":
    main()
