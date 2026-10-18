"""Dump stage-by-stage inputs and reference outputs from the Python pipeline for the C++ parity test.

  python -m pipeline.parity_dump --seq 00001 --scans 300:360 --stage-scans 3 --out <data root>/parity/00001
  <build>/parity <out>            # C++ writes cpp_*.npy next to the py_*.npy files
  python -m pipeline.parity_compare --dump <out> --json docs/measurements/cpp_parity.json
"""
import argparse
import collections
import json
from pathlib import Path

import cv2
import numpy as np
from rosbags.highlevel import AnyReader

from ml.fusion.projection import load_rellis_calib, perturb
from ml.lidar.range_projection import project
from pipeline import fusion as F
from pipeline.align_bag import BAYER_CODES
from pipeline.bev import GridParams, build_grid, load_costs, outside_ego
from pipeline.cluster import cluster
from pipeline.extract_odom import load_odom
from pipeline.geometry import T_root_frame, apply, deskew, tf_graph
from pipeline.models import CameraSeg, LidarSeg
from pipeline.rellis import ALIGN_DIR, BAGS, CAMERA_TOPIC, LIDAR_TOPIC, decode_bayer, decode_cloud, stamp_of
from pipeline.tracker import Tracker

ALPHA, LOGLINEAR, PITCH_DEG = 0.5, True, -0.5  # the shipped fusion config (selected on 00001 val)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--scans", required=True)
    ap.add_argument("--stage-scans", type=int, default=3)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    lo, hi = (int(v) for v in args.scans.split(":"))
    common = args.out / "common"
    common.mkdir(parents=True, exist_ok=True)
    (args.out / "tracker").mkdir(exist_ok=True)

    align = json.loads((ALIGN_DIR / f"{args.seq}.json").read_text())
    msg_to_scan = {r["msg"]: r["scan"] for r in align["lidar"] if r["scan"] is not None}
    odom = load_odom(args.seq)
    insp = json.loads(Path(f"docs/measurements/bag_{args.seq}_inspection.json").read_text())
    T_bl = T_root_frame(tf_graph(insp["tf_static"]), insp["pointcloud"]["frame_id"])
    cam, T_cl = load_rellis_calib(args.seq)
    T_cl = perturb(T_cl, pitch_deg=PITCH_DEG)  # shipped extrinsic includes the measured pitch correction
    lidar, camera = LidarSeg(args.device), CameraSeg(args.device)
    cost_lut, obstacle_lut = load_costs()
    gp = GridParams()

    np.save(common / "odom_t.npy", odom.t)
    np.save(common / "odom_p.npy", odom.p)
    np.save(common / "odom_q.npy", odom.slerp(odom.t).as_quat())
    np.save(common / "T_base_lidar.npy", T_bl)
    np.save(common / "T_cam_lidar.npy", T_cl)
    K = cam.K
    np.save(common / "cam.npy", np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2], *cam.D, cam.width, cam.height], np.float64))
    np.save(common / "train_to_onto.npy", lidar.train_to_onto.astype(np.int32))
    np.save(common / "bucket.npy", np.array(lidar.bucket, np.int32))
    np.save(common / "cam_class_ids.npy", camera.class_ids.astype(np.int32))
    np.save(common / "cost_lut.npy", cost_lut.astype(np.int32))
    np.save(common / "obstacle_lut.npy", obstacle_lut.astype(np.uint8))
    np.save(common / "fusion.npy", np.array([ALPHA, float(LOGLINEAR), gp.extent_m, gp.resolution_m]))

    tracker, det_rows, track_rows, step_t = Tracker(), [], [], []
    cams, pending, n_stage = collections.deque(maxlen=10), collections.deque(), 0

    def process(scan, msg):
        nonlocal n_stage
        stamp = stamp_of(msg)
        c = decode_cloud(msg)
        xyz, valid, rem = c["xyz"], c["range_mm"] > 0, c["remission"]
        point_t = stamp + c["t_ns"] * 1e-9
        p_lid = lidar(xyz, rem)
        _, idx0 = F.camera_points(xyz, point_t, valid, cam, T_cl)
        t_wedge = float(np.median(point_t[idx0]))
        t_c, cmsg = min(cams, key=lambda x: abs(x[0] - t_wedge))
        p_cam = camera(cv2.cvtColor(decode_bayer(cmsg), BAYER_CODES["BayerBG2RGB"]))
        vi = np.flatnonzero(valid)
        pts_cam_time = deskew(xyz[vi], point_t[vi], odom, T_bl, t_c)
        from ml.fusion.projection import project_points
        uv, i_hit, _ = project_points(pts_cam_time, cam, T_cl)
        hit_idx = vi[i_hit]
        probs = F.lidar_to_ontology(p_lid, lidar.train_to_onto, lidar.bucket)
        pc = F.camera_to_ontology(F.sample_camera(p_cam, uv), camera.class_ids)
        probs[hit_idx] = F.fuse(probs[hit_idx], pc, ALPHA, "loglinear" if LOGLINEAR else "linear")
        from_cam = np.zeros(len(xyz), bool)
        from_cam[hit_idx] = True
        xyz_base = apply(T_bl, deskew(xyz[vi], point_t[vi], odom, T_bl, stamp))
        keep = outside_ego(xyz_base)
        vb, xyz_base = vi[keep], xyz_base[keep]
        grid = build_grid(xyz_base, probs[vb], from_cam[vb], gp, cost_lut)
        dets = cluster(xyz_base, probs[vb], grid.geometric, gp, obstacle_lut)

        if n_stage < args.stage_scans:
            d = args.out / f"scan_{scan:06d}"
            d.mkdir(exist_ok=True)
            proj, *_ = project(xyz, rem, lidar.cfg)
            for name, arr in {"in_xyz": xyz.astype(np.float32), "in_rem": rem.astype(np.float32),
                              "in_point_t": point_t, "in_valid": valid.astype(np.uint8),
                              "in_t_cam": np.array([t_c]), "in_stamp": np.array([stamp]),
                              "in_plid": p_lid.astype(np.float32), "in_pcam": p_cam.astype(np.float32),
                              "py_range_input": proj, "py_deskew_cam": pts_cam_time.astype(np.float64),
                              "py_hits_uv": uv.astype(np.float64), "py_hits_idx": hit_idx.astype(np.int64),
                              "py_probs": probs[vi].astype(np.float32), "py_from_cam": from_cam[vb].astype(np.uint8),
                              "py_probs_base": probs[vb].astype(np.float32),
                              "py_xyz_base": xyz_base.astype(np.float64), "py_grid_cost": grid.cost,
                              "py_grid_conf": grid.confidence, "py_grid_class": grid.semantic_class,
                              "py_grid_flags": grid.source_flags, "py_grid_geometric": grid.geometric.astype(np.uint8),
                              "py_dets": np.array([[*dt.xyz, *dt.dims, dt.semantic_class, dt.class_confidence, dt.n_points]
                                                   for dt in dets], np.float64).reshape(-1, 9)}.items():
                np.save(d / f"{name}.npy", arr)
            n_stage += 1

        T_ob = odom.at(stamp)
        step = len(step_t)
        step_t.append(stamp)
        inputs = []
        for dt in dets:
            xo = apply(T_ob, dt.xyz[None])[0]
            det_rows.append([step, stamp, *xo, *dt.dims, dt.semantic_class, dt.class_confidence])
            inputs.append((xo, dt.dims, dt.semantic_class, dt.class_confidence))
        for tr in tracker.step(stamp, inputs):
            track_rows.append([step, tr.id, tr.state, *tr.x, tr.existence, tr.semantic_class])
        print(f"scan {scan}: {len(dets)} dets, {len(tracker.tracks)} tracks", flush=True)

    with AnyReader([BAGS[args.seq]]) as reader:
        conns = [c for c in reader.connections if c.topic in (LIDAR_TOPIC, CAMERA_TOPIC)]
        n_lidar = -1
        for conn, _, raw in reader.messages(connections=conns):
            if conn.topic == CAMERA_TOPIC:
                m = reader.deserialize(raw, conn.msgtype)
                cams.append((stamp_of(m), m))
                while pending and stamp_of(pending[0][1]) + 0.2 < cams[-1][0]:
                    process(*pending.popleft())
                continue
            n_lidar += 1
            scan = msg_to_scan.get(n_lidar)
            if scan is not None and lo <= scan < hi:
                pending.append((scan, reader.deserialize(raw, conn.msgtype)))
            elif scan is not None and scan >= hi and not pending:
                break
        while pending:
            process(*pending.popleft())

    np.save(args.out / "tracker/det_seq.npy", np.array(det_rows, np.float64).reshape(-1, 10))
    np.save(args.out / "tracker/step_t.npy", np.array(step_t, np.float64))
    np.save(args.out / "tracker/py_tracks.npy", np.array(track_rows, np.float64).reshape(-1, 9))
    print("dumped to", args.out)


if __name__ == "__main__":
    main()
