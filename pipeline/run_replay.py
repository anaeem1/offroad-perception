"""Replay a converted RELLIS-3D bag through the full reference pipeline and record metrics.

Per lidar scan: lidar seg -> camera frame selection -> motion-compensated projection -> late fusion ->
deskew to base_link at the scan stamp -> BEV grid -> clustering -> tracker. Inputs come from the bag;
ground-truth labels come from the aligned SemanticKITTI scans (used for scoring only).

Fusion variants are scored in the same pass (cheap: only points inside the camera wedge differ):
lidar weight alpha x pooling rule x motion compensation x pitch offset. BEV, clustering and tracking run
for the selected default variant only.

  python -m pipeline.run_replay --seq 00001 --scans 1047:2319 --tag val
Outputs go to <data root>/runs/replay_<seq>_<tag>/ (frames.jsonl, tracks.jsonl, gt_objects.jsonl,
grids.npz, metrics.json).
"""
import argparse
import collections
import datetime
import itertools
import json
import time
from pathlib import Path

import cv2
import numpy as np
from rosbags.highlevel import AnyReader

from ml.fusion.projection import load_rellis_calib, perturb
from ml.metrics import ConfusionMatrix
from pipeline import faults as FAULTS
from pipeline import fusion as F
from pipeline.align_bag import BAYER_CODES
from pipeline.bev import GridParams, build_grid, load_costs, outside_ego
from pipeline.bev_memory import GridMemory, MemoryParams
from pipeline.cluster import cluster
from pipeline.extract_odom import load_odom
from pipeline.geometry import T_root_frame, apply, deskew, tf_graph
from pipeline.models import CameraSeg, LidarSeg
from pipeline.planner import decide as planner_decide
from pipeline.rellis import (ALIGN_DIR, BAGS, CAMERA_TOPIC, DATA_ROOT, LIDAR_TOPIC, decode_bayer, decode_cloud,
                             load_scan_labels, stamp_of)
from pipeline.tracker import COASTING, CONFIRMED, Tracker

SCORED = [3, 4, 5, 6, 8, 15, 17, 18, 19, 23, 27, 31, 33, 34]  # the 14 classes both models can output
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
POOLS = ["linear", "loglinear"]
PITCHES = [0.0, -0.5]


def scored_index(ids):
    """ontology ids -> index into SCORED, or len(SCORED) for 'not scored'."""
    lut = np.full(256, len(SCORED), np.int64)
    lut[SCORED] = np.arange(len(SCORED))
    return lut[ids]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True, choices=sorted(BAGS))
    ap.add_argument("--scans", default=None, help="labelled scan range a:b to process (default: all)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--pool", default="loglinear", choices=POOLS)  # chosen on the 00001 val block
    ap.add_argument("--pitch", type=float, default=-0.5, help="extrinsic pitch offset (deg) for the default variant")
    ap.add_argument("--no-motion-comp", action="store_true")
    ap.add_argument("--extent", type=float, default=40.0)
    ap.add_argument("--resolution", type=float, default=0.2)
    ap.add_argument("--debayer", default="BayerBG2RGB", choices=sorted(BAYER_CODES))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--save-grids", type=int, default=20, help="save every Nth grid (0: none)")
    ap.add_argument("--memory-s", type=float, default=3.0, help="grid memory max age (0 disables), ADR-0008")
    ap.add_argument("--fault", action="append", help="sensor fault to inject, e.g. lidar_drop=0.5 (pipeline/faults.py)")
    ap.add_argument("--dump-render", type=int, default=0, help="save demo-video render data for every Nth scan")
    ap.add_argument("--default-only", action="store_true", help="score only the default fusion variant (fast)")
    args = ap.parse_args()
    lo, hi = (int(v) for v in args.scans.split(":")) if args.scans else (0, 10 ** 9)
    out_dir = DATA_ROOT / f"runs/replay_{args.seq}_{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    align = json.loads((ALIGN_DIR / f"{args.seq}.json").read_text())
    msg_to_scan = {r["msg"]: r["scan"] for r in align["lidar"] if r["scan"] is not None}
    odom = load_odom(args.seq)
    insp = json.loads(Path(f"docs/measurements/bag_{args.seq}_inspection.json").read_text())
    T_base_lidar = T_root_frame(tf_graph(insp["tf_static"]), insp["pointcloud"]["frame_id"])
    forward_deg = float(np.degrees(np.arctan2(T_base_lidar[0, 1], T_base_lidar[0, 0])))  # base +x seen from the lidar
    cam, T_cam_lidar = load_rellis_calib(args.seq)
    faults = FAULTS.parse(args.fault)
    T_cam_lidar = FAULTS.extrinsic(T_cam_lidar, faults)  # drift: the extrinsic the pipeline believes is wrong
    if args.dump_render:
        (out_dir / "render").mkdir(exist_ok=True)
    T_cam = {pitch: perturb(T_cam_lidar, pitch_deg=pitch) for pitch in set(PITCHES + [args.pitch])}
    lidar, camera = LidarSeg(args.device), CameraSeg(args.device)
    cost_lut, obstacle_lut = load_costs()
    gp = GridParams(extent_m=args.extent, resolution_m=args.resolution)
    tracker = Tracker()
    memory = GridMemory(gp.resolution_m, MemoryParams(max_age_s=args.memory_s)) if args.memory_s > 0 else None
    cm_bev_single = ConfusionMatrix(len(SCORED) + 1, ignore=(len(SCORED),), device="cpu")
    c_ = (np.arange(gp.n) + 0.5) * gp.resolution_m - gp.extent_m / 2
    CX, CY = np.meshgrid(c_, c_)
    # Forward corridor from the vehicle front (ego box x_max): the ground under the machine is never visible.
    CORRIDOR = [(lo_, hi_, (np.abs(CY) < 0.9) & (CX >= lo_) & (CX < hi_)) for lo_, hi_ in
                ((0.9, 2), (2, 3), (3, 4), (4, 6), (6, 10), (10, 15))]
    AGES = [2.0, 3.0, 5.0, 8.0]  # memory ages scored in the same pass (selection on val)
    corridor_unknown = {(mode, lo_, hi_): [] for mode in ["single"] + [f"memory_{a:g}s" for a in AGES]
                        for lo_, hi_, _ in CORRIDOR}

    n_sc = len(SCORED) + 1
    variants = [(a, pool, mc, pitch) for a, pool, mc, pitch in itertools.product(ALPHAS, POOLS, (True, False), PITCHES)]
    if args.default_only:
        variants = [(args.alpha, args.pool, not args.no_motion_comp, args.pitch)]
    projections = {(mc, pitch) for _, _, mc, pitch in variants} | {(True, 0.0)}  # (True, 0.0) defines the wedge
    cm = {(region, v): ConfusionMatrix(n_sc, ignore=(len(SCORED),), device="cpu")
          for region in ("all", "wedge") for v in variants}
    cm_bev = ConfusionMatrix(n_sc, ignore=(len(SCORED),), device="cpu")
    trav = np.zeros((5, 5), np.int64)  # cost buckets: free(<50) low(<128) high(<254) lethal unknown
    planner_v, planner_reasons = [], collections.Counter()
    frames_f = open(out_dir / "frames.jsonl", "w")
    tracks_f = open(out_dir / "tracks.jsonl", "w")
    gt_f = open(out_dir / "gt_objects.jsonl", "w")
    det_f = open(out_dir / "detections.jsonl", "w")  # for offline tracker tuning
    saved_grids = {}
    cams = collections.deque(maxlen=10)
    timing = collections.defaultdict(list)
    n_done = 0

    import torch
    onto_lut = np.array([F.ONTO_INDEX.get(k, 0) for k in range(1 << 16)])
    pending = collections.deque()  # (scan, msg): held until camera frames beyond the sweep have arrived

    def process(scan, msg):
            nonlocal n_done
            t0 = time.perf_counter()
            stamp = stamp_of(msg)
            c_clean = decode_cloud(msg)
            c = FAULTS.lidar(c_clean, faults, np.random.default_rng(scan), forward_deg)
            xyz, valid = c["xyz"], c["range_mm"] > 0
            point_t = stamp + c["t_ns"] * 1e-9
            p_lid = lidar(xyz, c["remission"])
            t1 = time.perf_counter()

            # Camera frame nearest to when the lidar swept the camera wedge.
            uv0, idx0 = F.camera_points(xyz, point_t, valid, cam, T_cam_lidar)
            t_wedge = float(np.median(point_t[idx0])) if len(idx0) else stamp
            t_c, cmsg = min(cams, key=lambda x: abs(x[0] - t_wedge)) if cams else (None, None)
            have_cam = cmsg is not None and abs(t_c - t_wedge) <= 0.05  # +/-50 ms of when the wedge was swept
            p_l_onto = F.lidar_to_ontology(p_lid, lidar.train_to_onto, lidar.bucket)
            if have_cam:
                rgb = FAULTS.image(cv2.cvtColor(decode_bayer(cmsg), BAYER_CODES[args.debayer]), faults)
                p_cam = camera(rgb)
            t2 = time.perf_counter()

            gt = load_scan_labels(args.seq, scan)
            gt_sc = scored_index(gt)
            lid_lab = F.lidar_labels(p_lid, lidar.train_to_onto)
            proj = {}
            if have_cam:
                for mc, pitch in projections:
                    proj[(mc, pitch)] = F.camera_points(xyz, point_t, valid, cam, T_cam[pitch],
                                                        odom if mc else None, T_base_lidar, t_c)
            wedge = np.zeros(len(xyz), bool)
            if have_cam:
                wedge[proj[(True, 0.0)][1]] = True
            for v in variants:
                a, pool, mc, pitch = v
                lab = lid_lab.copy()
                if have_cam and a < 1.0:
                    uv, idx = proj[(mc, pitch)]
                    pc = F.camera_to_ontology(F.sample_camera(p_cam, uv), camera.class_ids)
                    lab[idx] = F.ONTOLOGY_IDS[F.fuse(p_l_onto[idx], pc, a, pool).argmax(1)]
                pr = torch.from_numpy(scored_index(lab))
                g = torch.from_numpy(gt_sc)
                vm = torch.from_numpy(valid)
                cm[("all", v)].update(pr[vm], g[vm])
                wm = torch.from_numpy(wedge & valid)
                cm[("wedge", v)].update(pr[wm], g[wm])
            t3 = time.perf_counter()

            # Default variant -> fused distributions for every valid point.
            cross_disagree = float("nan")  # camera vs lidar argmax inside the camera view (monitor signal)
            probs = p_l_onto.copy()
            from_cam = np.zeros(len(xyz), bool)
            if have_cam and args.alpha < 1.0:
                uv, idx = proj.get((not args.no_motion_comp, args.pitch)) or F.camera_points(
                    xyz, point_t, valid, cam, T_cam[args.pitch], None if args.no_motion_comp else odom, T_base_lidar, t_c)
                pc = F.camera_to_ontology(F.sample_camera(p_cam, uv), camera.class_ids)
                cross_disagree = float(np.mean(pc.argmax(1) != p_l_onto[idx].argmax(1))) if len(idx) else float("nan")
                probs[idx] = F.fuse(p_l_onto[idx], pc, args.alpha, args.pool)
                from_cam[idx] = True
            # Lidar-only points: argmax in the lidar's own space (bucket -> void), as a one-hot-ish distribution.
            vi = np.flatnonzero(valid)
            xyz_base = apply(T_base_lidar, deskew(xyz[vi], point_t[vi], odom, T_base_lidar, stamp))
            keep = outside_ego(xyz_base)  # drop the machine's own returns before grid + clustering
            vi, xyz_base = vi[keep], xyz_base[keep]
            grid = build_grid(xyz_base, probs[vi], from_cam[vi], gp, cost_lut)
            dets = cluster(xyz_base, probs[vi], grid.geometric, gp, obstacle_lut)
            T_odom_base = odom.at(stamp)
            single_cost, single_cls, single_flags = grid.cost.copy(), grid.semantic_class.copy(), grid.source_flags.copy()
            n_remembered = 0
            if memory is not None:
                rows_, cols_ = memory._keys(grid, T_odom_base)
                age_ = (stamp - memory.t_seen[rows_, cols_]).reshape(grid.cost.shape)
                for a in AGES:  # what each candidate age would leave UNKNOWN
                    unk = (single_flags == 0) & ~(age_ <= a)
                    for lo_, hi_, m_ in CORRIDOR:
                        corridor_unknown[(f"memory_{a:g}s", lo_, hi_)].append(float(unk[m_].mean()))
                # Monitor signal: remembered-lethal cells that this sweep now reports free.
                both = (single_flags.ravel() > 0) & (age_.ravel() <= args.memory_s)
                mem_lethal = both & (memory.cost[rows_, cols_] == 254)
                memory_flip = float(np.mean(single_cost.ravel()[mem_lethal] < 50)) if mem_lethal.any() else 0.0
                n_remembered = memory.fill(grid, T_odom_base, stamp)  # old static evidence where this sweep is blind
                memory.update(grid, T_odom_base, stamp)
            for lo_, hi_, m_ in CORRIDOR:
                corridor_unknown[("single", lo_, hi_)].append(float((single_cost[m_] == 255).mean()))
            t4 = time.perf_counter()

            dets_odom = [(apply(T_odom_base, d.xyz[None])[0], d.dims, d.semantic_class, d.class_confidence) for d in dets]
            live = tracker.step(stamp, dets_odom)
            for d, (xo, *_rest) in zip(dets, dets_odom):
                det_f.write(json.dumps({"scan": scan, "t": stamp, "x": float(xo[0]), "y": float(xo[1]), "z": float(xo[2]),
                                        "dims": [float(v) for v in d.dims], "cls": d.semantic_class,
                                        "conf": round(d.class_confidence, 4), "n": d.n_points,
                                        "range": float(np.hypot(*d.xyz[:2]))}) + "\n")
            t5 = time.perf_counter()

            # Ground truth: BEV grid and obstacle clusters from the labels, same code path, on the clean cloud.
            vg, base_g = vi, xyz_base
            if c is not c_clean:
                vg = np.flatnonzero(c_clean["range_mm"] > 0)
                base_g = apply(T_base_lidar, deskew(c_clean["xyz"][vg], point_t[vg], odom, T_base_lidar, stamp))
                keep_g = outside_ego(base_g)
                vg, base_g = vg[keep_g], base_g[keep_g]
            onehot = np.zeros((len(vg), len(F.ONTOLOGY_IDS)), np.float32)
            onehot[np.arange(len(vg)), onto_lut[gt[vg]]] = 1.0
            grid_gt = build_grid(base_g, onehot, np.zeros(len(vg), bool), gp, cost_lut)
            dets_gt = cluster(base_g, onehot, np.zeros_like(grid_gt.geometric), gp, obstacle_lut)
            # BEV IoU: GT is this sweep's labels, so it can only score cells this sweep observed.
            both = (single_flags > 0) & (grid_gt.source_flags > 0)
            cm_bev_single.update(torch.from_numpy(scored_index(single_cls[both])),
                                 torch.from_numpy(scored_index(grid_gt.semantic_class[both])))
            both = (grid.source_flags > 0) & (grid_gt.source_flags > 0)
            cm_bev.update(torch.from_numpy(scored_index(grid.semantic_class[both])),
                          torch.from_numpy(scored_index(grid_gt.semantic_class[both])))
            bucket = lambda cst: np.select([cst < 50, cst < 128, cst < 254, cst == 254], [0, 1, 2, 3], 4)
            np.add.at(trav, (bucket(grid_gt.cost.ravel()), bucket(grid.cost.ravel())), 1)
            # Per-scan safety error (truly lethal reported free) and cheap runtime monitor signals, for monitor studies.
            gt_l = grid_gt.cost == 254
            corr = (np.abs(CY) <= 0.9) & (CX >= 0.9) & (CX <= 15.0)
            pv = p_lid[valid]
            ent = -(pv * np.log(np.clip(pv, 1e-9, 1))).sum(1) / np.log(pv.shape[1])
            err = {"gt_lethal": int(gt_l.sum()), "missed": int((gt_l & (grid.cost < 50)).sum()),
                   "unknown": int((gt_l & (grid.cost == 255)).sum()), "gt_lethal_corridor": int((gt_l & corr).sum()),
                   "missed_corridor": int((gt_l & corr & (grid.cost < 50)).sum())}
            monitor = {"density": round(float(valid.mean()), 4),
                       "near_coverage": round(float((single_flags[(CX ** 2 + CY ** 2) <= 100] > 0).mean()), 4),
                       "lidar_entropy": round(float(ent.mean()), 4), "cross_disagree": round(cross_disagree, 4),
                       "memory_flip": round(memory_flip, 4) if memory is not None else None}

            for tr in live:
                if tr.state in (CONFIRMED, COASTING):
                    tracks_f.write(json.dumps({"scan": scan, "t": stamp, "id": tr.id, "state": tr.state,
                                               "x": float(tr.x[0]), "y": float(tr.x[1]), "vx": float(tr.x[2]),
                                               "vy": float(tr.x[3]), "cls": int(tr.semantic_class),
                                               "existence": round(tr.existence, 3)}) + "\n")
            for d in dets_gt:
                xo = apply(T_odom_base, d.xyz[None])[0]
                gt_f.write(json.dumps({"scan": scan, "t": stamp, "x": float(xo[0]), "y": float(xo[1]),
                                       "cls": d.semantic_class, "n": d.n_points,
                                       "range": float(np.hypot(*d.xyz[:2]))}) + "\n")
            T_base_odom = np.linalg.inv(T_odom_base)
            trk_base = [(*apply(T_base_odom, np.array([[tr.x[0], tr.x[1], T_odom_base[2, 3]]]))[0][:2], *tr.dims[:2],
                         tr.state, tr.existence) for tr in live if tr.state in (CONFIRMED, COASTING)]
            v_cmd, why, d_blocked, d_obstacle = planner_decide(grid.cost, CX, CY, trk_base, have_cam)
            planner_v.append(v_cmd)
            planner_reasons[why] += 1
            frames_f.write(json.dumps({
                "err": err, "monitor": monitor,
                "planner_v": round(v_cmd, 3), "planner_reason": why, "d_blocked_m": round(d_blocked, 2),
                "scan": scan, "t": stamp, "camera_used": bool(have_cam),
                "cam_minus_scan_ms": None if t_c is None else round(1e3 * (t_c - stamp), 1),
                "cam_minus_wedge_ms": None if t_c is None else round(1e3 * (t_c - t_wedge), 1),
                "n_valid": int(valid.sum()), "n_wedge": int(wedge.sum()), "n_dets": len(dets),
                "n_dets_gt": len(dets_gt), "n_tracks": sum(t.state != 0 for t in live),
                "observed_cells": int((grid.source_flags > 0).sum()), "lethal_cells": int((grid.cost == 254).sum()),
                "geometric_cells": int(grid.geometric.sum()), "remembered_cells": n_remembered}) + "\n")
            if args.dump_render and n_done % args.dump_render == 0:  # demo video (pipeline/render_demo.py)
                fused = have_cam and args.alpha < 1.0
                jpg = cv2.imencode(".jpg", cv2.cvtColor(cv2.resize(rgb, (960, 600)), cv2.COLOR_RGB2BGR))[1] if have_cam else []
                trk = []
                for tr in live:
                    if tr.state in (CONFIRMED, COASTING):
                        p_b = apply(T_base_odom, np.array([[tr.x[0], tr.x[1], T_odom_base[2, 3]]]))[0]
                        v_b = T_base_odom[:2, :2] @ np.asarray(tr.x[2:4])
                        trk.append([tr.id, tr.state, p_b[0], p_b[1], v_b[0], v_b[1], tr.semantic_class, tr.existence])
                np.savez_compressed(
                    out_dir / "render" / f"{scan:06d}.npz", jpg=np.asarray(jpg, np.uint8).ravel(),
                    uv=uv.astype(np.float32) if fused else np.zeros((0, 2), np.float32),
                    uv_cls=(F.ONTOLOGY_IDS[probs[idx].argmax(1)] if fused else np.zeros(0)).astype(np.uint8),
                    cost=grid.cost, cls=grid.semantic_class, flags=grid.source_flags,
                    tracks=np.array(trk, np.float64).reshape(-1, 8),
                    dets=np.array([[*d.xyz[:2], *d.dims[:2], d.semantic_class] for d in dets], np.float64).reshape(-1, 5),
                    meta=np.array([stamp, have_cam, np.nan if t_c is None else 1e3 * (t_c - t_wedge), n_remembered,
                                   1e3 * (t5 - t0), v_cmd, d_blocked]), reason=np.array(why))
            if args.save_grids and n_done % args.save_grids == 0:
                saved_grids[scan] = np.stack([grid.cost, grid.semantic_class, grid.source_flags, grid_gt.cost,
                                              grid_gt.semantic_class])
            for k, dt in zip(("lidar_model", "camera_select_model", "fusion_variants", "fusion_bev_cluster",
                              "tracker"), (t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4)):
                timing[k].append(1e3 * dt)
            n_done += 1

    with AnyReader([BAGS[args.seq]]) as reader:
        conns = [c for c in reader.connections if c.topic in (LIDAR_TOPIC, CAMERA_TOPIC)]
        n_lidar = -1
        for conn, _, raw in reader.messages(connections=conns):
            if conn.topic == CAMERA_TOPIC:
                m = reader.deserialize(raw, conn.msgtype)
                cams.append((FAULTS.camera_stamp(stamp_of(m), faults), m))
                while pending and stamp_of(pending[0][1]) + 0.2 < cams[-1][0]:
                    process(*pending.popleft())
                continue
            n_lidar += 1
            scan = msg_to_scan.get(n_lidar)
            if scan is None or not lo <= scan < hi:
                if scan is not None and scan >= hi and not pending:
                    break
                continue
            pending.append((scan, reader.deserialize(raw, conn.msgtype)))
        while pending:
            process(*pending.popleft())


    onto_names = dict(zip(F.ONTOLOGY_IDS.tolist(), ["void", "dirt", "grass", "tree", "pole", "water", "sky",
                      "vehicle", "object", "asphalt", "building", "log", "person", "fence", "bush", "concrete",
                      "barrier", "puddle", "mud", "rubble"]))

    def summary(m):
        iou = m.iou("strict")
        sup = m.support()
        return {"miou": round(100 * float(np.mean([v for c, v in iou.items() if sup[c] > 0])), 2),
                "per_class_iou": {onto_names[SCORED[c]]: round(100 * v, 2) for c, v in iou.items() if sup[c] > 0},
                "support": {onto_names[SCORED[c]]: sup[c] for c in iou}}

    key = lambda v: f"alpha={v[0]}|{v[1]}|mc={'on' if v[2] else 'off'}|pitch={v[3]}"
    metrics = {
        "seq": args.seq, "scans": f"{lo}:{min(hi, lo + n_done)}", "n_scans": n_done, "tag": args.tag,
        "date": datetime.date.today().isoformat(),
        "default_variant": key((args.alpha, args.pool, not args.no_motion_comp, args.pitch)),
        "grid": {"extent_m": gp.extent_m, "resolution_m": gp.resolution_m,
                 "obstacle_height_m": gp.obstacle_height_m},
        "point_iou": {region: {key(v): summary(cm[(region, v)]) for v in variants} for region in ("all", "wedge")},
        "bev_cell_iou_default": summary(cm_bev_single),
        "bev_cell_iou_with_memory_on_currently_observed_gt": summary(cm_bev),
        "memory_max_age_s": args.memory_s,
        "faults": faults,
        "corridor_unknown_fraction": {f"{mode}|{lo_}-{hi_}m": round(float(np.mean(v)), 3)
                                      for (mode, lo_, hi_), v in corridor_unknown.items()},
        "traversability_confusion_gt_rows_pred_cols": {"buckets": ["free<50", "low<128", "high<254", "lethal", "unknown"],
                                                       "counts": trav.tolist()},
        "timing_ms_median_server": {k: round(float(np.median(v)), 2) for k, v in timing.items()},
        "planner_offline": {"fraction_moving": round(float(np.mean(np.array(planner_v) > 0)), 4),
                            "median_v_mps": round(float(np.median(planner_v)), 3),
                            "reasons": dict(planner_reasons.most_common())},
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    if saved_grids:
        np.savez_compressed(out_dir / "grids.npz", **{str(k): v for k, v in saved_grids.items()})
    best = max(variants, key=lambda v: metrics["point_iou"]["wedge"][key(v)]["miou"])
    print(f"{n_done} scans -> {out_dir}")
    print("default", metrics["default_variant"], "all", metrics["point_iou"]["all"][metrics["default_variant"]]["miou"],
          "wedge", metrics["point_iou"]["wedge"][metrics["default_variant"]]["miou"])
    print("best wedge variant", key(best), metrics["point_iou"]["wedge"][key(best)]["miou"])
    print("bev cell mIoU", metrics["bev_cell_iou_default"]["miou"], "| timing", metrics["timing_ms_median_server"])


if __name__ == "__main__":
    main()
