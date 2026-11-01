"""Per-scan safety error and runtime monitor signals on GOOSE (second benchmark of the monitor study), with
optional injected lidar faults. Same grid, cost table and error definition as the RELLIS-3D replays
(pipeline/run_replay.py), on single keyframes: GOOSE ships no sequences, so no grid memory, tracking or camera.

Frame: GOOSE scans are in the roof VLS-128 frame; base = lidar + LIDAR_HEIGHT_M up (median ground height of
labelled ground 4-10 m away on val, measured below). Ring index = nearest of the 128 measured elevation rows.

  python -m pipeline.goose_monitor_eval --run-dir <data root>/weights/salsanext_goose --split test --tag clean
  python -m pipeline.goose_monitor_eval ... --tag drop50 --fault lidar_drop=0.5
  python -m pipeline.goose_monitor_eval ... --split heldout:b --mc-passes 8 --ensemble <run dir> ...
-> <data root>/runs/goose_<split>_<tag>/frames.jsonl, metrics.json (":" in the split becomes "_")

Uncertainty baselines (2026-11-01), over lidar points inside the grid, normalised by log(classes):
entropy_grid (softmax entropy), mc_mi (MC-dropout mutual information, --mc-passes), ens_mi (deep-ensemble
mutual information over --run-dir plus the --ensemble members). The grid and the error always come from the
--run-dir model in eval mode, so these are monitors of that one deployed model.
"""
import argparse
import datetime
import json
import math
from pathlib import Path

import numpy as np
import torch
import yaml

from ml.lidar.eval_salsanext import load_model
from ml.lidar.goose import REPO, scans
from ml.lidar.range_projection import SensorConfig, project
from pipeline import faults as FAULTS
from pipeline import fusion as F
from pipeline.bev import GridParams, build_grid, load_costs
from pipeline.rellis import DATA_ROOT

LIDAR_HEIGHT_M = 2.09                 # VLS-128 above ground (median over val, labelled ground 4-10 m away)
EGO_BOX_LIDAR = {"x": (-1.5, -0.9), "y": (-0.7, 0.6), "z": (-0.7, -0.3)}  # roof rack seen by the lidar (ego_vehicle labels)
NOMINAL_POINTS = 187080               # median returns per clean val scan: density = valid returns / nominal


def ring_rows(sample, gap_deg=0.05):
    """128 elevation rows of the VLS-128, measured from clean scans: sorted elevations split at gaps."""
    p = []
    for b, _ in sample:
        s = np.fromfile(b, np.float32).reshape(-1, 4)
        r = np.linalg.norm(s[:, :3], axis=1)
        ok = r > 2
        p.append(np.degrees(np.arcsin(s[ok, 2] / r[ok])))
    p = np.sort(np.concatenate(p))
    edges = np.r_[0, np.flatnonzero(np.diff(p) > gap_deg) + 1, len(p)]
    return np.array([np.median(p[a:b]) for a, b in zip(edges[:-1], edges[1:])])


def mutual_info(p):
    """[members or passes, points, classes] -> per-point mutual information, normalised by log(classes)."""
    h = lambda q: -(q * torch.log(q.clamp(min=1e-9))).sum(-1)
    return (h(p.mean(0)) - h(p).mean(0)) / math.log(p.shape[-1])


def set_dropout(model, on):
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout2d):
            m.train(on)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--fault", action="append")
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--mc-passes", type=int, default=0, help="MC-dropout passes for mc_mi (0 = off)")
    ap.add_argument("--ensemble", type=Path, action="append", default=[], help="extra ensemble member run dirs")
    args = ap.parse_args()
    faults = FAULTS.parse(args.fault)
    out_dir = DATA_ROOT / f"runs/goose_{args.split.replace(':', '_')}_{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = SensorConfig.from_arch_cfg(args.run_dir / "arch_cfg.yaml")
    data = yaml.safe_load(open(args.run_dir / "data_cfg.yaml"))
    train_to_onto = np.array([data["learning_map_inv"][i] for i in range(len(data["learning_map_inv"]))])
    bucket = sorted(k for k, v in data["learning_map"].items() if v == 0)
    model = load_model(args.run_dir / "SalsaNext_valid_best", len(train_to_onto)).cuda().eval()
    members = []
    for d in args.ensemble:  # members must share the input normalisation (same training split)
        m_cfg = SensorConfig.from_arch_cfg(d / "arch_cfg.yaml")
        assert np.allclose(m_cfg.means, cfg.means) and np.allclose(m_cfg.stds, cfg.stds), f"{d}: other normalisation"
        members.append(load_model(d / "SalsaNext_valid_best", len(train_to_onto)).cuda().eval())
    goose_to_onto = yaml.safe_load(open(REPO / "config/goose_to_rellis.yaml"))
    g2o = np.zeros(1 << 16, np.int64)
    for g, o in goose_to_onto.items():
        g2o[g] = o
    onto_index = np.array([F.ONTO_INDEX.get(k, 0) for k in range(1 << 16)])
    cost_lut, _ = load_costs()
    gp = GridParams()
    c_ = (np.arange(gp.n) + 0.5) * gp.resolution_m - gp.extent_m / 2
    CX, CY = np.meshgrid(c_, c_)
    near = CX ** 2 + CY ** 2 <= 100
    corr = (np.abs(CY) <= 0.9) & (CX >= 3.0) & (CX <= 15.0)  # ahead of the car's front
    items = scans(args.split)[::args.every]
    rows = ring_rows(scans("val")[::60])
    assert len(rows) == 128, f"expected 128 VLS-128 rows, measured {len(rows)}"
    mids = (rows[1:] + rows[:-1]) / 2
    lift = np.array([0, 0, LIDAR_HEIGHT_M], np.float32)

    def to_base(xyz):
        e = EGO_BOX_LIDAR
        ego = ((xyz[:, 0] > e["x"][0]) & (xyz[:, 0] < e["x"][1]) & (xyz[:, 1] > e["y"][0]) & (xyz[:, 1] < e["y"][1])
               & (xyz[:, 2] > e["z"][0]) & (xyz[:, 2] < e["z"][1]))
        return xyz + lift, ~ego

    frames = open(out_dir / "frames.jsonl", "w")
    tot = {"gt_lethal": 0, "missed": 0, "unknown": 0}
    for i, (scan_p, label_p) in enumerate(items):
        s = np.fromfile(scan_p, np.float32).reshape(-1, 4)
        gt_onto = g2o[np.fromfile(label_p, np.uint32) & 0xFFFF]
        xyz, rem = s[:, :3].copy(), s[:, 3].copy()
        r = np.linalg.norm(xyz, axis=1)
        ring = np.digitize(np.degrees(np.arcsin(np.clip(xyz[:, 2] / np.maximum(r, 1e-6), -1, 1))), mids)
        cloud = {"xyz": xyz, "remission": rem, "range_mm": np.round(r * 1000).astype(np.uint32)}
        c = FAULTS.lidar(cloud, faults, np.random.default_rng(i), forward_deg=0.0, rings=ring, n_rings=128)
        valid = c["range_mm"] > 0
        proj, _pr, px, py, _d = project(c["xyz"][valid], c["remission"][valid], cfg)
        x = torch.from_numpy(proj)[None].cuda()
        iy, ix = torch.from_numpy(py).long().cuda(), torch.from_numpy(px).long().cuda()
        base, keep = to_base(c["xyz"][valid])
        in_grid = keep & (np.abs(base[:, 0]) < gp.extent_m / 2) & (np.abs(base[:, 1]) < gp.extent_m / 2)
        g = torch.from_numpy(in_grid).cuda()
        extra = {}
        with torch.no_grad():
            p_img = model(x)[0]
            p_pts = p_img[:, iy, ix].T.float()
            if args.mc_passes:
                set_dropout(model, True)  # Dropout2d draws a new mask for every batch element
                mc = model(x.expand(args.mc_passes, -1, -1, -1))[:, :, iy[g], ix[g]].permute(0, 2, 1).float()
                set_dropout(model, False)
                extra["mc_mi"] = round(float(mutual_info(mc).mean()), 5)
                del mc
            if members:
                ens = torch.stack([p_pts[g]] + [m(x)[0][:, iy[g], ix[g]].T.float() for m in members])
                extra["ens_mi"] = round(float(mutual_info(ens).mean()), 5)
                del ens
        p_lid = p_pts.cpu().numpy()
        probs = F.lidar_to_ontology(p_lid, train_to_onto, bucket)
        ent = -(p_lid * np.log(np.clip(p_lid, 1e-9, 1))).sum(1) / np.log(p_lid.shape[1])

        grid = build_grid(base[keep], probs[keep], np.zeros(keep.sum(), bool), gp, cost_lut)
        base_gt, keep_gt = to_base(xyz)
        onehot = np.zeros((keep_gt.sum(), len(F.ONTOLOGY_IDS)), np.float32)
        onehot[np.arange(keep_gt.sum()), onto_index[gt_onto[keep_gt]]] = 1.0
        grid_gt = build_grid(base_gt[keep_gt], onehot, np.zeros(keep_gt.sum(), bool), gp, cost_lut)

        gt_l = grid_gt.cost == 254
        err = {"gt_lethal": int(gt_l.sum()), "missed": int((gt_l & (grid.cost < 50)).sum()),
               "unknown": int((gt_l & (grid.cost == 255)).sum()), "gt_lethal_corridor": int((gt_l & corr).sum()),
               "missed_corridor": int((gt_l & corr & (grid.cost < 50)).sum())}
        monitor = {"density": round(float(valid.sum() / NOMINAL_POINTS), 4),
                   "near_coverage": round(float((grid.source_flags[near] > 0).mean()), 4),
                   "lidar_entropy": round(float(ent.mean()), 4),
                   "entropy_grid": round(float(ent[in_grid].mean()), 5), **extra}
        frames.write(json.dumps({"scan": i, "name": scan_p.stem, "sequence": scan_p.parent.name, "err": err,
                                 "monitor": monitor}) + "\n")
        for k in tot:
            tot[k] += err[k]
    frames.close()
    metrics = {"date": datetime.date.today().isoformat(), "split": args.split, "n_scans": len(items),
               "faults": faults, "run_dir": str(args.run_dir), "ensemble": [str(d) for d in args.ensemble],
               "mc_passes": args.mc_passes, "lidar_height_m": LIDAR_HEIGHT_M,
               "lethal_reported_free": round(tot["missed"] / max(1, tot["gt_lethal"]), 5),
               "lethal_reported_unknown": round(tot["unknown"] / max(1, tot["gt_lethal"]), 5),
               "gpu_peak_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
