"""Train SalsaNext on GOOSE lidar scans (Velodyne VLS-128) in the RELLIS-3D class set, so the network drops into
the existing pipeline for the monitor study's second benchmark (ml/lidar/goose.py).

Recipe after SalsaNext's own training: SGD (momentum 0.9, weight decay 1e-4), 1 warm-up epoch then cosine decay,
loss = class-weighted cross-entropy + Lovasz-softmax, yaw-rotation and mirror augmentation. Selection on the GOOSE
val split by mIoU over the 14 RELLIS training classes (void ignored, as in eval_salsanext.py).

  python -m ml.lidar.train_salsanext --epochs 40 --out <data root>/weights/salsanext_goose
writes <out>/arch_cfg.yaml (VLS-128 sensor + measured normalization), data_cfg.yaml (RELLIS class set),
SalsaNext_valid_best (state_dict, same format as the released RELLIS checkpoint), train_log.jsonl
"""
import argparse
import json
import math
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn
import yaml
from torch.utils.data import DataLoader

from ml.lidar.eval_salsanext import RUN_DIR
from ml.lidar.goose import VLS128, GooseLidar
from ml.lidar.range_projection import SensorConfig
from ml.lidar.salsanext import SalsaNext
from ml.metrics import ConfusionMatrix


def lovasz_softmax(probs, labels, ignore=0):
    """Lovasz-softmax loss (Berman et al., CVPR 2018) over the pixels whose label is not `ignore`.
    probs (B,C,H,W) softmax output, labels (B,H,W)."""
    C = probs.shape[1]
    p = probs.permute(0, 2, 3, 1).reshape(-1, C)
    y = labels.reshape(-1)
    keep = y != ignore
    p, y = p[keep], y[keep]
    if len(y) == 0:
        return probs.sum() * 0
    losses = []
    for c in range(1, C):
        fg = (y == c).float()
        if fg.sum() == 0:
            continue
        err = (fg - p[:, c]).abs()
        err_sorted, perm = torch.sort(err, descending=True)
        fg_sorted = fg[perm]
        inter = fg_sorted.sum() - fg_sorted.cumsum(0)
        union = fg_sorted.sum() + (1 - fg_sorted).cumsum(0)
        jac = 1 - inter / union
        jac[1:] = jac[1:] - jac[:-1]  # gradient of the Lovasz extension
        losses.append(torch.dot(err_sorted, jac))
    return torch.stack(losses).mean() if losses else probs.sum() * 0


def pixel_labels(batch):
    """(proj, proj_range, px, py, depth, labels) per item -> input images and per-pixel labels, nearest point wins
    each pixel as in the projection itself."""
    imgs, labs = [], []
    for proj, _r, px, py, depth, lab in batch:
        H, W = proj.shape[1:]
        img = torch.zeros(H, W, dtype=torch.long)
        order = torch.argsort(depth, descending=True)
        img[py[order], px[order]] = lab[order]
        imgs.append(proj)
        labs.append(img)
    return torch.stack(imgs), torch.stack(labs)


def measure_norm(ds, n=300):
    """Mean and std of range, x, y, z, remission over occupied pixels of n training scans (raw: identity cfg)."""
    acc = []
    for i in np.linspace(0, len(ds) - 1, n).astype(int):
        proj, proj_range = ds[i][0].numpy(), ds[i][1].numpy()
        acc.append(proj[:, proj_range > 0].T)
    a = np.concatenate(acc)
    return a.mean(0), a.std(0)


def seed_worker(_):
    """Forked workers inherit one copy of the dataset's augmentation stream; give each its own (from torch's
    per-worker seed, so --seed still fixes it). Runs before 2026-11-01 drew the same angles in every worker."""
    info = torch.utils.data.get_worker_info()
    info.dataset.rng = np.random.default_rng(info.seed % 2 ** 32)


def evaluate(model, ds, n_classes, device, every=1):
    cm = ConfusionMatrix(n_classes, device=device)
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, len(ds), every):
            proj, _r, px, py, _d, lab = ds[i]
            probs = model(proj[None].to(device))
            pred = probs[0].argmax(0)[py.to(device), px.to(device)]
            cm.update(pred, lab.to(device))
    model.train()
    iou = cm.iou("upstream")
    return float(np.mean(list(iou.values()))), iou


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--val-every", type=int, default=2, help="epochs between validations")
    ap.add_argument("--limit", type=int, default=None, help="cap scans per split (smoke tests)")
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--val-split", default="val")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    device = "cuda"
    data_cfg = yaml.safe_load(open(RUN_DIR / "data_cfg.yaml"))
    n_classes = len(data_cfg["learning_map_inv"])
    lmap = data_cfg["learning_map"]

    raw_cfg = SensorConfig(H=VLS128["H"], W=VLS128["W"], fov_up_deg=VLS128["fov_up"], fov_down_deg=VLS128["fov_down"],
                           means=np.zeros(5, np.float32), stds=np.ones(5, np.float32))
    means, stds = measure_norm(GooseLidar(args.train_split, raw_cfg, lmap, limit=args.limit))
    cfg = SensorConfig(H=raw_cfg.H, W=raw_cfg.W, fov_up_deg=raw_cfg.fov_up_deg, fov_down_deg=raw_cfg.fov_down_deg,
                       means=means.astype(np.float32), stds=stds.astype(np.float32))
    args.out.mkdir(parents=True, exist_ok=True)
    arch = yaml.safe_load(open(RUN_DIR / "arch_cfg.yaml"))
    arch["dataset"]["sensor"].update({"name": "VLS128", "fov_up": cfg.fov_up_deg, "fov_down": cfg.fov_down_deg,
                                      "img_prop": {"width": cfg.W, "height": cfg.H},
                                      "img_means": [float(v) for v in means], "img_stds": [float(v) for v in stds]})
    yaml.safe_dump(arch, open(args.out / "arch_cfg.yaml", "w"), sort_keys=False)
    shutil.copy(RUN_DIR / "data_cfg.yaml", args.out / "data_cfg.yaml")

    train = GooseLidar(args.train_split, cfg, lmap, augment=True, limit=args.limit)
    val = GooseLidar(args.val_split, cfg, lmap, limit=args.limit)
    # class weights 1 / (frequency + 0.001), as SalsaNext's epsilon_w
    counts = np.zeros(n_classes)
    for i in np.linspace(0, len(train) - 1, min(300, len(train))).astype(int):
        counts += np.bincount(train[i][5].numpy(), minlength=n_classes)
    freq = counts / counts.sum()
    w = torch.tensor(1.0 / (freq + 0.001), dtype=torch.float32, device=device)
    w[0] = 0.0
    loader = DataLoader(train, batch_size=args.batch, shuffle=True, drop_last=True, collate_fn=pixel_labels,
                        num_workers=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")), persistent_workers=True,
                        worker_init_fn=seed_worker)

    model = SalsaNext(n_classes).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    steps = args.epochs * len(loader)
    warm = len(loader)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    log = open(args.out / "train_log.jsonl", "a")
    print(f"train {len(train)} scans, val {len(val)}; {len(loader)} steps/epoch; means {np.round(means, 3)}")
    best = -1.0
    for epoch in range(args.epochs):
        t0, total = time.time(), 0.0
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                probs = model(x)  # SalsaNext ends in a softmax
            probs = probs.float()
            loss = Fn.nll_loss(torch.log(probs.clamp(min=1e-8)), y, weight=w, ignore_index=0) + lovasz_softmax(probs, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total += float(loss)
        row = {"epoch": epoch, "loss": round(total / len(loader), 4), "lr": opt.param_groups[0]["lr"],
               "sec": round(time.time() - t0)}
        if (epoch + 1) % args.val_every == 0 or epoch == args.epochs - 1:
            miou, _ = evaluate(model, val, n_classes, device, every=4)
            row["val_miou_every4"] = round(100 * miou, 2)
            if miou > best:
                best = miou
                torch.save({"state_dict": model.state_dict(), "epoch": epoch, "val_miou": miou},
                           args.out / "SalsaNext_valid_best")
        log.write(json.dumps(row) + "\n")
        log.flush()
        print(json.dumps(row), flush=True)
    print(f"best val mIoU (every 4th scan) {100 * best:.2f} -> {args.out}")


if __name__ == "__main__":
    main()
