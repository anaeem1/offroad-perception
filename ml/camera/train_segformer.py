"""Fine-tune SegFormer on RELLIS-3D camera images, keep the best val checkpoint, report test IoU as JSON.

Void (ontology id 0) is ignored in the loss and in scoring, so the model never predicts it; the runtime
expresses "don't know" through confidence instead. Classes are indexed in ontology-id order.

  sbatch scripts/slurm/train_segformer.sbatch   # 1x H100; compute nodes are offline, weights pre-fetched
"""
import argparse
import datetime
import json
import math
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset
from transformers import SegformerForSemanticSegmentation

from ml.metrics import ConfusionMatrix

cv2.setNumThreads(1)  # parallelism comes from DataLoader workers
DATA_ROOT = Path(os.environ.get("OFFROAD_DATA_ROOT", "/usace_share/Share_Abdullah/offroad_perception"))
ONTOLOGY = DATA_ROOT / "rellis3d/meta/Rellis_3D_ontology/ontology.yaml"
IGNORE = 255
MEAN = np.array([0.485, 0.456, 0.406], np.float32) * 255  # ImageNet stats, as SegFormer was pretrained
STD = np.array([0.229, 0.224, 0.225], np.float32) * 255


def class_table():
    """Training classes: every ontology id except void, in id order -> (ids, names)."""
    names = yaml.safe_load(open(ONTOLOGY))[0]
    ids = sorted(i for i in names if i != 0)
    return ids, [names[i] for i in ids]


class RellisImages(Dataset):
    def __init__(self, split, ids, size_hw, crop=None, limit=None):
        lst = DATA_ROOT / "rellis3d/meta" / f"{split}.lst"
        self.items = [line.split() for line in open(lst) if line.strip()][:limit]
        self.base = DATA_ROOT / "rellis3d/Rellis-3D"
        self.lut = np.full(256, IGNORE, np.uint8)
        for k, i in enumerate(ids):
            self.lut[i] = k
        self.h, self.w = size_hw
        self.crop = crop  # None = eval: whole image at size_hw, labels kept at full resolution

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        img_rel, lab_rel = self.items[i]
        img = cv2.cvtColor(cv2.imread(str(self.base / img_rel), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        lab = self.lut[cv2.imread(str(self.base / lab_rel), cv2.IMREAD_GRAYSCALE)]
        if self.crop:
            img, lab = self._augment(img, lab)
        else:
            img = cv2.resize(img, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        x = ((img.astype(np.float32) - MEAN) / STD).transpose(2, 0, 1)
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(lab.astype(np.int64))

    def _augment(self, img, lab):
        s = np.random.uniform(0.75, 1.5)
        h, w = int(self.h * s), int(self.w * s)
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        lab = cv2.resize(lab, (w, h), interpolation=cv2.INTER_NEAREST)
        ch, cw = self.crop
        ph, pw = max(ch - h, 0), max(cw - w, 0)
        if ph or pw:
            img = cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=MEAN.tolist())
            lab = cv2.copyMakeBorder(lab, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=IGNORE)
        y0 = np.random.randint(0, img.shape[0] - ch + 1)
        x0 = np.random.randint(0, img.shape[1] - cw + 1)
        img, lab = img[y0:y0 + ch, x0:x0 + cw], lab[y0:y0 + ch, x0:x0 + cw]
        if np.random.rand() < 0.5:
            img, lab = img[:, ::-1], lab[:, ::-1]
        img = np.clip(img.astype(np.float32) * np.random.uniform(0.8, 1.2) + np.random.uniform(-20, 20), 0, 255)
        return img, lab


@torch.no_grad()
def evaluate(model, loader, n_classes, device):
    model.eval()
    cm = ConfusionMatrix(n_classes + 1, ignore=(n_classes,), device=device)  # slot n_classes = IGNORE
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.bfloat16):
            logits = model(pixel_values=x).logits
        pred = F.interpolate(logits.float(), size=y.shape[-2:], mode="bilinear", align_corners=False).argmax(1)
        cm.update(pred, torch.where(y == IGNORE, n_classes, y))
    model.train()
    return cm


def summarize(cm, names):
    iou = cm.iou("strict")  # the model cannot predict the ignore slot, so both conventions agree
    support = cm.support()
    scored = {c: v for c, v in iou.items() if support[c] > 0}
    return {"per_class_iou": {names[c]: round(100 * v, 2) for c, v in scored.items()},
            "miou": round(100 * float(np.mean(list(scored.values()))), 2),
            "absent_classes": [names[c] for c in iou if support[c] == 0],
            "support_pixels": {names[c]: support[c] for c in iou}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="nvidia/segformer-b0-finetuned-cityscapes-1024-1024")
    ap.add_argument("--size", type=int, nargs=2, default=[640, 1024], metavar=("H", "W"),
                    help="network input size (deployment resolution); RELLIS images are 1200x1920")
    ap.add_argument("--crop", type=int, nargs=2, default=[640, 640])
    ap.add_argument("--iters", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=6e-5)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="cap images per split (smoke test)")
    ap.add_argument("--device", default="cuda", help="cpu only for smoke tests")
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="test metrics JSON (committed)")
    args = ap.parse_args()
    torch.manual_seed(0); np.random.seed(0)
    device = torch.device(args.device)

    ids, names = class_table()
    n = len(ids)
    model = SegformerForSemanticSegmentation.from_pretrained(
        args.init, num_labels=n, id2label=dict(enumerate(names)), label2id={v: k for k, v in enumerate(names)},
        ignore_mismatched_sizes=True, semantic_loss_ignore_index=IGNORE).to(device)

    train = DataLoader(RellisImages("train", ids, args.size, crop=args.crop, limit=args.limit),
                       batch_size=args.batch, shuffle=True, drop_last=True, num_workers=args.workers,
                       pin_memory=True, persistent_workers=args.workers > 0)
    val = DataLoader(RellisImages("val", ids, args.size, limit=args.limit), batch_size=4,
                     num_workers=args.workers, pin_memory=True)
    test = DataLoader(RellisImages("test", ids, args.size, limit=args.limit), batch_size=4,
                      num_workers=args.workers, pin_memory=True)

    head = [p for k, p in model.named_parameters() if k.startswith("decode_head")]
    body = [p for k, p in model.named_parameters() if not k.startswith("decode_head")]
    opt = torch.optim.AdamW([{"params": body, "lr": args.lr}, {"params": head, "lr": 10 * args.lr}],
                            weight_decay=0.01)
    warmup = min(300, args.iters // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda it: (it + 1) / warmup if it < warmup else (1 - (it - warmup) / (args.iters - warmup)) ** 1.0)

    args.run_dir.mkdir(parents=True, exist_ok=True)
    log = open(args.run_dir / "train_log.jsonl", "a")
    best, it, t0 = -1.0, 0, time.time()
    while it < args.iters:
        for x, y in train:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.bfloat16):
                loss = model(pixel_values=x, labels=y).loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step(); sched.step(); it += 1
            if it % 50 == 0:
                rec = {"it": it, "loss": round(loss.item(), 4), "lr": sched.get_last_lr()[0],
                       "elapsed_s": round(time.time() - t0)}
                print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
            if it % args.eval_every == 0 or it == args.iters:
                miou = summarize(evaluate(model, val, n, device), names)["miou"]
                rec = {"it": it, "val_miou": miou}
                print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
                if miou > best:
                    best = miou
                    model.save_pretrained(args.run_dir / "best")
            if it >= args.iters:
                break

    model = SegformerForSemanticSegmentation.from_pretrained(args.run_dir / "best").to(device)
    result = {
        "what": f"SegFormer fine-tuned on RELLIS-3D images from {args.init}",
        "checkpoint": str(args.run_dir / "best"), "date": datetime.date.today().isoformat(),
        "input_size_hw": args.size, "scored_at": "full label resolution 1200x1920 (logits upsampled)",
        "train": {"iters": args.iters, "batch": args.batch, "crop": args.crop, "lr": args.lr,
                  "wall_time_min": round((time.time() - t0) / 60, 1), "device": torch.cuda.get_device_name() if device.type == "cuda" else "cpu"},
        "best_val_miou": best,
        "test": summarize(evaluate(model, test, n, device), names),
        "published_reference": {"HRNet+OCR": 48.83, "GSCNN": 50.13, "note": "RELLIS-3D README, 18 classes"},
        "env": {"torch": torch.__version__},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print("test mIoU", result["test"]["miou"], "->", args.out)


if __name__ == "__main__":
    main()
