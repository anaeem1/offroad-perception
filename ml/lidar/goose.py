"""GOOSE lidar data (goose-dataset.de, CC BY-SA 4.0) in the RELLIS-3D class set, for the monitor study's second
benchmark: GOOSE label -> RELLIS ontology id (config/goose_to_rellis.yaml) -> RELLIS SalsaNext training id
(the released run's data_cfg learning_map). The trained network then drops into the existing pipeline.

Layout (scripts/download_goose.sh): <data root>/goose/3d/**/lidar/<split>/<scenario>/*_vls128.bin (float32 x, y,
z, remission) and .../labels/<split>/<scenario>/*_goose.label (uint32, semantic id in the lower 16 bits).
"""
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

from ml.lidar.range_projection import SensorConfig, project
from pipeline.rellis import DATA_ROOT

REPO = Path(__file__).resolve().parents[2]
GOOSE_3D = DATA_ROOT / "goose/3d"
# Velodyne VLS-128 (MuCAR-3 roof): 128 rings over +15 to -25 deg; 2048 columns as for the Ouster.
VLS128 = {"H": 128, "W": 2048, "fov_up": 15.0, "fov_down": -25.0}


def _official(split):
    out = []
    for b in sorted(GOOSE_3D.glob(f"**/lidar/{split}/**/*_vls128.bin")):
        lab = Path(str(b).replace(f"/lidar/{split}/", f"/labels/{split}/")).with_name(
            b.name.replace("_vls128.bin", "_goose.label"))
        out.append((b, lab))
    return out


def held_out(fold="a"):
    """Recordings held out in one fold of config/goose_split.yaml (a = the heldout list, b and c = folds)."""
    cfg = yaml.safe_load(open(REPO / "config/goose_split.yaml"))
    return set(cfg["heldout"] if fold == "a" else cfg["folds"][fold])


def scans(split):
    """Sorted (scan .bin, label) pairs. Official splits (train, val, test) or the recording-level split of
    config/goose_split.yaml: fit_train, fit_val (training recordings), heldout (held-out recordings, both
    official splits), for fold a; append :b or :c for the other folds (fit_train:b, heldout:b)."""
    if split in ("train", "val", "test"):
        return _official(split)
    kind, _, fold = split.partition(":")
    held = held_out(fold or "a")
    rec = lambda item: item[0].parent.name
    if kind == "fit_train":
        return [it for it in _official("train") if rec(it) not in held]
    if kind == "fit_val":
        return [it for it in _official("val") if rec(it) not in held]
    if kind == "heldout":
        return sorted((it for it in _official("train") + _official("val") if rec(it) in held), key=lambda it: it[0])
    raise ValueError(f"unknown split {split}")


def label_lut(learning_map):
    """GOOSE label key -> RELLIS SalsaNext training id (0 = void, ignored)."""
    goose_to_onto = yaml.safe_load(open(REPO / "config/goose_to_rellis.yaml"))
    lut = np.zeros(1 << 16, np.int64)
    for g, onto in goose_to_onto.items():
        lut[g] = learning_map.get(onto, 0)
    return lut


class GooseLidar(Dataset):
    def __init__(self, split, cfg: SensorConfig, learning_map, augment=False, limit=None):
        self.items = scans(split)[:limit]
        self.cfg, self.augment = cfg, augment
        self.lut = label_lut(learning_map)
        self.rng = np.random.default_rng()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        scan_p, label_p = self.items[i]
        scan = np.fromfile(scan_p, np.float32).reshape(-1, 4)
        raw = np.fromfile(label_p, np.uint32) & 0xFFFF
        assert len(raw) == len(scan), f"{scan_p.name}: {len(scan)} points vs {len(raw)} labels"
        xyz = scan[:, :3].copy()
        if self.augment:  # yaw rotation and a left-right mirror, as in SalsaNext's own training
            a = self.rng.uniform(-np.pi, np.pi)
            c, s = np.cos(a), np.sin(a)
            xyz[:, :2] = xyz[:, :2] @ np.array([[c, -s], [s, c]], np.float32).T
            if self.rng.random() < 0.5:
                xyz[:, 1] = -xyz[:, 1]
        proj, proj_range, px, py, depth = project(xyz, scan[:, 3].copy(), self.cfg)
        return (torch.from_numpy(proj), torch.from_numpy(proj_range), torch.from_numpy(px).long(),
                torch.from_numpy(py).long(), torch.from_numpy(depth), torch.from_numpy(self.lut[raw]))
