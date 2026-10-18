"""Obstacle candidates: connected components of obstacle points on a BEV raster (ports directly to C++).

Obstacle points are points whose fused class is a tracked class in config/class_costs.yaml, plus void
points in cells the height check marked lethal: generic objects the lidar model cannot name (ADR-0003).
"""
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from pipeline.bev import GridParams, cell_index
from pipeline.fusion import ONTO_INDEX, ONTOLOGY_IDS


@dataclass
class Detection:
    xyz: np.ndarray          # centroid (base_link)
    dims: np.ndarray         # axis-aligned extent x, y, z
    semantic_class: int      # ontology id
    class_confidence: float
    n_points: int


def obstacle_mask(cls_idx, ij, geometric_cells, obstacle_lut, void_index=ONTO_INDEX[0]):
    sem = obstacle_lut[cls_idx]
    n = geometric_cells.shape[0]
    inside = (ij >= 0).all(1) & (ij < n).all(1)
    geo = np.zeros(len(cls_idx), bool)
    geo[inside] = geometric_cells[ij[inside, 1], ij[inside, 0]]
    return sem | ((cls_idx == void_index) & geo)


def cluster(xyz_base, probs_onto, geometric_cells, p: GridParams, obstacle_lut, min_points=5, bridge_cells=1):
    cls = probs_onto.argmax(1)
    ij, inside = cell_index(xyz_base[:, :2], p)
    m = inside & obstacle_mask(cls, ij, geometric_cells, obstacle_lut)
    if not m.any():
        return []
    pts, cls_m, conf, ij = xyz_base[m], cls[m], probs_onto[m].max(1), ij[m]
    occ = np.zeros((p.n, p.n), bool)
    occ[ij[:, 1], ij[:, 0]] = True
    # Bridge 1-cell gaps (thin objects seen as broken rings), then label the original cells.
    grown = ndimage.binary_dilation(occ, iterations=bridge_cells) if bridge_cells else occ
    labels, n_lab = ndimage.label(grown, structure=np.ones((3, 3)))
    pid = labels[ij[:, 1], ij[:, 0]]
    out = []
    for k in range(1, n_lab + 1):
        sel = pid == k
        if sel.sum() < min_points:
            continue
        c = np.bincount(cls_m[sel], minlength=len(ONTOLOGY_IDS))
        major = int(c.argmax())
        lo, hi = pts[sel].min(0), pts[sel].max(0)
        out.append(Detection(xyz=pts[sel].mean(0), dims=hi - lo, semantic_class=int(ONTOLOGY_IDS[major]),
                             class_confidence=float(conf[sel][cls_m[sel] == major].mean() * c[major] / sel.sum()),
                             n_points=int(sel.sum())))
    return out
