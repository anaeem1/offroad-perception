"""BEV traversability grid in base_link (contract: perception_msgs/TraversabilityGrid).

Per cell: dominant class (confidence-weighted vote), semantic cost from config/class_costs.yaml, raised to
lethal by a height-above-ground check (generic obstacles the lidar model cannot name, ADR-0003), confidence,
and source flags. Cells without points stay UNKNOWN (255): absence of evidence is never free space.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from scipy import ndimage

from pipeline.fusion import ONTO_INDEX, ONTOLOGY_IDS

UNKNOWN, LETHAL = 255, 254
COSTS = Path(__file__).resolve().parents[1] / "config/class_costs.yaml"


def load_costs(path=COSTS):
    classes = yaml.safe_load(open(path))["classes"]
    cost = np.full(len(ONTOLOGY_IDS), UNKNOWN, np.int16)
    obstacle = np.zeros(len(ONTOLOGY_IDS), bool)
    for cid, c in classes.items():
        cost[ONTO_INDEX[cid]] = -1 if c["cost"] is None else c["cost"]  # -1: contributes nothing (sky)
        obstacle[ONTO_INDEX[cid]] = c["obstacle"]
    return cost, obstacle


@dataclass
class GridParams:
    extent_m: float = 40.0          # square, centred on base_link
    resolution_m: float = 0.2
    z_min: float = -3.0             # base_link z band kept (drops spurious far-below / sky returns)
    z_max: float = 3.0
    ground_window_m: float = 1.0    # local ground = lowest point within this window
    obstacle_height_m: float = 0.3  # step above local ground that makes a cell lethal
    clearance_m: float = 2.5        # points higher than this above ground (canopy) are ignored
    geometric_exempt: tuple = (3, 19)  # grass, bush: their semantic cost already reflects height

    @property
    def n(self):
        return int(round(self.extent_m / self.resolution_m))


@dataclass
class Grid:
    cost: np.ndarray            # (n,n) uint8, row-major, row = y index, col = x index
    confidence: np.ndarray      # (n,n) uint8
    semantic_class: np.ndarray  # (n,n) uint8 ontology id
    source_flags: np.ndarray    # (n,n) uint8 bitmask: 1 lidar, 2 camera
    geometric: np.ndarray       # (n,n) bool, raised by the height check
    height: np.ndarray          # (n,n) float32 max height above local ground (nan if unobserved)
    params: GridParams

    @property
    def origin_xy(self):        # base_link position of cell (0,0)'s corner
        return (-self.params.extent_m / 2, -self.params.extent_m / 2)


def cell_index(xy, p: GridParams):
    ij = np.floor((xy + p.extent_m / 2) / p.resolution_m).astype(np.int64)
    inside = (ij >= 0).all(1) & (ij < p.n).all(1)
    return ij, inside


def build_grid(xyz_base, probs_onto, from_camera, p: GridParams, cost_lut, sky_index=ONTO_INDEX[7]):
    """xyz_base (N,3) valid points in base_link; probs_onto (N,20) fused distributions;
    from_camera (N,) bool. Returns a Grid."""
    n = p.n
    ij, inside = cell_index(xyz_base[:, :2], p)
    keep = inside & (xyz_base[:, 2] > p.z_min) & (xyz_base[:, 2] < p.z_max)
    cls = probs_onto.argmax(1)
    keep &= cls != sky_index
    ij, z, probs, cls, cam = ij[keep], xyz_base[keep, 2], probs_onto[keep], cls[keep], from_camera[keep]
    flat = ij[:, 1] * n + ij[:, 0]

    zmin = np.full(n * n, np.inf)
    np.minimum.at(zmin, flat, z)
    win = max(1, int(round(p.ground_window_m / p.resolution_m)))
    ground = ndimage.minimum_filter(zmin.reshape(n, n), size=win, mode="nearest").ravel()
    h = z - ground[flat]
    body = h < p.clearance_m  # ignore canopy and other overhangs
    hmax = np.full(n * n, -np.inf)
    np.maximum.at(hmax, flat[body], h[body])

    votes = np.zeros((n * n, len(ONTOLOGY_IDS)), np.float32)
    np.add.at(votes, (flat, cls), probs[np.arange(len(cls)), cls])
    observed = votes.sum(1) > 0
    dom = votes.argmax(1)
    share = np.where(observed, votes[np.arange(n * n), dom] / np.maximum(votes.sum(1), 1e-9), 0)
    meanp = np.zeros(n * n)
    cnt = np.zeros(n * n)
    np.add.at(meanp, flat, probs.max(1))
    np.add.at(cnt, flat, 1)
    meanp = np.where(cnt > 0, meanp / np.maximum(cnt, 1), 0)

    cost = np.full(n * n, UNKNOWN, np.int16)
    sem = cost_lut[dom]
    cost[observed] = np.where(sem[observed] < 0, UNKNOWN, sem[observed])
    exempt = np.isin(ONTOLOGY_IDS[dom], p.geometric_exempt)
    geometric = observed & (hmax > p.obstacle_height_m) & ~exempt
    cost[geometric] = LETHAL
    flags = np.zeros(n * n, np.uint8)
    flags[observed] = 1
    cam_any = np.zeros(n * n, bool)
    np.logical_or.at(cam_any, flat, cam)
    flags[cam_any] |= 2
    sem_class = np.where(observed, ONTOLOGY_IDS[dom], 0)
    height = np.where(np.isfinite(hmax), hmax, np.nan)
    shape = (n, n)
    return Grid(cost.astype(np.uint8).reshape(shape), np.round(255 * share * meanp).astype(np.uint8).reshape(shape),
                sem_class.astype(np.uint8).reshape(shape), flags.reshape(shape), geometric.reshape(shape),
                height.astype(np.float32).reshape(shape), p)
