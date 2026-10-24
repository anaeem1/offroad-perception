"""Grid memory (ADR-0008): fill cells the current sweep cannot see with static evidence from earlier sweeps.

The lidar never sees the ground within ~3 m of the machine, so a single-sweep grid has a permanently
UNKNOWN near field, and a contract-following planner can never move (docs/measurements/grid_memory_*.json).
Memory is keyed by odom-frame cells, so it stays valid as the machine moves.

Per sweep: build the current grid -> fill(): unobserved cells take a remembered value no older than
max_age_s, with confidence decayed linearly by age -> update(): store the current sweep's observed cells,
except dynamic classes (person, vehicle) which only the current sweep and the tracker may report.
Cells filled from memory get source_flags = 4 (REMEMBERED; 1 = lidar, 2 = camera).
"""
from dataclasses import dataclass

import numpy as np

from pipeline.bev import Grid
from pipeline.geometry import apply

REMEMBERED = 4


@dataclass
class MemoryParams:
    max_age_s: float = 3.0
    dynamic_classes: tuple = (17, 8)   # person, vehicle: never remembered
    half_size_m: float = 250.0         # dense odom-frame store; covers a RELLIS-3D sequence


class GridMemory:
    def __init__(self, resolution_m: float, p: MemoryParams = MemoryParams()):
        self.r, self.p = resolution_m, p
        self.n = int(2 * p.half_size_m / resolution_m)
        self.cost = np.full((self.n, self.n), 255, np.uint8)
        self.conf = np.zeros((self.n, self.n), np.uint8)
        self.cls = np.zeros((self.n, self.n), np.uint8)
        self.flags = np.zeros((self.n, self.n), np.uint8)
        self.t_seen = np.full((self.n, self.n), -np.inf)

    def _keys(self, grid: Grid, T_odom_base):
        """Odom-frame memory cell of every output-grid cell centre -> (row, col) arrays."""
        g = grid.params
        c = (np.arange(g.n) + 0.5) * g.resolution_m - g.extent_m / 2
        X, Y = np.meshgrid(c, c)  # rows = y, cols = x, as in the grid arrays
        xy = apply(T_odom_base, np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], 1))[:, :2]
        k = np.floor(xy / self.r).astype(np.int64) + self.n // 2
        assert (k >= 0).all() and (k < self.n).all(), "memory store too small for this trajectory"
        return k[:, 1], k[:, 0]  # (row = y, col = x)

    def fill(self, grid: Grid, T_odom_base, t):
        rows, cols = self._keys(grid, T_odom_base)
        unseen = grid.source_flags.ravel() == 0
        age = t - self.t_seen[rows, cols]
        use = unseen & (age <= self.p.max_age_s)
        decay = np.clip(1.0 - age / self.p.max_age_s, 0.0, 1.0)
        for name, arr in (("cost", self.cost), ("semantic_class", self.cls)):
            getattr(grid, name).ravel()[use] = arr[rows[use], cols[use]]
        grid.confidence.ravel()[use] = np.round(self.conf[rows[use], cols[use]] * decay[use]).astype(np.uint8)
        grid.source_flags.ravel()[use] = REMEMBERED
        return int(use.sum())

    def update(self, grid: Grid, T_odom_base, t):
        rows, cols = self._keys(grid, T_odom_base)
        flags = grid.source_flags.ravel()
        store = ((flags & 3) > 0) & ~np.isin(grid.semantic_class.ravel(), self.p.dynamic_classes)
        r, c = rows[store], cols[store]
        self.cost[r, c] = grid.cost.ravel()[store]
        self.conf[r, c] = grid.confidence.ravel()[store]
        self.cls[r, c] = grid.semantic_class.ravel()[store]
        self.t_seen[r, c] = t
