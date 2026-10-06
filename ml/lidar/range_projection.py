"""Spherical range-image projection, matching the RELLIS-3D SalsaNext training pipeline exactly.

Port of LaserScan.do_range_projection + the parser's normalization (lidar-bonnetal, MIT). The C++
runtime node must reproduce this function bit-for-bit, so it stays deliberately literal, quirks included.
"""
from dataclasses import dataclass

import numpy as np
import yaml


@dataclass(frozen=True)
class SensorConfig:
    H: int
    W: int
    fov_up_deg: float
    fov_down_deg: float
    means: np.ndarray  # (5,) range, x, y, z, remission
    stds: np.ndarray

    @classmethod
    def from_arch_cfg(cls, path):
        s = yaml.safe_load(open(path))["dataset"]["sensor"]
        return cls(H=s["img_prop"]["height"], W=s["img_prop"]["width"],
                   fov_up_deg=s["fov_up"], fov_down_deg=s["fov_down"],
                   means=np.asarray(s["img_means"], np.float32), stds=np.asarray(s["img_stds"], np.float32))


def project(points, remissions, cfg: SensorConfig):
    """points (N,3) float32, remissions (N,) float32 ->
    proj (5,H,W) normalized network input, proj_range (H,W), proj_x (N,), proj_y (N,), depth (N,)."""
    fov_up = cfg.fov_up_deg / 180.0 * np.pi
    fov_down = cfg.fov_down_deg / 180.0 * np.pi
    fov = abs(fov_down) + abs(fov_up)

    depth = np.linalg.norm(points, 2, axis=1)
    depth[depth == 0] = 1e-4
    yaw = -np.arctan2(points[:, 1], points[:, 0])
    pitch = np.arcsin(points[:, 2] / depth)

    proj_x = np.floor(0.5 * (yaw / np.pi + 1.0) * cfg.W)
    proj_x = np.maximum(0, np.minimum(cfg.W - 1, proj_x)).astype(np.int32)
    proj_y = np.floor((1.0 - (pitch + abs(fov_down)) / fov) * cfg.H)
    proj_y = np.maximum(0, np.minimum(cfg.H - 1, proj_y)).astype(np.int32)

    # Far-to-near write order, so the nearest point wins each pixel.
    order = np.argsort(depth)[::-1]
    proj_range = np.full((cfg.H, cfg.W), -1, np.float32)
    proj_xyz = np.full((cfg.H, cfg.W, 3), -1, np.float32)
    proj_rem = np.full((cfg.H, cfg.W), -1, np.float32)
    proj_idx = np.full((cfg.H, cfg.W), -1, np.int32)
    py, px = proj_y[order], proj_x[order]
    proj_range[py, px] = depth[order]
    proj_xyz[py, px] = points[order]
    proj_rem[py, px] = remissions[order]
    proj_idx[py, px] = np.arange(len(depth), dtype=np.int32)[order]
    # Upstream uses `> 0`, so the pixel holding point index 0 counts as empty. Kept for fidelity.
    proj_mask = (proj_idx > 0).astype(np.float32)

    proj = np.concatenate([proj_range[None], proj_xyz.transpose(2, 0, 1), proj_rem[None]])
    proj = (proj - cfg.means[:, None, None]) / cfg.stds[:, None, None]
    proj *= proj_mask[None]
    return proj.astype(np.float32), proj_range, proj_x, proj_y, depth.astype(np.float32)
