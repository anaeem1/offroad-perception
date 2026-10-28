"""Sensor fault injection for the robustness study: degrade the replayed inputs the way a real fault would,
before any perception code sees them. Ground truth is always computed from the clean data, so lost coverage
shows up as UNKNOWN where the labels know the answer (hazards H-4, H-5, H-13; docs/hazard_analysis.md).

Spec strings for run_replay --fault (repeatable):
  lidar_drop=<p>      lose a random fraction p of returns (rain, dust, a dirty window)
  beams=<n>           keep n of the 64 rings, evenly spaced (a cheaper sensor than the one trained on)
  lidar_sector=<deg>  lose every return in a sector this wide, centred on the vehicle's forward direction
                      (mud or a splash on the front of the window: spatially correlated loss)
  range_noise=<m>     Gaussian range noise with this sigma, along each ray
  cam_offset=<s>      camera clock offset (lost time sync: frames carry wrong stamps)
  cam_yaw=<deg>       camera-lidar extrinsic drift about the camera's vertical axis
  cam_pitch=<deg>     ... about its horizontal axis
  cam_gain=<g>        image brightness gain (g < 1: underexposure, g > 1: overexposure, clipped)
"""
import numpy as np

from ml.fusion.projection import perturb

KINDS = {"lidar_drop", "beams", "lidar_sector", "range_noise", "cam_offset", "cam_yaw", "cam_pitch", "cam_gain"}
BEAMS = 64  # Ouster OS1-64; the cloud is column-major, point i is ring i % 64


def parse(specs):
    out = {}
    for s in specs or []:
        k, v = s.split("=")
        if k not in KINDS:
            raise ValueError(f"unknown fault {k}; one of {sorted(KINDS)}")
        out[k] = float(v)
    return out


def lidar(cloud, f, rng, forward_deg=0.0):
    """Decoded cloud (rellis.decode_cloud) -> faulted copy. Lost returns read as the sensor reports them:
    xyz = 0, range 0. forward_deg: azimuth of the vehicle's +x axis in the lidar frame."""
    if not ({"lidar_drop", "beams", "lidar_sector", "range_noise"} & f.keys()):
        return cloud
    c = dict(cloud, xyz=cloud["xyz"].copy(), remission=cloud["remission"].copy(), range_mm=cloud["range_mm"].copy())
    n = len(c["xyz"])
    lost = np.zeros(n, bool)
    if "lidar_drop" in f:
        lost |= rng.random(n) < f["lidar_drop"]
    if "beams" in f:
        keep = np.round(np.linspace(0, BEAMS - 1, int(f["beams"]))).astype(int)
        lost |= ~np.isin(np.arange(n) % BEAMS, keep)
    if "lidar_sector" in f:
        az = np.degrees(np.arctan2(c["xyz"][:, 1], c["xyz"][:, 0]))
        lost |= np.abs((az - forward_deg + 180.0) % 360.0 - 180.0) < f["lidar_sector"] / 2
    if "range_noise" in f:
        r = np.linalg.norm(c["xyz"], axis=1)
        ok = r > 0
        scale = 1.0 + rng.normal(0.0, f["range_noise"], n)[ok] / r[ok]
        c["xyz"][ok] *= scale[:, None].astype(np.float32)
        c["range_mm"][ok] = np.maximum(1, np.round(c["range_mm"][ok] * scale)).astype(np.uint32)
    c["xyz"][lost] = 0.0
    c["remission"][lost] = 0.0
    c["range_mm"][lost] = 0
    return c


def extrinsic(T_cam_lidar, f):
    return perturb(T_cam_lidar, yaw_deg=f.get("cam_yaw", 0.0), pitch_deg=f.get("cam_pitch", 0.0))


def camera_stamp(t, f):
    return t + f.get("cam_offset", 0.0)


def image(rgb, f):
    return rgb if "cam_gain" not in f else np.clip(rgb.astype(np.float32) * f["cam_gain"], 0, 255).astype(np.uint8)
