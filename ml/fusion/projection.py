"""Lidar -> camera projection for RELLIS-3D. Used by the calibration overlay check and by late fusion."""
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

DATA_ROOT = Path(os.environ.get("OFFROAD_DATA_ROOT", "/usace_share/Share_Abdullah/offroad_perception"))
# Plumb-bob distortion from the RELLIS-3D lidar2img example (utils/lidar2img.ipynb). The dataset's
# camera_info.txt has no distortion; cross-check against /pylon_camera_node/camera_info in the bag.
RELLIS_DIST = np.array([-0.134313, -0.025905, 0.002181, 0.00084, 0.0])


@dataclass(frozen=True)
class Camera:
    K: np.ndarray  # 3x3
    D: np.ndarray  # (5,) plumb-bob
    width: int = 1920
    height: int = 1200


def load_rellis_calib(seq: str, root: Path = DATA_ROOT):
    """-> (Camera, T_cam_lidar 4x4). Extrinsics differ between sequences, so always load per sequence."""
    fx, fy, cx, cy = np.loadtxt(root / f"rellis3d/calib/Rellis-3D/{seq}/camera_info.txt")
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    # transforms.yaml holds the camera pose in the lidar frame (T_lidar_cam); invert to map points.
    tf = yaml.safe_load(open(root / f"rellis3d/calib/Rellis_3D/{seq}/transforms.yaml"))
    tf = tf["os1_cloud_node-pylon_camera_node"]
    T_lidar_cam = np.eye(4)
    T_lidar_cam[:3, :3] = Rotation.from_quat([tf["q"][k] for k in "xyzw"]).as_matrix()
    T_lidar_cam[:3, 3] = [tf["t"][k] for k in "xyz"]
    return Camera(K=K, D=RELLIS_DIST), np.linalg.inv(T_lidar_cam)


def project_points(xyz, cam: Camera, T_cam_lidar, min_depth=0.5, fov_margin=0.1):
    """xyz (N,3) lidar frame -> uv (M,2) pixel coords, idx (M,) indices into xyz, depth (M,) metres.
    Points behind the camera or far outside its FOV are culled *before* distortion, because the
    polynomial model folds such points back into the image."""
    pc = xyz @ T_cam_lidar[:3, :3].T + T_cam_lidar[:3, 3]
    z = pc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        xn, yn = pc[:, 0] / z, pc[:, 1] / z
    fx, fy, cx, cy = cam.K[0, 0], cam.K[1, 1], cam.K[0, 2], cam.K[1, 2]
    keep = ((z > min_depth)
            & (xn > -cx / fx - fov_margin) & (xn < (cam.width - cx) / fx + fov_margin)
            & (yn > -cy / fy - fov_margin) & (yn < (cam.height - cy) / fy + fov_margin))
    idx = np.flatnonzero(keep)
    if len(idx) == 0:  # cv2.projectPoints rejects an empty point set
        return np.zeros((0, 2)), idx, np.zeros(0)
    uv, _ = cv2.projectPoints(pc[idx].reshape(-1, 1, 3), np.zeros(3), np.zeros(3), cam.K, cam.D)
    uv = uv.reshape(-1, 2)
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < cam.width) & (uv[:, 1] >= 0) & (uv[:, 1] < cam.height)
    return uv[inside], idx[inside], z[idx[inside]]


def perturb(T_cam_lidar, yaw_deg=0.0, pitch_deg=0.0):
    """Rotate the extrinsic about the camera's y (yaw) and x (pitch) axes, for sensitivity checks."""
    R = np.eye(4)
    R[:3, :3] = Rotation.from_euler("yx", [yaw_deg, pitch_deg], degrees=True).as_matrix()
    return R @ T_cam_lidar
