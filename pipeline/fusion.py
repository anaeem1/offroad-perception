"""Late fusion of lidar and camera class probabilities in RELLIS-3D ontology space (ADR-0002).

Class spaces differ: SalsaNext folds {void, dirt, sky, object, asphalt, building} into one training class,
SegFormer has 19 classes and no void. Both are mapped onto the 20 ontology ids. The lidar's bucket mass is
spread evenly over the bucket ids (it cannot tell them apart), so inside the camera wedge the camera
decides among them. Outside the wedge, points keep the lidar result.
"""
import numpy as np

from ml.fusion.projection import project_points
from pipeline.geometry import deskew

ONTOLOGY_IDS = np.array([0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 17, 18, 19, 23, 27, 31, 33, 34])
ONTO_INDEX = {c: i for i, c in enumerate(ONTOLOGY_IDS)}


def lidar_to_ontology(p_lidar, train_to_onto, bucket):
    """(N,15) lidar probs -> (N,20) over ONTOLOGY_IDS."""
    out = np.zeros((len(p_lidar), len(ONTOLOGY_IDS)), np.float32)
    for j in range(1, p_lidar.shape[1]):
        out[:, ONTO_INDEX[train_to_onto[j]]] = p_lidar[:, j]
    for c in bucket:
        out[:, ONTO_INDEX[c]] = p_lidar[:, 0] / len(bucket)
    return out


def camera_to_ontology(p_cam_points, class_ids):
    """(N,19) camera probs sampled at points -> (N,20) over ONTOLOGY_IDS (void stays 0)."""
    out = np.zeros((len(p_cam_points), len(ONTOLOGY_IDS)), np.float32)
    out[:, [ONTO_INDEX[c] for c in class_ids]] = p_cam_points
    return out


def lidar_labels(p_lidar, train_to_onto):
    """Lidar-only argmax in ontology ids (the bucket class maps to void, 0)."""
    return train_to_onto[p_lidar.argmax(1)]


def sample_camera(probs_cam, uv, img_wh=(1920, 1200)):
    """probs_cam (C,h,w) at 1/4 network resolution; uv (M,2) original-image pixels -> (M,C)."""
    C, h, w = probs_cam.shape
    u = np.clip((uv[:, 0] * w / img_wh[0]).astype(int), 0, w - 1)
    v = np.clip((uv[:, 1] * h / img_wh[1]).astype(int), 0, h - 1)
    return probs_cam[:, v, u].T


def fuse(p_l_onto, p_c_onto, alpha, mode="linear"):
    """Pool two (M,20) distributions; alpha = lidar weight."""
    if mode == "linear":
        return alpha * p_l_onto + (1 - alpha) * p_c_onto
    eps = 1e-6  # log-linear (product of experts)
    logp = alpha * np.log(p_l_onto + eps) + (1 - alpha) * np.log(p_c_onto + eps)
    p = np.exp(logp - logp.max(1, keepdims=True))
    return p / p.sum(1, keepdims=True)


def camera_points(xyz, point_t, valid, cam, T_cam_lidar, odom=None, T_base_lidar=None, t_cam=None):
    """Project lidar points into the camera image. With odom, each point is first moved from its own
    measurement time to the camera exposure time t_cam (motion compensation); without, raw geometry.
    Returns (uv, idx) for points that land in the image."""
    sel = np.flatnonzero(valid)
    pts = xyz[sel]
    if odom is not None:
        pts = deskew(pts, point_t[sel], odom, T_base_lidar, t_cam)
    uv, i, _ = project_points(pts, cam, T_cam_lidar)
    return uv, sel[i]
