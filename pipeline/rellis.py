"""RELLIS-3D access shared by the replay pipeline: paths, bag message decoding, dataset frame indexes."""
import os
import re
from pathlib import Path

import numpy as np

DATA_ROOT = Path(os.environ.get("OFFROAD_DATA_ROOT", "/usace_share/Share_Abdullah/offroad_perception"))
SEQ_ROOT = DATA_ROOT / "rellis3d/Rellis-3D"
BAGS = {"00004": DATA_ROOT / "bags/ros2/20200215_trail_1", "00001": DATA_ROOT / "bags/ros2/20200213_trail_1"}
ALIGN_DIR = DATA_ROOT / "rellis3d/alignment"

LIDAR_TOPIC = "/os1_cloud_node/points"
CAMERA_TOPIC = "/pylon_camera_node/image_raw"
INFO_TOPIC = "/pylon_camera_node/camera_info"
ODOM_TOPIC = "/odometry/filtered"


def stamp_of(msg):
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


def decode_cloud(msg):
    """Ouster PointCloud2 (64x2048, point_step 48) -> dict of per-point arrays in sensor order.
    remission = intensity / 65535, which is exactly the 4th channel of the SemanticKITTI .bin files the
    lidar model was trained on (verified on sequence 00001)."""
    buf = np.frombuffer(msg.data, np.uint8).reshape(-1, msg.point_step)
    return {
        "xyz": buf[:, 0:12].copy().view(np.float32).reshape(-1, 3),
        "remission": buf[:, 16:20].copy().view(np.float32).ravel() / np.float32(65535.0),
        "t_ns": buf[:, 20:24].copy().view(np.uint32).ravel(),
        "range_mm": buf[:, 32:36].copy().view(np.uint32).ravel(),
        "height": msg.height, "width": msg.width,
    }


def decode_bayer(msg):
    """bayer_rggb8 image -> (H, W) uint8 raw mosaic."""
    assert msg.encoding == "bayer_rggb8", msg.encoding
    return np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)[:, :msg.width]


def image_frames(seq):
    """{frame_index: (stamp_s, jpg_path)} from dataset filenames like frame000123-1581623790_349.jpg."""
    out = {}
    for p in (SEQ_ROOT / seq / "pylon_camera_node").glob("frame*.jpg"):
        m = re.match(r"frame(\d+)-(\d+)_(\d+)\.jpg", p.name)
        out[int(m.group(1))] = (int(m.group(2)) + int(m.group(3)) / 1000.0, p)
    return out


def scan_paths(seq, idx):
    d = SEQ_ROOT / seq
    return d / "os1_cloud_node_kitti_bin" / f"{idx:06d}.bin", d / "os1_cloud_node_semantickitti_label_id" / f"{idx:06d}.label"


def load_scan_labels(seq, idx):
    """Ground-truth ontology ids per point for labelled scan idx (same point order as the bag cloud)."""
    return (np.fromfile(scan_paths(seq, idx)[1], np.uint32) & 0xFFFF).astype(np.int32)
