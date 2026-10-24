"""Run the exported ONNX models (the same files the Jetson builds engines from) with ONNX Runtime."""
import os

import cv2
import numpy as np
import yaml

from ml.camera.train_segformer import class_table
from ml.lidar.range_projection import SensorConfig, project
from pipeline.rellis import DATA_ROOT

SALSANEXT_RUN = DATA_ROOT / "weights/salsanext_rellis/2020-10-14-17:37rellis"
ONNX_DIR = DATA_ROOT / "onnx"


def _session(path, device):
    if device == "cuda":
        import torch  # noqa: F401  (not enough on its own: env/activate.sh puts cuDNN 9 on LD_LIBRARY_PATH)
    import onnxruntime as ort
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = int(os.environ.get("SLURM_CPUS_PER_TASK", "0"))  # 0 = ORT default
    return ort.InferenceSession(str(path), sess_options=opts, providers=providers)


class LidarSeg:
    """SalsaNext. Returns per-point class probabilities over its 15 training classes."""

    def __init__(self, device="cuda", onnx=ONNX_DIR / "salsanext_rellis_op17.onnx"):
        self.sess = _session(onnx, device)
        self.cfg = SensorConfig.from_arch_cfg(SALSANEXT_RUN / "arch_cfg.yaml")
        data = yaml.safe_load(open(SALSANEXT_RUN / "data_cfg.yaml"))
        self.train_to_onto = np.array([data["learning_map_inv"][i] for i in range(len(data["learning_map_inv"]))])
        # Ontology ids folded into training class 0 ("void" bucket): void, dirt, sky, object, asphalt, building.
        self.bucket = sorted(k for k, v in data["learning_map"].items() if v == 0)

    def __call__(self, xyz, remission):
        proj, _, px, py, _ = project(xyz, remission, self.cfg)
        probs, _ = self.sess.run(None, {"input": proj[None]})
        return probs[0][:, py, px].T  # (N, 15)


class CameraSeg:
    """SegFormer-B0. Returns class probabilities at 1/4 of the network input resolution."""

    def __init__(self, device="cuda", onnx=ONNX_DIR / "segformer_b0_640x1024_op17.onnx", size_hw=(640, 1024)):
        self.sess = _session(onnx, device)
        self.h, self.w = size_hw
        self.class_ids = np.array(class_table()[0])  # ontology id of each output channel (void excluded)

    def __call__(self, rgb):
        x = cv2.resize(rgb, (self.w, self.h), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1)[None]
        probs, _ = self.sess.run(None, {"input": np.ascontiguousarray(x, np.float32)})
        return probs[0]  # (19, h/4, w/4)
