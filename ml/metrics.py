"""Confusion-matrix IoU shared by the lidar and camera evaluations."""
import torch


class ConfusionMatrix:
    """Full [gt, pred] matrix. Two IoU conventions over the non-ignored classes:
    - "upstream": predictions of an ignored class are dropped (lidar-bonnetal / RELLIS-3D benchmark
      ioueval), so the result is comparable to published numbers.
    - "strict": predicting an ignored class (e.g. void) on a labeled point is a miss for that class.
      This is what a consumer of the output actually experiences."""

    def __init__(self, n_classes, ignore=(0,), device="cuda"):
        self.n = n_classes
        self.ignore = list(ignore)
        self.mat = torch.zeros(n_classes, n_classes, dtype=torch.int64, device=device)

    @torch.no_grad()
    def update(self, pred, gt):
        """pred, gt: integer tensors of identical shape, values in [0, n_classes)."""
        idx = gt.reshape(-1).long() * self.n + pred.reshape(-1).long()
        self.mat += torch.bincount(idx, minlength=self.n * self.n).reshape(self.n, self.n)

    def iou(self, convention="upstream"):
        m = self.mat.double().clone()
        m[self.ignore, :] = 0
        if convention == "upstream":
            m[:, self.ignore] = 0
        tp = m.diag()
        iou = tp / (m.sum(0) + m.sum(1) - tp).clamp(min=1)
        return {c: iou[c].item() for c in range(self.n) if c not in self.ignore}

    def support(self):
        """Number of ground-truth samples per scored class."""
        return {c: int(self.mat[c].sum().item()) for c in range(self.n) if c not in self.ignore}
