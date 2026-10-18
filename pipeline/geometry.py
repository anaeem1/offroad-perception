"""Rigid-body helpers: TF chains, odometry interpolation, and per-point deskew.

Conventions: T_a_b maps points from frame b into frame a (x_a = T_a_b @ x_b). Times are float seconds.
"""
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def make_T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def inv(T):
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti


def apply(T, xyz):
    return xyz @ T[:3, :3].T + T[:3, 3]


def tf_graph(edges):
    """[(parent, child, t_xyz, q_xyzw), ...] -> {child: (parent, T_parent_child)}. Later entries override
    earlier ones, as in tf2."""
    g = {}
    for parent, child, t, q in edges:
        g[child] = (parent, make_T(Rotation.from_quat(q).as_matrix(), t))
    return g


def T_root_frame(g, frame, root="base_link"):
    T = np.eye(4)
    while frame != root:
        parent, T_pc = g[frame]
        T = T_pc @ T
        frame = parent
    return T


class PoseTrack:
    """Time-indexed SE(3) poses T_world_body (e.g. odom->base_link). Linear position, slerp rotation;
    clamps outside the recorded span (callers check `covers` when that matters)."""

    def __init__(self, stamps, positions, quats_xyzw):
        order = np.argsort(stamps)
        self.t = np.asarray(stamps, np.float64)[order]
        keep = np.concatenate([[True], np.diff(self.t) > 0])  # Slerp needs strictly increasing times
        self.t = self.t[keep]
        self.p = np.asarray(positions, np.float64)[order][keep]
        self.slerp = Slerp(self.t, Rotation.from_quat(np.asarray(quats_xyzw)[order][keep]))

    def covers(self, t):
        return self.t[0] <= np.min(t) and np.max(t) <= self.t[-1]

    def at(self, t):
        """t scalar or (N,) -> (4,4) or (N,4,4)."""
        scalar = np.ndim(t) == 0
        tq = np.clip(np.atleast_1d(np.asarray(t, np.float64)), self.t[0], self.t[-1])
        T = np.repeat(np.eye(4)[None], len(tq), axis=0)
        T[:, :3, :3] = self.slerp(tq).as_matrix()
        for k in range(3):
            T[:, k, 3] = np.interp(tq, self.t, self.p[:, k])
        return T[0] if scalar else T


def deskew(xyz, point_t, odom: PoseTrack, T_base_sensor, t_target):
    """Move points measured at individual times into the sensor frame at t_target.
    xyz (N,3) in the sensor frame at point_t (N,) seconds. Points sharing a timestamp (an Ouster column)
    share one pose, so this costs one interpolation per unique time."""
    uniq, inverse = np.unique(point_t, return_inverse=True)
    T_w_s = odom.at(uniq) @ T_base_sensor                      # sensor pose at each firing time
    T_target = inv(odom.at(t_target) @ T_base_sensor)           # world -> sensor at t_target
    T = T_target[None] @ T_w_s                                  # (U,4,4): sensor@t_i -> sensor@t_target
    R, tr = T[inverse, :3, :3], T[inverse, :3, 3]
    return np.einsum("nij,nj->ni", R, xyz) + tr
