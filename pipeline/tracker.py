"""Multi-object tracker: constant-velocity Kalman filter + Hungarian assignment, in the odom frame so that
static obstacles stay static while the machine moves. Reference implementation for the week 3 C++ node.

Track life cycle (contract: perception_msgs/Obstacle):
  TENTATIVE  -> CONFIRMED after `confirm_hits` hits within the last `confirm_window` frames
  CONFIRMED  -> COASTING on a miss (prediction only), back to CONFIRMED on a hit
  deleted after `max_misses` consecutive misses (tentative tracks after 1 miss)
existence_probability: hit p <- p + (1-p)*h, miss p <- p*(1-m). Consumers gate on it.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

TENTATIVE, CONFIRMED, COASTING = 0, 1, 2


@dataclass
class TrackerParams:
    accel_std: float = 1.0        # m/s^2, process noise of the constant-velocity model
    meas_std: float = 0.3         # m, centroid noise
    gate_chi2: float = 9.21       # 2-dof, 99 %
    confirm_hits: int = 3
    confirm_window: int = 5
    max_misses: int = 5           # 0.5 s at 10 Hz
    p_init: float = 0.3
    p_hit: float = 0.3
    p_miss: float = 0.3


@dataclass
class Track:
    id: int
    x: np.ndarray                 # [px, py, vx, vy] in odom
    P: np.ndarray
    z: float
    dims: np.ndarray
    class_votes: dict
    class_conf: float
    state: int = TENTATIVE
    existence: float = 0.3
    age: int = 1
    misses: int = 0
    history: list = field(default_factory=list)  # hit/miss, newest last

    @property
    def semantic_class(self):
        return max(self.class_votes, key=self.class_votes.get)


class Tracker:
    def __init__(self, params: TrackerParams = TrackerParams()):
        self.p = params
        self.tracks: list[Track] = []
        self.next_id = 1
        self.t = None

    def _F_Q(self, dt):
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        q = self.p.accel_std ** 2
        G = np.array([[dt * dt / 2, 0], [0, dt * dt / 2], [dt, 0], [0, dt]])
        return F, G @ G.T * q

    def step(self, t, detections_odom):
        """detections_odom: list of (xyz_odom (3,), dims (3,), class id, class_conf). Returns live tracks."""
        p = self.p
        if self.t is not None:
            F, Q = self._F_Q(t - self.t)
            for tr in self.tracks:
                tr.x = F @ tr.x
                tr.P = F @ tr.P @ F.T + Q
        self.t = t
        H = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
        R = np.eye(2) * p.meas_std ** 2
        nT, nD = len(self.tracks), len(detections_odom)
        cost = np.full((nT, nD), 1e6)
        for i, tr in enumerate(self.tracks):
            S_inv = np.linalg.inv(H @ tr.P @ H.T + R)
            for j, d in enumerate(detections_odom):
                r = d[0][:2] - tr.x[:2]
                m = r @ S_inv @ r
                if m < p.gate_chi2:
                    cost[i, j] = m
        rows, cols = linear_sum_assignment(cost) if nT and nD else ([], [])
        hit_t, hit_d = set(), set()
        for i, j in zip(rows, cols):
            if cost[i, j] >= 1e6:
                continue
            tr, d = self.tracks[i], detections_odom[j]
            S = H @ tr.P @ H.T + R
            K = tr.P @ H.T @ np.linalg.inv(S)
            tr.x = tr.x + K @ (d[0][:2] - tr.x[:2])
            tr.P = (np.eye(4) - K @ H) @ tr.P
            tr.z, tr.dims = 0.7 * tr.z + 0.3 * d[0][2], 0.7 * tr.dims + 0.3 * d[1]
            tr.class_votes[d[2]] = tr.class_votes.get(d[2], 0) + d[3]
            tr.class_conf = 0.7 * tr.class_conf + 0.3 * d[3]
            tr.existence += (1 - tr.existence) * p.p_hit
            tr.misses = 0
            tr.history.append(1)
            hit_t.add(i)
            hit_d.add(j)
        for i, tr in enumerate(self.tracks):
            tr.age += 1
            if i not in hit_t:
                tr.misses += 1
                tr.existence *= 1 - p.p_miss
                tr.history.append(0)
            tr.history = tr.history[-p.confirm_window:]
            if tr.state == TENTATIVE and sum(tr.history) >= p.confirm_hits:
                tr.state = CONFIRMED
            if tr.state in (CONFIRMED, COASTING):
                tr.state = CONFIRMED if i in hit_t else COASTING
        self.tracks = [tr for tr in self.tracks
                       if not (tr.state == TENTATIVE and tr.misses >= 1) and tr.misses <= p.max_misses]
        for j, d in enumerate(detections_odom):
            if j in hit_d:
                continue
            P0 = np.diag([p.meas_std ** 2, p.meas_std ** 2, 4.0, 4.0])
            self.tracks.append(Track(id=self.next_id, x=np.r_[d[0][:2], 0, 0], P=P0, z=float(d[0][2]),
                                     dims=np.asarray(d[1], float), class_votes={d[2]: d[3]}, class_conf=d[3],
                                     existence=p.p_init, history=[1]))
            self.next_id += 1
        return self.tracks
