"""Reference port of the contract-consuming planner (src/offroad_perception/src/planner_consumer_node.cpp), so
the offline replay can report what a planner that follows the contract would do with each output:
  1. UNKNOWN (255) and lethal (254) cells in the forward corridor both block it; other costs slow it down
  2. obstacles count only above the existence gate; COASTING tracks get a doubled lateral margin
  3. no camera -> LIDAR_ONLY speed cap. (Output age and SAFE_STOP are runtime properties; offline they do
     not apply, so this measures what the outputs themselves allow.)
Parameters are the deployed ones (pipeline/gen_ros_params.py).
"""
import numpy as np

V_MAX, LIDAR_ONLY_FACTOR, HALF_WIDTH, CORRIDOR_START, LOOKAHEAD, STOP_DIST, GAIN, EXISTENCE_GATE = \
    2.0, 0.5, 0.9, 0.9, 15.0, 2.0, 0.5, 0.5
COASTING = 2  # pipeline.tracker state


def decide(cost, cx, cy, tracks_base, camera_used):
    """cost: grid (rows = y, cols = x); cx, cy: cell-centre coordinates (meshgrid, base_link);
    tracks_base: [(x, y, dim_x, dim_y, state, existence)] in base_link. Returns (v, reason, d_blocked, d_obstacle)."""
    corr = (cx >= CORRIDOR_START) & (cx <= LOOKAHEAD) & (np.abs(cy) <= HALF_WIDTH)
    c = cost[corr]
    blocked = c >= 254
    d_blocked = float(cx[corr][blocked].min()) if blocked.any() else LOOKAHEAD
    worst = float(c[~blocked].max()) if (~blocked).any() else 0.0
    d_obstacle = LOOKAHEAD
    for x, y, dx, dy, state, existence in tracks_base:
        margin = (2.0 if state == COASTING else 1.0) * (HALF_WIDTH + dy / 2)
        if existence >= EXISTENCE_GATE and x > 0 and abs(y) < margin:
            d_obstacle = min(d_obstacle, x - dx / 2)
    factor = (1.0 - worst / 254.0) * (1.0 if camera_used else LIDAR_ONLY_FACTOR)
    v = float(np.clip(GAIN * (min(d_blocked, d_obstacle) - STOP_DIST), 0.0, V_MAX) * factor)
    if v > 0:
        reason = "ok" if camera_used else "LIDAR_ONLY cap"
    elif d_obstacle < d_blocked:
        reason = "obstacle ahead"
    else:
        first = corr & (cx == d_blocked) & (cost >= 254)
        reason = "unknown ahead" if (cost[first] == 255).any() else "lethal ahead"
    return v, reason, d_blocked, d_obstacle
