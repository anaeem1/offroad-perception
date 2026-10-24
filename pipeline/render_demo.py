"""Demo video from a replay's render dump (run_replay --dump-render 1): the camera view with lidar points coloured
by their fused class | the BEV traversability grid with tracks, clusters and the planner's corridor | status.

  python -m pipeline.render_demo --run <data root>/runs/replay_00001_demo --out docs/figures/demo_00001.mp4 \
      [--scans 0:1047] [--speed 1.0] [--gif docs/figures/demo_00001.gif --gif-scans 300:450]
"""
import argparse
import json
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import yaml
from PIL import Image

from pipeline.bev import EGO_BOX
from pipeline.bev_memory import REMEMBERED
from pipeline.render_bev import UNKNOWN_RGB, cost_rgb
from pipeline.rellis import DATA_ROOT

W, H = 1920, 1080
BG, FG, DIM, ACCENT = (22, 24, 28), (235, 235, 235), (150, 154, 160), (90, 200, 250)
FONT = cv2.FONT_HERSHEY_SIMPLEX
NAMES = {0: "void", 1: "dirt", 3: "grass", 4: "tree", 5: "pole", 6: "water", 7: "sky", 8: "vehicle", 9: "object",
         10: "asphalt", 12: "building", 15: "log", 17: "person", 18: "fence", 19: "bush", 23: "concrete",
         27: "barrier", 31: "puddle", 33: "mud", 34: "rubble"}
LEGEND = [3, 4, 19, 17, 8, 23, 31, 33, 34, 27, 15, 5]
CONFIRMED, COASTING = 1, 2  # pipeline.tracker states
SCALE = 4  # px per 0.2 m cell -> 800 px for 40 m


def text(img, s, xy, size=0.6, color=FG, thick=1):
    cv2.putText(img, s, xy, FONT, size, color, thick, cv2.LINE_AA)


def palette():
    colors = yaml.safe_load(open(DATA_ROOT / "rellis3d/meta/Rellis_3D_ontology/ontology.yaml"))[1]
    lut = np.full((256, 3), 255, np.uint8)
    for k, v in colors.items():
        lut[int(k)] = v
    return lut


def camera_panel(d, lut):
    if d["jpg"].size == 0:
        img = np.full((600, 960, 3), 40, np.uint8)
        text(img, "no camera frame within +/-50 ms of the lidar sweep: lidar-only", (150, 300), 0.8, DIM, 2)
        return img
    img = cv2.cvtColor(cv2.imdecode(d["jpg"], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    uv = (d["uv"] * 0.5).astype(int)  # projected in the 1920x1200 image; panel is 960x600
    ok = (uv[:, 0] >= 0) & (uv[:, 0] < 960) & (uv[:, 1] >= 0) & (uv[:, 1] < 600)
    for (u, v), c in zip(uv[ok], d["uv_cls"][ok]):
        cv2.circle(img, (int(u), int(v)), 2, tuple(int(x) for x in lut[c]), -1, cv2.LINE_AA)
    return img


def bev_panel(d, lut, extent, res):
    rgb = cost_rgb(d["cost"]).astype(np.float32)
    mem = d["flags"] == REMEMBERED  # cells filled from grid memory only (ADR-0008)
    rgb[mem] = 0.45 * rgb[mem] + 0.55 * np.array([70, 110, 255])
    img = cv2.resize(rgb.astype(np.uint8)[::-1], None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_NEAREST)
    img = np.ascontiguousarray(cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE))  # forward (+x) up, left left

    def px(x, y):
        return int(round((extent / 2 - y) / res * SCALE)), int(round((extent / 2 - x) / res * SCALE))

    for r in (5, 10, 15):  # range rings
        cv2.circle(img, px(0, 0), int(r / res * SCALE), (90, 90, 96), 1, cv2.LINE_AA)
        text(img, f"{r} m", (px(r, 0)[0] + 4, px(r, 0)[1] - 4), 0.45, (170, 170, 176))
    cv2.rectangle(img, px(15.0, 0.9), px(0.9, -0.9), ACCENT, 1, cv2.LINE_AA)  # planner corridor
    cv2.rectangle(img, px(EGO_BOX["x"][1], EGO_BOX["y"][1]), px(EGO_BOX["x"][0], EGO_BOX["y"][0]), FG, 2)
    for x, y, dx, dy, _ in d["dets"]:
        cv2.rectangle(img, px(x + dx / 2, y + dy / 2), px(x - dx / 2, y - dy / 2), (255, 255, 255), 1)
    for tid, state, x, y, vx, vy, cls, _ex in d["tracks"]:
        c = tuple(int(v) for v in lut[int(cls)])
        p = px(x, y)
        cv2.circle(img, p, 8, c, -1 if state == CONFIRMED else 2, cv2.LINE_AA)
        cv2.arrowedLine(img, p, px(x + vx, y + vy), c, 2, cv2.LINE_AA, tipLength=0.3)  # 1 s ahead
        text(img, f"#{int(tid)} {NAMES.get(int(cls), '?')}", (p[0] + 10, p[1] - 8), 0.5, (255, 255, 255), 1)
    return img


def legend(img, x0, y0, lut):
    for i, cid in enumerate(LEGEND):
        x, y = x0 + (i % 6) * 150, y0 + (i // 6) * 28
        cv2.rectangle(img, (x, y - 14), (x + 18, y + 4), tuple(int(v) for v in lut[cid]), -1)
        text(img, NAMES[cid], (x + 26, y), 0.55, DIM)


def frame(d, lut, extent, res, t0, n_frames_seen):
    img = np.full((H, W, 3), BG, np.uint8)
    stamp, have_cam, cam_dt, n_mem, _ms, v_cmd, d_blocked = d["meta"]
    text(img, "Off-road perception stack", (40, 44), 1.0, FG, 2)
    text(img, "RELLIS-3D sequence 00001, held-out test block: reference pipeline replayed from the raw sensor bag",
         (470, 42), 0.6, DIM)
    text(img, f"t = {stamp - t0:5.1f} s", (1700, 44), 0.75, FG, 2)
    img[80:680, 40:1000] = camera_panel(d, lut)
    text(img, "Camera (SegFormer) + lidar points coloured by fused class (late fusion, ADR-0002)", (40, 705), 0.55, DIM)
    bev = bev_panel(d, lut, extent, res)
    img[80:80 + bev.shape[0], 1060:1060 + bev.shape[1]] = bev
    text(img, f"BEV traversability grid, {extent:.0f} m x {extent:.0f} m @ {res:.1f} m, forward = up", (1060, 905), 0.55, DIM)
    # cost legend
    bar = cost_rgb(np.tile(np.arange(0, 255, dtype=np.uint8)[None, :254], (14, 1)))
    img[925:939, 1060:1314] = bar
    text(img, "free", (1060, 960), 0.5, DIM)
    text(img, "lethal", (1262, 960), 0.5, DIM)
    cv2.rectangle(img, (1340, 925), (1360, 939), UNKNOWN_RGB, -1)
    text(img, "UNKNOWN (never free)", (1368, 938), 0.5, DIM)
    cv2.rectangle(img, (1560, 925), (1580, 939), (70, 110, 255), -1)
    text(img, "blue tint: remembered (<= 5 s)", (1588, 938), 0.5, DIM)
    text(img, "dot = confirmed track, ring = coasting, arrow = 1 s ahead", (1060, 985), 0.5, DIM)
    text(img, "white boxes = obstacle clusters, blue outline = planner corridor, white square = vehicle", (1060, 1010), 0.5, DIM)

    n_conf = int(np.sum(d["tracks"][:, 1] == CONFIRMED)) if len(d["tracks"]) else 0
    n_coast = len(d["tracks"]) - n_conf
    y = 750
    cam = (f"camera frame matched {cam_dt:+.0f} ms from the lidar's sweep of its view" if have_cam
           else "no camera frame: LIDAR_ONLY")
    for line in (cam, f"tracks: {n_conf} confirmed, {n_coast} coasting   |   clusters: {len(d['dets'])}"
                 f"   |   remembered cells: {int(n_mem):,}"):
        text(img, line, (40, y), 0.65, FG)
        y += 34
    reason = str(d["reason"])
    if v_cmd > 0:
        msg, col = f"planner (contract rules): {v_cmd:.2f} m/s, corridor clear to {d_blocked:.1f} m", (120, 220, 120)
    else:
        msg, col = f"planner (contract rules): STOP, {reason} at {d_blocked:.1f} m", (240, 140, 90)
    text(img, msg, (40, y), 0.65, col, 2)
    legend(img, 40, y + 50, lut)
    text(img, "Every number in the write-up comes from a committed measurement (docs/measurements/).", (40, 1055), 0.5, DIM)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--scans", default=None)
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed (10 Hz lidar)")
    ap.add_argument("--gif", type=Path)
    ap.add_argument("--gif-scans", default=None)
    args = ap.parse_args()
    m = json.loads((args.run / "metrics.json").read_text())
    extent, res = m["grid"]["extent_m"], m["grid"]["resolution_m"]
    files = sorted((args.run / "render").glob("*.npz"))
    if args.scans:
        lo, hi = (int(v) for v in args.scans.split(":"))
        files = [f for f in files if lo <= int(f.stem) < hi]
    lut = palette()
    t0 = float(np.load(files[0])["meta"][0])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio_ffmpeg.write_frames(str(args.out), (W, H), fps=10 * args.speed, codec="libx264",
                                         pix_fmt_out="yuv420p", bitrate="2500k", macro_block_size=8)
    writer.send(None)
    gif = []
    g_lo, g_hi = (int(v) for v in args.gif_scans.split(":")) if args.gif_scans else (-1, -1)
    for i, f in enumerate(files):
        img = frame(np.load(f), lut, extent, res, t0, i)
        writer.send(np.ascontiguousarray(img).tobytes())
        if args.gif and g_lo <= int(f.stem) < g_hi and int(f.stem) % 2 == 0:  # 5 fps GIF
            gif.append(Image.fromarray(cv2.resize(img, (800, 450), interpolation=cv2.INTER_AREA)))
    writer.close()
    print("wrote", args.out, len(files), "frames")
    if gif:
        gif[0].save(args.gif, save_all=True, append_images=gif[1:], duration=200, loop=0, optimize=True)
        print("wrote", args.gif, len(gif), "frames")


if __name__ == "__main__":
    main()
