#include "offroad_perception/projection.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace offroad {

RangeImage project_range(const float* xyz, const float* remission, size_t n, const RangeConfig& cfg) {
  const int H = cfg.H, W = cfg.W;
  // NumPy evaluates these in float64 and then mixes them into float32 arrays as float32 scalars.
  const double fov_up = cfg.fov_up_deg / 180.0 * M_PI, fov_down = cfg.fov_down_deg / 180.0 * M_PI;
  const float fov_f = static_cast<float>(std::abs(fov_down) + std::abs(fov_up));
  const float fov_down_abs_f = static_cast<float>(std::abs(fov_down));
  const float pi_f = static_cast<float>(M_PI);

  RangeImage out;
  out.px.resize(n);
  out.py.resize(n);
  std::vector<float> depth(n);
  for (size_t i = 0; i < n; ++i) {
    const float x = xyz[3 * i], y = xyz[3 * i + 1], z = xyz[3 * i + 2];
    float d = std::sqrt((x * x + y * y) + z * z);
    if (d == 0.0f) d = 1e-4f;
    depth[i] = d;
    const float yaw = -std::atan2(y, x);
    const float pitch = std::asin(z / d);
    float fx = std::floor(0.5f * (yaw / pi_f + 1.0f) * static_cast<float>(W));
    float fy = std::floor((1.0f - (pitch + fov_down_abs_f) / fov_f) * static_cast<float>(H));
    fx = std::max(0.0f, std::min(static_cast<float>(W - 1), fx));
    fy = std::max(0.0f, std::min(static_cast<float>(H - 1), fy));
    out.px[i] = static_cast<int32_t>(fx);
    out.py[i] = static_cast<int32_t>(fy);
  }
  // Far-to-near write order so the nearest point wins each pixel.
  std::vector<uint32_t> order(n);
  std::iota(order.begin(), order.end(), 0u);
  std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return depth[a] > depth[b]; });

  const size_t HW = static_cast<size_t>(H) * W;
  out.range.assign(HW, -1.0f);
  std::vector<float> chx(HW, -1.0f), chy(HW, -1.0f), chz(HW, -1.0f), chr(HW, -1.0f);
  std::vector<int32_t> idx(HW, -1);
  for (uint32_t i : order) {
    const size_t p = static_cast<size_t>(out.py[i]) * W + out.px[i];
    out.range[p] = depth[i];
    chx[p] = xyz[3 * i];
    chy[p] = xyz[3 * i + 1];
    chz[p] = xyz[3 * i + 2];
    chr[p] = remission[i];
    idx[p] = static_cast<int32_t>(i);
  }
  out.input.resize(5 * HW);
  const std::vector<float>* ch[5] = {&out.range, &chx, &chy, &chz, &chr};
  for (int c = 0; c < 5; ++c)
    for (size_t p = 0; p < HW; ++p) {
      const float mask = idx[p] > 0 ? 1.0f : 0.0f;  // upstream quirk: index 0 counts as empty
      out.input[c * HW + p] = ((*ch[c])[p] - cfg.means[c]) / cfg.stds[c] * mask;
    }
  return out;
}

std::vector<CameraHit> project_to_camera(const std::vector<Vec3>& pts, const std::vector<uint32_t>& ids,
                                         const Camera& cam, const Mat4& T, double min_depth, double fov_margin) {
  std::vector<CameraHit> hits;
  const double xmin = -cam.cx / cam.fx - fov_margin, xmax = (cam.width - cam.cx) / cam.fx + fov_margin;
  const double ymin = -cam.cy / cam.fy - fov_margin, ymax = (cam.height - cam.cy) / cam.fy + fov_margin;
  const double k1 = cam.D[0], k2 = cam.D[1], p1 = cam.D[2], p2 = cam.D[3], k3 = cam.D[4];
  for (size_t i = 0; i < pts.size(); ++i) {
    const Vec3 pc = apply(T, pts[i]);
    if (!(pc.z > min_depth)) continue;
    const double xn = pc.x / pc.z, yn = pc.y / pc.z;
    if (!(xn > xmin && xn < xmax && yn > ymin && yn < ymax)) continue;
    const double r2 = xn * xn + yn * yn;
    const double radial = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2;
    const double xd = xn * radial + 2 * p1 * xn * yn + p2 * (r2 + 2 * xn * xn);
    const double yd = yn * radial + p1 * (r2 + 2 * yn * yn) + 2 * p2 * xn * yn;
    const double u = cam.fx * xd + cam.cx, v = cam.fy * yd + cam.cy;
    if (u >= 0 && u < cam.width && v >= 0 && v < cam.height) hits.push_back({u, v, ids[i], pc.z});
  }
  return hits;
}

}  // namespace offroad
