#include "offroad_perception/projection.hpp"

#include <algorithm>
#include <cmath>

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
#pragma omp parallel for schedule(static)
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
  // The nearest point wins each pixel. The reference writes far-to-near after a sort; a single
  // pass keeping the minimum depth per pixel gives the same image except for exact depth ties
  // (whose order the reference's unstable sort leaves undefined anyway), at O(N) instead of O(N log N).
  const size_t HW = static_cast<size_t>(H) * W;
  std::vector<int32_t> idx(HW, -1);
  for (size_t i = 0; i < n; ++i) {
    const size_t p = static_cast<size_t>(out.py[i]) * W + out.px[i];
    const int32_t cur = idx[p];
    if (cur < 0 || depth[i] < depth[cur]) idx[p] = static_cast<int32_t>(i);
  }
  // One pass from the winning point straight to the normalized input (empty pixels read -1, as in the
  // reference, and are then masked). Memory-bound, so deliberately single-threaded.
  out.range.assign(HW, -1.0f);
  out.input.resize(5 * HW);
  float* in = out.input.data();
  for (size_t p = 0; p < HW; ++p) {
    const int32_t i = idx[p];
    const float mask = i > 0 ? 1.0f : 0.0f;  // upstream quirk: index 0 counts as empty
    float v[5] = {-1.0f, -1.0f, -1.0f, -1.0f, -1.0f};
    if (i >= 0) {
      v[0] = out.range[p] = depth[i];
      v[1] = xyz[3 * i]; v[2] = xyz[3 * i + 1]; v[3] = xyz[3 * i + 2];
      v[4] = remission[i];
    }
    for (int c = 0; c < 5; ++c) in[c * HW + p] = (v[c] - cfg.means[c]) / cfg.stds[c] * mask;
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
