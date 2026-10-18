// Lidar range-image projection (SalsaNext input) and lidar->camera projection.
// C++ ports of ml/lidar/range_projection.py and ml/fusion/projection.py; parity-tested against them.
#pragma once
#include <array>
#include <cstdint>
#include <vector>

#include "offroad_perception/math.hpp"

namespace offroad {

struct RangeConfig {
  int H{64}, W{2048};
  float fov_up_deg{22.5f}, fov_down_deg{-22.5f};
  std::array<float, 5> means{4.84649722f, -0.187910314f, 0.193718327f, -0.246564824f, 0.00260723157f};
  std::array<float, 5> stds{6.05381850f, 5.61048984f, 5.27298844f, 0.849105890f, 0.00284712457f};
};

struct RangeImage {
  std::vector<float> input;   // (5, H, W) normalized network input
  std::vector<float> range;   // (H, W), -1 where empty
  std::vector<int32_t> px, py;  // per input point: pixel it projects to
};

// xyz: N*3 floats (x,y,z interleaved, sensor order); remission: N floats (intensity / 65535).
// Float32 arithmetic in the same order as the NumPy reference, including its quirks (zero points get
// depth 1e-4; the pixel holding point index 0 counts as empty).
RangeImage project_range(const float* xyz, const float* remission, size_t n, const RangeConfig& cfg);

struct Camera {
  double fx, fy, cx, cy;
  std::array<double, 5> D;  // plumb_bob k1 k2 p1 p2 k3
  int width{1920}, height{1200};
};

struct CameraHit {
  double u, v;    // pixel
  uint32_t idx;   // index into the input points
  double depth;   // camera z
};

// Points (sensor frame, doubles) -> pixel hits inside the image. Culls points behind the camera or far
// outside its FOV before distortion, exactly as the Python reference does.
std::vector<CameraHit> project_to_camera(const std::vector<Vec3>& pts, const std::vector<uint32_t>& ids,
                                         const Camera& cam, const Mat4& T_cam_sensor, double min_depth = 0.5,
                                         double fov_margin = 0.1);

}  // namespace offroad
