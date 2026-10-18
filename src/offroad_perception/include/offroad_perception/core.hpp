// Post-model pipeline stages: pose interpolation + deskew, late fusion, BEV grid, clustering.
// C++ ports of pipeline/{geometry,fusion,bev,cluster}.py; parity-tested against them (test/parity.cpp).
#pragma once
#include <array>
#include <cstdint>
#include <vector>

#include "offroad_perception/math.hpp"

namespace offroad {

constexpr int kNumClasses = 20;  // RELLIS-3D ontology ids below, in this order
constexpr std::array<int, kNumClasses> kOntologyIds{0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 17, 18, 19, 23, 27, 31, 33, 34};
int onto_index(int ontology_id);  // -1 if not an ontology id

// ---- poses -------------------------------------------------------------------------------------------
class PoseTrack {  // T_world_body over time; linear position, slerp rotation, clamped at the ends
 public:
  void add(double t, const Vec3& p, const Quat& q);  // must be called in increasing time
  Mat4 at(double t) const;
  bool empty() const { return t_.empty(); }
  double t_front() const { return t_.front(); }
  double t_back() const { return t_.back(); }
 private:
  std::vector<double> t_;
  std::vector<Vec3> p_;
  std::vector<Quat> q_;
};

// Move points measured at their own times into the sensor frame at t_target (one pose per unique time).
std::vector<Vec3> deskew(const std::vector<Vec3>& pts, const std::vector<double>& point_t, const PoseTrack& odom,
                         const Mat4& T_base_sensor, double t_target);

// ---- fusion ------------------------------------------------------------------------------------------
using Dist = std::array<float, kNumClasses>;
// Lidar probabilities over its training classes -> ontology space. train_to_onto[j] = ontology id of
// training class j; bucket = ontology ids folded into training class 0.
Dist lidar_to_ontology(const float* p_train, int n_train, const std::vector<int>& train_to_onto,
                       const std::vector<int>& bucket);
Dist camera_to_ontology(const float* p_cam, const std::vector<int>& class_ids);
Dist fuse(const Dist& lidar, const Dist& camera, float alpha, bool loglinear);

// ---- BEV grid ----------------------------------------------------------------------------------------
struct GridParams {
  double extent_m{40.0}, resolution_m{0.2}, z_min{-3.0}, z_max{3.0};
  double ground_window_m{1.0}, obstacle_height_m{0.3}, clearance_m{2.5};
  std::vector<int> geometric_exempt{3, 19};
  int n() const { return static_cast<int>(std::lround(extent_m / resolution_m)); }
};

struct Grid {
  int n{0};
  std::vector<uint8_t> cost, confidence, semantic_class, source_flags, geometric;  // n*n row-major (row = y)
};

// cost_lut[k]: cost of ontology class index k (255 unknown, -1 = contributes nothing, e.g. sky).
Grid build_grid(const std::vector<Vec3>& xyz_base, const std::vector<Dist>& probs, const std::vector<uint8_t>& from_camera,
                const GridParams& p, const std::array<int, kNumClasses>& cost_lut);

// ---- clustering --------------------------------------------------------------------------------------
struct Detection {
  Vec3 xyz, dims;
  int semantic_class{0};
  double class_confidence{0};
  int n_points{0};
};

std::vector<Detection> cluster(const std::vector<Vec3>& xyz_base, const std::vector<Dist>& probs,
                               const std::vector<uint8_t>& geometric_cells, const GridParams& p,
                               const std::array<bool, kNumClasses>& obstacle_lut, int min_points = 5);

}  // namespace offroad
