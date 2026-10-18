// Multi-object tracker: constant-velocity Kalman filter + Hungarian assignment in the odom frame.
// C++ port of pipeline/tracker.py (same parameters, life cycle and existence update); parity-tested.
#pragma once
#include <array>
#include <cstdint>
#include <map>
#include <vector>

#include "offroad_perception/math.hpp"

namespace offroad {

enum TrackState : uint8_t { TENTATIVE = 0, CONFIRMED = 1, COASTING = 2 };

struct TrackerParams {
  double accel_std{1.0}, meas_std{0.3}, gate_chi2{9.21};
  int confirm_hits{3}, confirm_window{5}, max_misses{5};
  double p_init{0.3}, p_hit{0.3}, p_miss{0.3};
};

struct TrackInput {  // one detection, already in the odom frame
  Vec3 xyz, dims;
  int semantic_class;
  double class_confidence;
};

struct Track {
  uint32_t id;
  std::array<double, 4> x;   // px, py, vx, vy
  std::array<double, 16> P;  // 4x4 row-major
  double z;
  Vec3 dims;
  std::map<int, double> class_votes;
  double class_conf;
  TrackState state{TENTATIVE};
  double existence;
  uint32_t age{1};
  int misses{0};
  std::vector<int> history;
  int semantic_class() const;
};

// Rectangular assignment minimizing total cost (same optimum as scipy.optimize.linear_sum_assignment).
std::vector<std::pair<int, int>> hungarian(const std::vector<double>& cost, int rows, int cols);

class Tracker {
 public:
  explicit Tracker(TrackerParams p = {}) : p_(p) {}
  const std::vector<Track>& step(double t, const std::vector<TrackInput>& dets);
  const std::vector<Track>& tracks() const { return tracks_; }
 private:
  TrackerParams p_;
  std::vector<Track> tracks_;
  uint32_t next_id_{1};
  bool has_t_{false};
  double t_{0};
};

}  // namespace offroad
