#include "offroad_perception/tracker.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>

namespace offroad {

int Track::semantic_class() const {
  int best = 0;
  double bv = -1;
  for (const auto& [c, v] : class_votes)  // std::map iterates in key order; first max wins, like Python max()
    if (v > bv) { bv = v; best = c; }
  return best;
}

std::vector<std::pair<int, int>> hungarian(const std::vector<double>& cost, int rows, int cols) {
  // Shortest augmenting path (Jonker-Volgenant style) on the transposed problem when rows > cols.
  const bool transpose = rows > cols;
  const int n = transpose ? cols : rows, m = transpose ? rows : cols;
  auto c = [&](int i, int j) { return transpose ? cost[j * cols + i] : cost[i * cols + j]; };
  const double INF = std::numeric_limits<double>::infinity();
  std::vector<double> u(n + 1, 0), v(m + 1, 0);
  std::vector<int> p(m + 1, 0), way(m + 1, 0);
  for (int i = 1; i <= n; ++i) {
    p[0] = i;
    int j0 = 0;
    std::vector<double> minv(m + 1, INF);
    std::vector<char> used(m + 1, false);
    do {
      used[j0] = true;
      const int i0 = p[j0];
      double delta = INF;
      int j1 = 0;
      for (int j = 1; j <= m; ++j) {
        if (used[j]) continue;
        const double cur = c(i0 - 1, j - 1) - u[i0] - v[j];
        if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
        if (minv[j] < delta) { delta = minv[j]; j1 = j; }
      }
      for (int j = 0; j <= m; ++j) {
        if (used[j]) { u[p[j]] += delta; v[j] -= delta; }
        else minv[j] -= delta;
      }
      j0 = j1;
    } while (p[j0] != 0);
    do { const int j1 = way[j0]; p[j0] = p[j1]; j0 = j1; } while (j0);
  }
  std::vector<std::pair<int, int>> out;
  for (int j = 1; j <= m; ++j)
    if (p[j]) out.push_back(transpose ? std::make_pair(j - 1, p[j] - 1) : std::make_pair(p[j] - 1, j - 1));
  std::sort(out.begin(), out.end());
  return out;
}

namespace {
using M4 = std::array<double, 16>;
M4 mul4(const M4& a, const M4& b) {
  M4 o{};
  for (int r = 0; r < 4; ++r)
    for (int k = 0; k < 4; ++k)
      for (int c = 0; c < 4; ++c) o[r * 4 + c] += a[r * 4 + k] * b[k * 4 + c];
  return o;
}
M4 transpose4(const M4& a) {
  M4 o{};
  for (int r = 0; r < 4; ++r)
    for (int c = 0; c < 4; ++c) o[c * 4 + r] = a[r * 4 + c];
  return o;
}
}  // namespace

const std::vector<Track>& Tracker::step(double t, const std::vector<TrackInput>& dets) {
  if (has_t_) {
    const double dt = t - t_, q = p_.accel_std * p_.accel_std;
    const M4 F{1, 0, dt, 0, 0, 1, 0, dt, 0, 0, 1, 0, 0, 0, 0, 1};
    const double g0 = dt * dt / 2, g1 = dt;  // G = [[g0,0],[0,g0],[g1,0],[0,g1]], Q = G G^T q
    const M4 Q{g0 * g0 * q, 0, g0 * g1 * q, 0, 0, g0 * g0 * q, 0, g0 * g1 * q,
               g1 * g0 * q, 0, g1 * g1 * q, 0, 0, g1 * g0 * q, 0, g1 * g1 * q};
    for (Track& tr : tracks_) {
      tr.x = {tr.x[0] + dt * tr.x[2], tr.x[1] + dt * tr.x[3], tr.x[2], tr.x[3]};
      M4 P = mul4(mul4(F, tr.P), transpose4(F));
      for (int k = 0; k < 16; ++k) P[k] += Q[k];
      tr.P = P;
    }
  }
  has_t_ = true;
  t_ = t;
  const double r = p_.meas_std * p_.meas_std;
  const int nT = static_cast<int>(tracks_.size()), nD = static_cast<int>(dets.size());
  std::vector<double> cost(static_cast<size_t>(nT) * nD, 1e6);
  for (int i = 0; i < nT; ++i) {
    const auto& P = tracks_[i].P;
    const double s00 = P[0] + r, s01 = P[1], s10 = P[4], s11 = P[5] + r;  // S = H P H^T + R
    const double det = s00 * s11 - s01 * s10;
    for (int j = 0; j < nD; ++j) {
      const double dx = dets[j].xyz.x - tracks_[i].x[0], dy = dets[j].xyz.y - tracks_[i].x[1];
      const double m = (dx * (s11 * dx - s01 * dy) + dy * (-s10 * dx + s00 * dy)) / det;
      if (m < p_.gate_chi2) cost[static_cast<size_t>(i) * nD + j] = m;
    }
  }
  std::set<int> hit_t, hit_d;
  if (nT && nD) {
    for (auto [i, j] : hungarian(cost, nT, nD)) {
      if (cost[static_cast<size_t>(i) * nD + j] >= 1e6) continue;
      Track& tr = tracks_[i];
      const TrackInput& d = dets[j];
      const auto& P = tr.P;
      const double s00 = P[0] + r, s01 = P[1], s10 = P[4], s11 = P[5] + r, det = s00 * s11 - s01 * s10;
      const double i00 = s11 / det, i01 = -s01 / det, i10 = -s10 / det, i11 = s00 / det;
      double K[4][2];  // K = P H^T S^-1
      for (int k = 0; k < 4; ++k) {
        K[k][0] = P[k * 4 + 0] * i00 + P[k * 4 + 1] * i10;
        K[k][1] = P[k * 4 + 0] * i01 + P[k * 4 + 1] * i11;
      }
      const double rx = d.xyz.x - tr.x[0], ry = d.xyz.y - tr.x[1];
      for (int k = 0; k < 4; ++k) tr.x[k] += K[k][0] * rx + K[k][1] * ry;
      M4 IKH{};
      for (int a = 0; a < 4; ++a)
        for (int b = 0; b < 4; ++b) IKH[a * 4 + b] = (a == b) - (b == 0 ? K[a][0] : b == 1 ? K[a][1] : 0.0);
      tr.P = mul4(IKH, P);
      tr.z = 0.7 * tr.z + 0.3 * d.xyz.z;
      tr.dims = {0.7 * tr.dims.x + 0.3 * d.dims.x, 0.7 * tr.dims.y + 0.3 * d.dims.y, 0.7 * tr.dims.z + 0.3 * d.dims.z};
      tr.class_votes[d.semantic_class] += d.class_confidence;
      tr.class_conf = 0.7 * tr.class_conf + 0.3 * d.class_confidence;
      tr.existence += (1 - tr.existence) * p_.p_hit;
      tr.misses = 0;
      tr.history.push_back(1);
      hit_t.insert(i);
      hit_d.insert(j);
    }
  }
  for (int i = 0; i < nT; ++i) {
    Track& tr = tracks_[i];
    tr.age += 1;
    if (!hit_t.count(i)) {
      tr.misses += 1;
      tr.existence *= 1 - p_.p_miss;
      tr.history.push_back(0);
    }
    if (static_cast<int>(tr.history.size()) > p_.confirm_window)
      tr.history.erase(tr.history.begin(), tr.history.end() - p_.confirm_window);
    int hits = 0;
    for (int h : tr.history) hits += h;
    if (tr.state == TENTATIVE && hits >= p_.confirm_hits) tr.state = CONFIRMED;
    if (tr.state == CONFIRMED || tr.state == COASTING) tr.state = hit_t.count(i) ? CONFIRMED : COASTING;
  }
  tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(), [&](const Track& tr) {
                  return (tr.state == TENTATIVE && tr.misses >= 1) || tr.misses > p_.max_misses;
                }), tracks_.end());
  for (int j = 0; j < nD; ++j) {
    if (hit_d.count(j)) continue;
    const TrackInput& d = dets[j];
    Track tr;
    tr.id = next_id_++;
    tr.x = {d.xyz.x, d.xyz.y, 0, 0};
    tr.P = {r, 0, 0, 0, 0, r, 0, 0, 0, 0, 4.0, 0, 0, 0, 0, 4.0};
    tr.z = d.xyz.z;
    tr.dims = d.dims;
    tr.class_votes[d.semantic_class] = d.class_confidence;
    tr.class_conf = d.class_confidence;
    tr.existence = p_.p_init;
    tr.history = {1};
    tracks_.push_back(tr);
  }
  return tracks_;
}

}  // namespace offroad
