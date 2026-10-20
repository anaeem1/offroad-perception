#include "offroad_perception/core.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <queue>

namespace offroad {

int onto_index(int id) {
  for (int k = 0; k < kNumClasses; ++k)
    if (kOntologyIds[k] == id) return k;
  return -1;
}

// ---- poses -------------------------------------------------------------------------------------------
void PoseTrack::add(double t, const Vec3& p, const Quat& q) {
  if (!t_.empty() && t <= t_.back()) return;  // keep strictly increasing, as the Python reference does
  t_.push_back(t);
  p_.push_back(p);
  q_.push_back(q);
}

void PoseTrack::trim_before(double t) {
  size_t k = 0;
  while (k + 2 < t_.size() && t_[k + 1] < t) ++k;
  t_.erase(t_.begin(), t_.begin() + k);
  p_.erase(p_.begin(), p_.begin() + k);
  q_.erase(q_.begin(), q_.begin() + k);
}

Mat4 PoseTrack::at(double t) const {
  t = std::min(std::max(t, t_.front()), t_.back());
  size_t i = std::upper_bound(t_.begin(), t_.end(), t) - t_.begin();
  if (i == 0) i = 1;
  if (i >= t_.size()) i = t_.size() - 1;
  if (t_.size() == 1) return from_qt(q_[0], p_[0]);
  const double t0 = t_[i - 1], t1 = t_[i], u = (t - t0) / (t1 - t0);
  const Vec3 p{p_[i - 1].x + u * (p_[i].x - p_[i - 1].x), p_[i - 1].y + u * (p_[i].y - p_[i - 1].y),
               p_[i - 1].z + u * (p_[i].z - p_[i - 1].z)};
  return from_qt(slerp(q_[i - 1], q_[i], u), p);
}

std::vector<Vec3> deskew(const std::vector<Vec3>& pts, const std::vector<double>& point_t, const PoseTrack& odom,
                         const Mat4& T_base_sensor, double t_target) {
  const Mat4 T_target = inverse_rigid(mul(odom.at(t_target), T_base_sensor));
  // Ouster points arrive column by column and a column shares one firing time, so the transform is
  // recomputed only when the time changes (2048 interpolations per sweep instead of one per point).
  std::vector<Vec3> out(pts.size());
  Mat4 T;
  double last = std::numeric_limits<double>::quiet_NaN();
  for (size_t i = 0; i < pts.size(); ++i) {
    if (!(point_t[i] == last)) { T = mul(T_target, mul(odom.at(point_t[i]), T_base_sensor)); last = point_t[i]; }
    out[i] = apply(T, pts[i]);
  }
  return out;
}

// ---- fusion ------------------------------------------------------------------------------------------
Dist lidar_to_ontology(const float* p, int n_train, const std::vector<int>& train_to_onto, const std::vector<int>& bucket) {
  Dist out{};
  for (int j = 1; j < n_train; ++j) out[onto_index(train_to_onto[j])] = p[j];
  for (int c : bucket) out[onto_index(c)] = p[0] / static_cast<float>(bucket.size());
  return out;
}

Dist camera_to_ontology(const float* p_cam, const std::vector<int>& class_ids) {
  Dist out{};
  for (size_t j = 0; j < class_ids.size(); ++j) out[onto_index(class_ids[j])] = p_cam[j];
  return out;
}

Dist fuse(const Dist& l, const Dist& c, float alpha, bool loglinear) {
  Dist out{};
  if (!loglinear) {
    for (int k = 0; k < kNumClasses; ++k) out[k] = alpha * l[k] + (1 - alpha) * c[k];
    return out;
  }
  float mx = -std::numeric_limits<float>::infinity();
  for (int k = 0; k < kNumClasses; ++k) {
    out[k] = alpha * std::log(l[k] + 1e-6f) + (1 - alpha) * std::log(c[k] + 1e-6f);
    mx = std::max(mx, out[k]);
  }
  float s = 0;
  for (int k = 0; k < kNumClasses; ++k) s += (out[k] = std::exp(out[k] - mx));
  for (int k = 0; k < kNumClasses; ++k) out[k] /= s;
  return out;
}

namespace {
int argmax(const Dist& d) { return static_cast<int>(std::max_element(d.begin(), d.end()) - d.begin()); }

bool cell_of(const Vec3& p, const GridParams& g, int& i, int& j) {
  i = static_cast<int>(std::floor((p.x + g.extent_m / 2) / g.resolution_m));
  j = static_cast<int>(std::floor((p.y + g.extent_m / 2) / g.resolution_m));
  return i >= 0 && j >= 0 && i < g.n() && j < g.n();
}
}  // namespace

// ---- BEV grid ----------------------------------------------------------------------------------------
Grid build_grid(const std::vector<Vec3>& xyz, const std::vector<Dist>& probs, const std::vector<uint8_t>& from_camera,
                const GridParams& g, const std::array<int, kNumClasses>& cost_lut) {
  const int n = g.n();
  const size_t nn = static_cast<size_t>(n) * n;
  const int sky = onto_index(7);
  struct P { size_t cell; double z; int cls; float pc, pmax; bool cam; };
  std::vector<P> kept;
  kept.reserve(xyz.size());
  for (size_t k = 0; k < xyz.size(); ++k) {
    int i, j;
    const int cls = argmax(probs[k]);
    if (!cell_of(xyz[k], g, i, j) || !(xyz[k].z > g.z_min && xyz[k].z < g.z_max) || cls == sky) continue;
    kept.push_back({static_cast<size_t>(j) * n + i, xyz[k].z, cls, probs[k][cls],
                    *std::max_element(probs[k].begin(), probs[k].end()), from_camera[k] != 0});
  }
  const double inf = std::numeric_limits<double>::infinity();
  std::vector<double> zmin(nn, inf), tmp(nn, inf), ground(nn, inf), hmax(nn, -inf), meanp(nn, 0.0), cnt(nn, 0.0);
  for (const P& q : kept) zmin[q.cell] = std::min(zmin[q.cell], q.z);
  // Local ground = minimum over a win x win window (scipy minimum_filter placement). Separable: row pass,
  // then column pass; the minimum over a rectangle is the minimum of row minima, so this is exact.
  const int win = std::max(1, static_cast<int>(std::lround(g.ground_window_m / g.resolution_m)));
  const int lo = win / 2, hi = win - 1 - lo;
  for (int r = 0; r < n; ++r)
    for (int c = 0; c < n; ++c) {
      double m = inf;
      for (int cc = std::max(0, c - lo); cc <= std::min(n - 1, c + hi); ++cc) m = std::min(m, zmin[r * n + cc]);
      tmp[r * n + c] = m;
    }
  for (int r = 0; r < n; ++r)
    for (int c = 0; c < n; ++c) {
      double m = inf;
      for (int rr = std::max(0, r - lo); rr <= std::min(n - 1, r + hi); ++rr) m = std::min(m, tmp[rr * n + c]);
      ground[r * n + c] = m;
    }
  // Votes only for occupied cells (compact slots), accumulated in point order as in the reference.
  std::vector<int32_t> slot(nn, -1);
  std::vector<std::array<float, kNumClasses>> votes;
  std::vector<size_t> cell_of_slot;
  std::vector<uint8_t> cam_any(nn, 0);
  for (const P& q : kept) {
    const double h = q.z - ground[q.cell];
    if (h < g.clearance_m) hmax[q.cell] = std::max(hmax[q.cell], h);
    if (slot[q.cell] < 0) {
      slot[q.cell] = static_cast<int32_t>(votes.size());
      votes.push_back({});
      cell_of_slot.push_back(q.cell);
    }
    votes[slot[q.cell]][q.cls] += q.pc;
    meanp[q.cell] += q.pmax;
    cnt[q.cell] += 1;
    if (q.cam) cam_any[q.cell] = 1;
  }
  Grid out;
  out.n = n;
  out.cost.assign(nn, 255);
  out.confidence.assign(nn, 0);
  out.semantic_class.assign(nn, 0);
  out.source_flags.assign(nn, 0);
  out.geometric.assign(nn, 0);
  for (size_t s_ = 0; s_ < votes.size(); ++s_) {
    const size_t c = cell_of_slot[s_];
    const auto& v = votes[s_];
    float total = 0;
    for (float x : v) total += x;
    if (!(total > 0)) continue;
    const int dom = static_cast<int>(std::max_element(v.begin(), v.end()) - v.begin());
    const double share = v[dom] / std::max(total, 1e-9f);
    const double mp = cnt[c] > 0 ? meanp[c] / cnt[c] : 0.0;
    const int sem = cost_lut[dom];
    out.cost[c] = static_cast<uint8_t>(sem < 0 ? 255 : sem);
    const bool exempt = std::find(g.geometric_exempt.begin(), g.geometric_exempt.end(), kOntologyIds[dom]) != g.geometric_exempt.end();
    if (hmax[c] > g.obstacle_height_m && !exempt) { out.cost[c] = 254; out.geometric[c] = 1; }
    out.source_flags[c] = 1 | (cam_any[c] ? 2 : 0);
    out.semantic_class[c] = static_cast<uint8_t>(kOntologyIds[dom]);
    out.confidence[c] = static_cast<uint8_t>(std::nearbyint(255.0 * share * mp));  // round half to even, like NumPy
  }
  return out;
}

// ---- grid memory -------------------------------------------------------------------------------------
int64_t GridMemory::key(const GridParams& p, const Mat4& T, int i, int j) const {
  const Vec3 c{(i + 0.5) * p.resolution_m - p.extent_m / 2, (j + 0.5) * p.resolution_m - p.extent_m / 2, 0.0};
  const Vec3 o = apply(T, c);
  const int64_t kx = static_cast<int64_t>(std::floor(o.x / r_)), ky = static_cast<int64_t>(std::floor(o.y / r_));
  return (ky << 32) ^ (kx & 0xffffffffLL);
}

int GridMemory::fill(Grid& g, const GridParams& p, const Mat4& T, double t) const {
  int n_filled = 0;
  for (int j = 0; j < g.n; ++j)
    for (int i = 0; i < g.n; ++i) {
      const size_t c = static_cast<size_t>(j) * g.n + i;
      if (g.source_flags[c] != 0) continue;
      const auto it = cells_.find(key(p, T, i, j));
      if (it == cells_.end()) continue;
      const double age = t - it->second.t_seen;
      if (!(age <= max_age_)) continue;
      const double decay = std::clamp(1.0 - age / max_age_, 0.0, 1.0);
      g.cost[c] = it->second.cost;
      g.semantic_class[c] = it->second.cls;
      g.confidence[c] = static_cast<uint8_t>(std::nearbyint(it->second.conf * decay));
      g.source_flags[c] = kRemembered;
      ++n_filled;
    }
  return n_filled;
}

void GridMemory::update(const Grid& g, const GridParams& p, const Mat4& T, double t) {
  for (int j = 0; j < g.n; ++j)
    for (int i = 0; i < g.n; ++i) {
      const size_t c = static_cast<size_t>(j) * g.n + i;
      if ((g.source_flags[c] & 3) == 0) continue;
      if (std::find(dynamic_.begin(), dynamic_.end(), g.semantic_class[c]) != dynamic_.end()) continue;
      cells_[key(p, T, i, j)] = {g.cost[c], g.confidence[c], g.semantic_class[c], t};
    }
}

void GridMemory::prune(double t) {
  for (auto it = cells_.begin(); it != cells_.end();)
    it = (t - it->second.t_seen > max_age_) ? cells_.erase(it) : std::next(it);
}

// ---- clustering --------------------------------------------------------------------------------------
std::vector<Detection> cluster(const std::vector<Vec3>& xyz, const std::vector<Dist>& probs,
                               const std::vector<uint8_t>& geometric, const GridParams& g,
                               const std::array<bool, kNumClasses>& obstacle_lut, int min_points) {
  const int n = g.n(), voidk = onto_index(0);
  struct P { int i, j, cls; float conf; size_t k; };
  std::vector<P> pts;
  std::vector<uint8_t> occ(static_cast<size_t>(n) * n, 0);
  for (size_t k = 0; k < xyz.size(); ++k) {
    int i, j;
    if (!cell_of(xyz[k], g, i, j)) continue;
    const int cls = argmax(probs[k]);
    const bool geo = geometric[static_cast<size_t>(j) * n + i] != 0;
    if (!(obstacle_lut[cls] || (cls == voidk && geo))) continue;
    pts.push_back({i, j, cls, *std::max_element(probs[k].begin(), probs[k].end()), k});
    occ[static_cast<size_t>(j) * n + i] = 1;
  }
  if (pts.empty()) return {};
  // One iteration of 4-neighbour dilation (scipy's default structure), then 8-connected labelling with
  // labels numbered in raster order of each component's first cell.
  std::vector<uint8_t> grown(occ);
  for (int r = 0; r < n; ++r)
    for (int c = 0; c < n; ++c)
      if (occ[r * n + c]) {
        if (r > 0) grown[(r - 1) * n + c] = 1;
        if (r < n - 1) grown[(r + 1) * n + c] = 1;
        if (c > 0) grown[r * n + c - 1] = 1;
        if (c < n - 1) grown[r * n + c + 1] = 1;
      }
  std::vector<int> label(static_cast<size_t>(n) * n, 0);
  int n_lab = 0;
  for (int r = 0; r < n; ++r)
    for (int c = 0; c < n; ++c) {
      if (!grown[r * n + c] || label[r * n + c]) continue;
      label[r * n + c] = ++n_lab;
      std::queue<std::pair<int, int>> q;
      q.push({r, c});
      while (!q.empty()) {
        auto [rr, cc] = q.front();
        q.pop();
        for (int dr = -1; dr <= 1; ++dr)
          for (int dc = -1; dc <= 1; ++dc) {
            const int r2 = rr + dr, c2 = cc + dc;
            if (r2 < 0 || c2 < 0 || r2 >= n || c2 >= n || !grown[r2 * n + c2] || label[r2 * n + c2]) continue;
            label[r2 * n + c2] = n_lab;
            q.push({r2, c2});
          }
      }
    }
  std::vector<std::vector<const P*>> members(n_lab + 1);
  for (const P& p : pts) members[label[static_cast<size_t>(p.j) * n + p.i]].push_back(&p);
  std::vector<Detection> out;
  for (int k = 1; k <= n_lab; ++k) {
    const auto& m = members[k];
    if (static_cast<int>(m.size()) < min_points) continue;
    std::array<int, kNumClasses> count{};
    Vec3 sum{}, lo{1e300, 1e300, 1e300}, hi{-1e300, -1e300, -1e300};
    for (const P* p : m) {
      ++count[p->cls];
      const Vec3& v = xyz[p->k];
      sum = {sum.x + v.x, sum.y + v.y, sum.z + v.z};
      lo = {std::min(lo.x, v.x), std::min(lo.y, v.y), std::min(lo.z, v.z)};
      hi = {std::max(hi.x, v.x), std::max(hi.y, v.y), std::max(hi.z, v.z)};
    }
    const int major = static_cast<int>(std::max_element(count.begin(), count.end()) - count.begin());
    double conf = 0;
    for (const P* p : m) if (p->cls == major) conf += p->conf;
    const double sz = static_cast<double>(m.size());
    Detection d;
    d.xyz = {sum.x / sz, sum.y / sz, sum.z / sz};
    d.dims = {hi.x - lo.x, hi.y - lo.y, hi.z - lo.z};
    d.semantic_class = kOntologyIds[major];
    d.class_confidence = conf / count[major] * count[major] / sz;
    d.n_points = static_cast<int>(m.size());
    out.push_back(d);
  }
  return out;
}

}  // namespace offroad
