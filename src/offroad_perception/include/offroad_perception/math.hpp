// Small fixed-size rigid-body math (no Eigen dependency, so the core builds the same on the server and
// the Jetson). Conventions match pipeline/geometry.py: T_a_b maps points from frame b into frame a.
#pragma once
#include <array>
#include <cmath>

namespace offroad {

struct Vec3 {
  double x{0}, y{0}, z{0};
};

struct Quat {  // x, y, z, w (ROS order)
  double x{0}, y{0}, z{0}, w{1};
};

struct Mat4 {
  std::array<double, 16> m{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};  // row-major
  double& operator()(int r, int c) { return m[r * 4 + c]; }
  double operator()(int r, int c) const { return m[r * 4 + c]; }
};

inline Mat4 mul(const Mat4& a, const Mat4& b) {
  Mat4 o;
  for (int r = 0; r < 4; ++r)
    for (int c = 0; c < 4; ++c) {
      double s = 0;
      for (int k = 0; k < 4; ++k) s += a(r, k) * b(k, c);
      o(r, c) = s;
    }
  return o;
}

inline Vec3 apply(const Mat4& T, const Vec3& p) {
  return {T(0, 0) * p.x + T(0, 1) * p.y + T(0, 2) * p.z + T(0, 3),
          T(1, 0) * p.x + T(1, 1) * p.y + T(1, 2) * p.z + T(1, 3),
          T(2, 0) * p.x + T(2, 1) * p.y + T(2, 2) * p.z + T(2, 3)};
}

inline Mat4 inverse_rigid(const Mat4& T) {
  Mat4 o;
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c) o(r, c) = T(c, r);
  for (int r = 0; r < 3; ++r) o(r, 3) = -(o(r, 0) * T(0, 3) + o(r, 1) * T(1, 3) + o(r, 2) * T(2, 3));
  return o;
}

inline Quat normalize(Quat q) {
  double n = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
  return {q.x / n, q.y / n, q.z / n, q.w / n};
}

inline Mat4 from_qt(Quat q, const Vec3& t) {
  q = normalize(q);
  Mat4 T;
  const double x = q.x, y = q.y, z = q.z, w = q.w;
  T(0, 0) = 1 - 2 * (y * y + z * z); T(0, 1) = 2 * (x * y - z * w);     T(0, 2) = 2 * (x * z + y * w);
  T(1, 0) = 2 * (x * y + z * w);     T(1, 1) = 1 - 2 * (x * x + z * z); T(1, 2) = 2 * (y * z - x * w);
  T(2, 0) = 2 * (x * z - y * w);     T(2, 1) = 2 * (y * z + x * w);     T(2, 2) = 1 - 2 * (x * x + y * y);
  T(0, 3) = t.x; T(1, 3) = t.y; T(2, 3) = t.z;
  return T;
}

// Spherical linear interpolation, shortest arc (matches scipy Slerp).
inline Quat slerp(Quat a, Quat b, double u) {
  a = normalize(a);
  b = normalize(b);
  double d = a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
  if (d < 0) { b = {-b.x, -b.y, -b.z, -b.w}; d = -d; }
  if (d > 0.9995) {
    return normalize({a.x + u * (b.x - a.x), a.y + u * (b.y - a.y), a.z + u * (b.z - a.z), a.w + u * (b.w - a.w)});
  }
  const double th = std::acos(d), s = std::sin(th);
  const double wa = std::sin((1 - u) * th) / s, wb = std::sin(u * th) / s;
  return {wa * a.x + wb * b.x, wa * a.y + wb * b.y, wa * a.z + wb * b.z, wa * a.w + wb * b.w};
}

}  // namespace offroad
