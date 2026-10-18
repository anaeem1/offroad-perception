// Small ROS helpers shared by the nodes: time conversion, PointCloud2 field access, CSV timing logs.
#pragma once
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <string>

#include <builtin_interfaces/msg/time.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

namespace offroad {

inline double to_sec(const builtin_interfaces::msg::Time& t) { return t.sec + t.nanosec * 1e-9; }

inline builtin_interfaces::msg::Time to_msg(double s) {
  builtin_interfaces::msg::Time t;
  t.sec = static_cast<int32_t>(std::floor(s));
  t.nanosec = static_cast<uint32_t>(std::llround((s - t.sec) * 1e9) % 1000000000LL);
  return t;
}

inline int field_offset(const sensor_msgs::msg::PointCloud2& m, const std::string& name) {
  for (const auto& f : m.fields)
    if (f.name == name) return static_cast<int>(f.offset);
  throw std::runtime_error("PointCloud2 has no field '" + name + "'");
}

template <typename T>
inline T read_at(const uint8_t* p) {
  T v;
  std::memcpy(&v, p, sizeof(T));
  return v;
}

using Clock = std::chrono::steady_clock;
inline double ms_since(Clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// Append-only CSV for latency measurements (the README latency table is built from these files).
class CsvLog {
 public:
  CsvLog() = default;
  void open(const std::string& path, const std::string& header) {
    if (path.empty()) return;
    f_.open(path, std::ios::out | std::ios::trunc);
    f_ << header << "\n";
  }
  template <typename... Args>
  void row(Args... args) {
    if (!f_.is_open()) return;
    std::lock_guard<std::mutex> lk(m_);
    bool first = true;
    ((f_ << (first ? "" : ",") << args, first = false), ...);
    f_ << "\n";
    f_.flush();
  }

 private:
  std::ofstream f_;
  std::mutex m_;
};

}  // namespace offroad
