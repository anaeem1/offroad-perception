// Micro-benchmark of the CPU stages on a real scan from a parity dump (same inputs the runtime sees).
//   bench <dump_dir>/scan_XXXXXX <dump_dir>/common [repeats]
#include <malloc.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>

#include "offroad_perception/core.hpp"
#include "offroad_perception/npy.hpp"
#include "offroad_perception/projection.hpp"

using namespace offroad;
using Clk = std::chrono::steady_clock;

template <typename F>
static double time_ms(F&& f, int reps) {
  std::vector<double> t;
  for (int r = 0; r < reps; ++r) {
    const auto t0 = Clk::now();
    f();
    t.push_back(std::chrono::duration<double, std::milli>(Clk::now() - t0).count());
  }
  std::sort(t.begin(), t.end());
  return t[t.size() / 2];
}

int main(int argc, char** argv) {
  if (argc < 3) { std::cerr << "usage: bench <scan_dir> <common_dir> [repeats]\n"; return 2; }
  const std::string d = argv[1], c = argv[2];
  const int reps = argc > 3 ? std::stoi(argv[3]) : 20;
  if (std::getenv("BENCH_NO_MMAP")) mallopt(M_MMAP_THRESHOLD, 1 << 30);  // probe: reuse heap memory instead of fresh pages
  const auto xyz = npy::load<float>(d + "/in_xyz.npy");
  const auto rem = npy::load<float>(d + "/in_rem.npy");
  const auto pt = npy::load<double>(d + "/in_point_t.npy").data;
  const auto valid = npy::load<uint8_t>(d + "/in_valid.npy").data;
  const double t_cam = npy::load<double>(d + "/in_t_cam.npy").data[0], stamp = npy::load<double>(d + "/in_stamp.npy").data[0];
  PoseTrack odom;
  {
    auto t = npy::load<double>(c + "/odom_t.npy"), p = npy::load<double>(c + "/odom_p.npy"), q = npy::load<double>(c + "/odom_q.npy");
    for (size_t i = 0; i < t.size(); ++i)
      odom.add(t.data[i], {p.data[3 * i], p.data[3 * i + 1], p.data[3 * i + 2]},
               {q.data[4 * i], q.data[4 * i + 1], q.data[4 * i + 2], q.data[4 * i + 3]});
  }
  Mat4 T_bl, T_cl;
  { auto a = npy::load<double>(c + "/T_base_lidar.npy"); std::copy_n(a.data.begin(), 16, T_bl.m.begin()); }
  { auto a = npy::load<double>(c + "/T_cam_lidar.npy"); std::copy_n(a.data.begin(), 16, T_cl.m.begin()); }
  const auto cc = npy::load<double>(c + "/cam.npy").data;
  const Camera cam{cc[0], cc[1], cc[2], cc[3], {cc[4], cc[5], cc[6], cc[7], cc[8]}, static_cast<int>(cc[9]), static_cast<int>(cc[10])};
  const size_t N = rem.size();
  std::vector<Vec3> vpts;
  std::vector<double> vt;
  std::vector<uint32_t> ids;
  for (size_t i = 0; i < N; ++i)
    if (valid[i]) { vpts.push_back({xyz.data[3 * i], xyz.data[3 * i + 1], xyz.data[3 * i + 2]}); vt.push_back(pt[i]); ids.push_back(ids.size()); }
  const auto base = npy::load<double>(d + "/py_xyz_base.npy");
  std::vector<Vec3> pbase(base.shape[0]);
  for (size_t i = 0; i < pbase.size(); ++i) pbase[i] = {base.data[3 * i], base.data[3 * i + 1], base.data[3 * i + 2]};
  const auto pr = npy::load<float>(d + "/py_probs_base.npy");
  std::vector<Dist> probs(pbase.size());
  for (size_t i = 0; i < probs.size(); ++i) std::copy_n(&pr.data[i * kNumClasses], kNumClasses, probs[i].begin());
  const auto fc = npy::load<uint8_t>(d + "/py_from_cam.npy").data;
  std::array<int, kNumClasses> cost{};
  std::array<bool, kNumClasses> obst{};
  { auto a = npy::load<int32_t>(c + "/cost_lut.npy"); auto b = npy::load<uint8_t>(c + "/obstacle_lut.npy");
    for (int k = 0; k < kNumClasses; ++k) { cost[k] = a.data[k]; obst[k] = b.data[k]; } }
  GridParams gp;
  Grid g;

  std::cout << "points " << N << ", valid " << vpts.size() << ", repeats " << reps << " (median ms)\n";
  std::cout << "  range_projection      " << time_ms([&] { project_range(xyz.data.data(), rem.data.data(), N, RangeConfig{}); }, reps) << "\n";
  std::cout << "  camera_proj_all_valid " << time_ms([&] { project_to_camera(vpts, ids, cam, T_cl); }, reps) << "\n";
  std::cout << "  deskew_valid          " << time_ms([&] { deskew(vpts, vt, odom, T_bl, t_cam); }, reps) << "\n";
  std::cout << "  build_grid            " << time_ms([&] { g = build_grid(pbase, probs, fc, gp, cost); }, reps) << "\n";
  std::cout << "  cluster               " << time_ms([&] { cluster(pbase, probs, g.geometric, gp, obst); }, reps) << "\n";
  GridMemory mem(gp.resolution_m, 5.0);
  const Mat4 T_ob = odom.at(stamp);
  std::cout << "  memory_fill_update    " << time_ms([&] { Grid h = g; mem.fill(h, gp, T_ob, stamp); mem.update(h, gp, T_ob, stamp); }, reps) << "\n";
  return 0;
}
