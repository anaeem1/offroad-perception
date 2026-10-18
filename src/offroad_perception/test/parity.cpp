// Stage-isolated parity against the Python reference: every stage is fed the Python output of the stage
// before it, so a mismatch points at one stage. Inputs come from `python -m pipeline.parity_dump`;
// outputs (cpp_*.npy) are compared by `python -m pipeline.parity_compare`.
//   parity <dump_dir>
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <string>

#include "offroad_perception/core.hpp"
#include "offroad_perception/npy.hpp"
#include "offroad_perception/projection.hpp"
#include "offroad_perception/tracker.hpp"

namespace fs = std::filesystem;
using namespace offroad;

static Mat4 mat4(const npy::Array<double>& a) {
  Mat4 T;
  std::copy(a.data.begin(), a.data.begin() + 16, T.m.begin());
  return T;
}

static std::vector<Vec3> vec3s(const npy::Array<double>& a) {
  std::vector<Vec3> v(a.shape[0]);
  for (size_t i = 0; i < v.size(); ++i) v[i] = {a.data[3 * i], a.data[3 * i + 1], a.data[3 * i + 2]};
  return v;
}

static void save_vec3s(const std::string& path, const std::vector<Vec3>& v) {
  std::vector<double> flat;
  for (const Vec3& p : v) flat.insert(flat.end(), {p.x, p.y, p.z});
  npy::save<double>(path, {v.size(), 3}, flat.data());
}

int main(int argc, char** argv) {
  if (argc != 2) { std::cerr << "usage: parity <dump_dir>\n"; return 2; }
  const fs::path root(argv[1]), common = root / "common";
  auto L = [&](const fs::path& d, const char* n) { return npy::load<double>((d / n).string()); };

  PoseTrack odom;
  {
    auto t = L(common, "odom_t.npy"), p = L(common, "odom_p.npy"), q = L(common, "odom_q.npy");
    for (size_t i = 0; i < t.size(); ++i)
      odom.add(t.data[i], {p.data[3 * i], p.data[3 * i + 1], p.data[3 * i + 2]},
               {q.data[4 * i], q.data[4 * i + 1], q.data[4 * i + 2], q.data[4 * i + 3]});
  }
  const Mat4 T_base_lidar = mat4(L(common, "T_base_lidar.npy")), T_cam_lidar = mat4(L(common, "T_cam_lidar.npy"));
  const auto c = L(common, "cam.npy").data;
  const Camera cam{c[0], c[1], c[2], c[3], {c[4], c[5], c[6], c[7], c[8]}, static_cast<int>(c[9]), static_cast<int>(c[10])};
  const auto train_to_onto_a = npy::load<int32_t>((common / "train_to_onto.npy").string());
  const auto bucket_a = npy::load<int32_t>((common / "bucket.npy").string());
  const auto cam_ids_a = npy::load<int32_t>((common / "cam_class_ids.npy").string());
  const std::vector<int> train_to_onto(train_to_onto_a.data.begin(), train_to_onto_a.data.end());
  const std::vector<int> bucket(bucket_a.data.begin(), bucket_a.data.end()), cam_ids(cam_ids_a.data.begin(), cam_ids_a.data.end());
  std::array<int, kNumClasses> cost_lut{};
  std::array<bool, kNumClasses> obstacle_lut{};
  {
    auto cl = npy::load<int32_t>((common / "cost_lut.npy").string());
    auto ol = npy::load<uint8_t>((common / "obstacle_lut.npy").string());
    for (int k = 0; k < kNumClasses; ++k) { cost_lut[k] = cl.data[k]; obstacle_lut[k] = ol.data[k] != 0; }
  }
  const auto fusion_cfg = L(common, "fusion.npy").data;
  const float alpha = static_cast<float>(fusion_cfg[0]);
  const bool loglinear = fusion_cfg[1] != 0;
  GridParams gp;
  gp.extent_m = fusion_cfg[2];
  gp.resolution_m = fusion_cfg[3];

  int n_scans = 0;
  for (const auto& e : fs::directory_iterator(root)) {
    if (e.path().filename().string().rfind("scan_", 0) != 0) continue;
    const fs::path d = e.path();
    ++n_scans;
    // 1. range projection
    const auto xyz = npy::load<float>((d / "in_xyz.npy").string());
    const auto rem = npy::load<float>((d / "in_rem.npy").string());
    const size_t N = rem.size();
    const RangeImage ri = project_range(xyz.data.data(), rem.data.data(), N, RangeConfig{});
    npy::save<float>((d / "cpp_range_input.npy").string(), {5, 64, 2048}, ri.input.data());

    // 2. deskew valid points to the camera time (lidar frame)
    const auto point_t = L(d, "in_point_t.npy").data;
    const auto valid = npy::load<uint8_t>((d / "in_valid.npy").string()).data;
    std::vector<Vec3> vpts;
    std::vector<double> vt;
    std::vector<uint32_t> vidx;
    for (size_t i = 0; i < N; ++i)
      if (valid[i]) {
        vpts.push_back({xyz.data[3 * i], xyz.data[3 * i + 1], xyz.data[3 * i + 2]});
        vt.push_back(point_t[i]);
        vidx.push_back(static_cast<uint32_t>(i));
      }
    const double t_cam = L(d, "in_t_cam.npy").data[0], stamp = L(d, "in_stamp.npy").data[0];
    save_vec3s((d / "cpp_deskew_cam.npy").string(), deskew(vpts, vt, odom, T_base_lidar, t_cam));

    // 3. camera projection of the Python-deskewed points
    const auto hits = project_to_camera(vec3s(L(d, "py_deskew_cam.npy")), vidx, cam, T_cam_lidar);
    std::vector<double> uv;
    std::vector<int64_t> hidx;
    for (const auto& h : hits) { uv.insert(uv.end(), {h.u, h.v}); hidx.push_back(h.idx); }
    npy::save<double>((d / "cpp_hits_uv.npy").string(), {hits.size(), 2}, uv.data());
    npy::save<int64_t>((d / "cpp_hits_idx.npy").string(), {hidx.size()}, hidx.data());

    // 4. fusion on the Python hits
    const auto plid = npy::load<float>((d / "in_plid.npy").string());
    const auto pcam = npy::load<float>((d / "in_pcam.npy").string());
    const auto py_uv = L(d, "py_hits_uv.npy").data;
    const auto py_idx = npy::load<int64_t>((d / "py_hits_idx.npy").string()).data;
    const int ntr = static_cast<int>(plid.shape[1]), C = static_cast<int>(pcam.shape[0]);
    const int h4 = static_cast<int>(pcam.shape[1]), w4 = static_cast<int>(pcam.shape[2]);
    std::vector<Dist> probs_all(N);
    for (size_t i = 0; i < N; ++i) probs_all[i] = lidar_to_ontology(&plid.data[i * ntr], ntr, train_to_onto, bucket);
    std::vector<uint8_t> from_cam_all(N, 0);
    std::vector<float> pc(C);
    for (size_t k = 0; k < py_idx.size(); ++k) {
      int u = static_cast<int>(py_uv[2 * k] * w4 / cam.width), v = static_cast<int>(py_uv[2 * k + 1] * h4 / cam.height);
      u = std::clamp(u, 0, w4 - 1);
      v = std::clamp(v, 0, h4 - 1);
      for (int ch = 0; ch < C; ++ch) pc[ch] = pcam.data[(static_cast<size_t>(ch) * h4 + v) * w4 + u];
      const size_t i = static_cast<size_t>(py_idx[k]);
      probs_all[i] = fuse(probs_all[i], camera_to_ontology(pc.data(), cam_ids), alpha, loglinear);
      from_cam_all[i] = 1;
    }
    std::vector<float> fused;
    for (uint32_t i : vidx) fused.insert(fused.end(), probs_all[i].begin(), probs_all[i].end());
    npy::save<float>((d / "cpp_probs.npy").string(), {vidx.size(), static_cast<size_t>(kNumClasses)}, fused.data());

    // 5. deskew to the scan stamp, into base_link, minus the machine's own returns
    std::vector<Vec3> base;
    const EgoBox ego;
    for (const Vec3& p : deskew(vpts, vt, odom, T_base_lidar, stamp)) {
      const Vec3 b = apply(T_base_lidar, p);
      if (!ego.contains(b)) base.push_back(b);
    }
    save_vec3s((d / "cpp_xyz_base.npy").string(), base);

    // 6. BEV grid on the Python points/probabilities
    const auto py_base = vec3s(L(d, "py_xyz_base.npy"));
    const auto py_probs_a = npy::load<float>((d / "py_probs_base.npy").string());
    std::vector<Dist> py_probs(py_base.size());
    for (size_t i = 0; i < py_probs.size(); ++i)
      std::copy_n(&py_probs_a.data[i * kNumClasses], kNumClasses, py_probs[i].begin());
    const auto py_from_cam = npy::load<uint8_t>((d / "py_from_cam.npy").string()).data;
    const Grid g = build_grid(py_base, py_probs, py_from_cam, gp, cost_lut);
    const size_t n = static_cast<size_t>(g.n);
    npy::save<uint8_t>((d / "cpp_grid_cost.npy").string(), {n, n}, g.cost.data());
    npy::save<uint8_t>((d / "cpp_grid_conf.npy").string(), {n, n}, g.confidence.data());
    npy::save<uint8_t>((d / "cpp_grid_class.npy").string(), {n, n}, g.semantic_class.data());
    npy::save<uint8_t>((d / "cpp_grid_flags.npy").string(), {n, n}, g.source_flags.data());
    npy::save<uint8_t>((d / "cpp_grid_geometric.npy").string(), {n, n}, g.geometric.data());

    // 7. clustering on the Python grid's geometric cells
    const auto py_geo = npy::load<uint8_t>((d / "py_grid_geometric.npy").string()).data;
    std::vector<double> dets;
    for (const Detection& det : cluster(py_base, py_probs, py_geo, gp, obstacle_lut))
      dets.insert(dets.end(), {det.xyz.x, det.xyz.y, det.xyz.z, det.dims.x, det.dims.y, det.dims.z,
                               static_cast<double>(det.semantic_class), det.class_confidence, static_cast<double>(det.n_points)});
    npy::save<double>((d / "cpp_dets.npy").string(), {dets.size() / 9, 9}, dets.data());
  }

  // 8. tracker over the Python detection sequence
  if (fs::exists(root / "tracker")) {
    const auto seq = L(root / "tracker", "det_seq.npy");  // rows: step, t, x, y, z, dx, dy, dz, cls, conf
    Tracker tracker;
    std::vector<double> out;  // rows: step, id, state, x, y, vx, vy, existence, cls
    size_t r = 0;
    const size_t rows = seq.shape[0];
    const int n_steps = rows ? static_cast<int>(seq.data[(rows - 1) * 10]) + 1 : 0;
    const auto steps = L(root / "tracker", "step_t.npy").data;
    for (int s = 0; s < static_cast<int>(steps.size()); ++s) {
      std::vector<TrackInput> in;
      while (r < rows && static_cast<int>(seq.data[r * 10]) == s) {
        const double* q = &seq.data[r * 10];
        in.push_back({{q[2], q[3], q[4]}, {q[5], q[6], q[7]}, static_cast<int>(q[8]), q[9]});
        ++r;
      }
      for (const Track& t : tracker.step(steps[s], in))
        out.insert(out.end(), {static_cast<double>(s), static_cast<double>(t.id), static_cast<double>(t.state), t.x[0], t.x[1],
                               t.x[2], t.x[3], t.existence, static_cast<double>(t.semantic_class())});
    }
    (void)n_steps;
    npy::save<double>((root / "tracker" / "cpp_tracks.npy").string(), {out.size() / 9, 9}, out.data());
  }
  std::cout << "parity: processed " << n_scans << " scans\n";
  return 0;
}
