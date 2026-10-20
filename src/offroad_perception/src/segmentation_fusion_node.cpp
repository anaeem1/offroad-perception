// segmentation_fusion: camera + lidar TensorRT segmentation, camera frame matched to the lidar's sweep of
// the camera FOV, motion-compensated late fusion, deskew to base_link, ego filter.
// Publishes /perception/fused_points (PointCloud2 in base_link at the scan stamp; fields x y z conf label cam).
// All algorithms come from the parity-tested core (pipeline/*.py is the reference).
#include <algorithm>
#include <deque>
#include <memory>
#include <mutex>

#include <nav_msgs/msg/odometry.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include "offroad_perception/core.hpp"
#include "offroad_perception/projection.hpp"
#include "offroad_perception/ros_util.hpp"
#include "offroad_perception/trt_engine.hpp"

namespace offroad {

namespace {
Mat4 rot_x_deg(double deg) {
  const double r = deg * M_PI / 180.0, c = std::cos(r), s = std::sin(r);
  Mat4 R;
  R(1, 1) = c; R(1, 2) = -s; R(2, 1) = s; R(2, 2) = c;
  return R;
}
}  // namespace

class SegmentationFusionNode : public rclcpp::Node {
 public:
  explicit SegmentationFusionNode(const rclcpp::NodeOptions& opts) : Node("segmentation_fusion", opts) {
    lidar_ = std::make_unique<TrtEngine>(declare_parameter<std::string>("lidar_engine"));
    camera_ = std::make_unique<TrtEngine>(declare_parameter<std::string>("camera_engine"));
    alpha_ = declare_parameter<double>("fusion_alpha", 0.5);
    loglinear_ = declare_parameter<bool>("fusion_loglinear", true);
    window_ = declare_parameter<double>("camera_match_window_s", 0.05);
    base_frame_ = declare_parameter<std::string>("base_frame", "base_link");
    const auto q = declare_parameter<std::vector<double>>("lidar_to_camera_q_xyzw");
    const auto t = declare_parameter<std::vector<double>>("lidar_to_camera_t_xyz");
    const double pitch = declare_parameter<double>("pitch_correction_deg", -0.5);
    // Calibration file gives the camera pose in the lidar frame (T_lidar_cam); points need its inverse.
    T_cam_lidar_ = mul(rot_x_deg(pitch), inverse_rigid(from_qt({q[0], q[1], q[2], q[3]}, {t[0], t[1], t[2]})));
    const auto tto = declare_parameter<std::vector<int64_t>>("lidar_train_to_ontology",
        std::vector<int64_t>{0, 3, 4, 5, 6, 8, 15, 17, 18, 19, 23, 27, 31, 33, 34});
    const auto bucket = declare_parameter<std::vector<int64_t>>("lidar_void_bucket", std::vector<int64_t>{0, 1, 7, 9, 10, 12});
    const auto cids = declare_parameter<std::vector<int64_t>>("camera_class_ids",
        std::vector<int64_t>{1, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 17, 18, 19, 23, 27, 31, 33, 34});
    train_to_onto_.assign(tto.begin(), tto.end());
    bucket_.assign(bucket.begin(), bucket.end());
    cam_ids_.assign(cids.begin(), cids.end());
    const auto ego = declare_parameter<std::vector<double>>("ego_box", std::vector<double>{-0.9, 0.9, -0.8, 0.8, -0.5, 2.0});
    ego_ = {ego[0], ego[1], ego[2], ego[3], ego[4], ego[5]};
    lidar_log_.open(declare_parameter<std::string>("lidar_timing_csv", ""),
                    "stamp,receive_delay_ms,decode_ms,range_proj_ms,lidar_gpu_ms,lidar_infer_ms,match_ms,fuse_ms,"
                    "base_ms,publish_ms,total_ms,camera_used,cam_minus_wedge_ms,age_at_publish_ms");
    camera_log_.open(declare_parameter<std::string>("camera_timing_csv", ""),
                     "stamp,debayer_ms,resize_ms,camera_gpu_ms,camera_infer_ms,total_ms");

    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);
    cam_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    lidar_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    rclcpp::SubscriptionOptions cam_opts, lidar_opts;
    cam_opts.callback_group = cam_group_;
    lidar_opts.callback_group = lidar_group_;
    pub_ = create_publisher<sensor_msgs::msg::PointCloud2>("/perception/fused_points", 5);
    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
        "/odometry/filtered", 100, [this](nav_msgs::msg::Odometry::ConstSharedPtr m) { on_odom(m); }, lidar_opts);
    info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
        "/pylon_camera_node/camera_info", 5, [this](sensor_msgs::msg::CameraInfo::ConstSharedPtr m) { on_info(m); }, cam_opts);
    img_sub_ = create_subscription<sensor_msgs::msg::Image>(
        "/pylon_camera_node/image_raw", 5, [this](sensor_msgs::msg::Image::ConstSharedPtr m) { on_image(m); }, cam_opts);
    cloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        "/os1_cloud_node/points", 5, [this](sensor_msgs::msg::PointCloud2::ConstSharedPtr m) { on_cloud(m); }, lidar_opts);
    RCLCPP_INFO(get_logger(), "engines loaded: lidar in %ldx%ldx%ld, camera in %ldx%ldx%ld", lidar_->input_shape()[1],
                lidar_->input_shape()[2], lidar_->input_shape()[3], camera_->input_shape()[1],
                camera_->input_shape()[2], camera_->input_shape()[3]);
  }

 private:
  struct CamFrame { double stamp; std::vector<float> probs; };

  void on_odom(nav_msgs::msg::Odometry::ConstSharedPtr m) {
    std::lock_guard<std::mutex> lk(odom_mutex_);
    const auto& p = m->pose.pose;
    const double t = to_sec(m->header.stamp);
    odom_.add(t, {p.position.x, p.position.y, p.position.z}, {p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w});
    odom_.trim_before(t - 5.0);
  }

  void on_info(sensor_msgs::msg::CameraInfo::ConstSharedPtr m) {
    std::lock_guard<std::mutex> lk(cam_mutex_);
    cam_ = Camera{m->k[0], m->k[4], m->k[2], m->k[5], {m->d[0], m->d[1], m->d[2], m->d[3], m->d.size() > 4 ? m->d[4] : 0.0},
                  static_cast<int>(m->width), static_cast<int>(m->height)};
    have_info_ = true;
  }

  void on_image(sensor_msgs::msg::Image::ConstSharedPtr m) {
    const auto t0 = Clock::now();
    if (m->encoding != "bayer_rggb8") { RCLCPP_ERROR_ONCE(get_logger(), "unexpected encoding %s", m->encoding.c_str()); return; }
    const cv::Mat raw(static_cast<int>(m->height), static_cast<int>(m->width), CV_8UC1, const_cast<uint8_t*>(m->data.data()), m->step);
    cv::Mat rgb, small;
    cv::cvtColor(raw, rgb, cv::COLOR_BayerBG2RGB);  // ROS bayer_rggb8 == OpenCV BayerBG (measured, sensors.yaml)
    const double t_debayer = ms_since(t0);
    const auto& s = camera_->input_shape();  // 1, 3, H, W
    const int H = static_cast<int>(s[2]), W = static_cast<int>(s[3]);
    cv::resize(rgb, small, cv::Size(W, H), 0, 0, cv::INTER_LINEAR);
    std::vector<float> chw(3 * H * W);
    for (int y = 0; y < H; ++y) {
      const uint8_t* row = small.ptr<uint8_t>(y);
      for (int x = 0; x < W; ++x)
        for (int c = 0; c < 3; ++c) chw[(static_cast<size_t>(c) * H + y) * W + x] = row[3 * x + c];
    }
    const double t_resize = ms_since(t0) - t_debayer;
    const auto t1 = Clock::now();
    auto f = std::make_shared<CamFrame>();
    f->stamp = to_sec(m->header.stamp);
    std::vector<int32_t> labels;
    const float gpu = camera_->infer(chw.data(), f->probs, labels);
    const double t_infer = ms_since(t1);
    {
      std::lock_guard<std::mutex> lk(cam_mutex_);
      frames_.push_back(std::move(f));
      while (frames_.size() > 10) frames_.pop_front();
    }
    camera_log_.row(std::to_string(to_sec(m->header.stamp)), t_debayer, t_resize, gpu, t_infer, ms_since(t0));
  }

  void on_cloud(sensor_msgs::msg::PointCloud2::ConstSharedPtr m) {
    const auto t0 = Clock::now();
    const double stamp = to_sec(m->header.stamp);
    const double receive_delay = (now().seconds() - stamp) * 1e3;
    if (!have_base_) {
      try {
        const auto tf = tf_buffer_->lookupTransform(base_frame_, m->header.frame_id, tf2::TimePointZero);
        const auto& r = tf.transform.rotation;
        const auto& tr = tf.transform.translation;
        T_base_lidar_ = from_qt({r.x, r.y, r.z, r.w}, {tr.x, tr.y, tr.z});
        have_base_ = true;
      } catch (const std::exception& e) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "waiting for %s <- %s: %s", base_frame_.c_str(),
                             m->header.frame_id.c_str(), e.what());
        return;
      }
    }
    // Decode (field offsets from the message; remission = intensity / 65535, sensors.yaml).
    const size_t N = static_cast<size_t>(m->width) * m->height;
    const int ox = field_offset(*m, "x"), oi = field_offset(*m, "intensity"), ot = field_offset(*m, "t"),
              orng = field_offset(*m, "range");
    std::vector<float> xyz(3 * N), rem(N);
    std::vector<double> pt(N);
    std::vector<uint8_t> valid(N);
    for (size_t i = 0; i < N; ++i) {
      const uint8_t* p = &m->data[i * m->point_step];
      xyz[3 * i] = read_at<float>(p + ox);
      xyz[3 * i + 1] = read_at<float>(p + ox + 4);
      xyz[3 * i + 2] = read_at<float>(p + ox + 8);
      rem[i] = read_at<float>(p + oi) / 65535.0f;
      pt[i] = stamp + read_at<uint32_t>(p + ot) * 1e-9;
      valid[i] = read_at<uint32_t>(p + orng) > 0;
    }
    const double t_decode = ms_since(t0);
    // Lidar segmentation.
    auto t1 = Clock::now();
    const RangeImage ri = project_range(xyz.data(), rem.data(), N, RangeConfig{});
    const double t_proj = ms_since(t1);
    t1 = Clock::now();
    std::vector<float> lp;
    std::vector<int32_t> ll;
    const float lidar_gpu = lidar_->infer(ri.input.data(), lp, ll);
    const double t_lidar = ms_since(t1);
    const int K = static_cast<int>(lidar_->probs_shape()[1]), HW = 64 * 2048;
    std::vector<uint32_t> vidx;
    std::vector<Vec3> vpts;
    std::vector<double> vt;
    for (size_t i = 0; i < N; ++i)
      if (valid[i]) { vidx.push_back(static_cast<uint32_t>(i)); vpts.push_back({xyz[3 * i], xyz[3 * i + 1], xyz[3 * i + 2]}); vt.push_back(pt[i]); }
    std::vector<Dist> probs(vidx.size());
    std::vector<float> pl(K);
    for (size_t k = 0; k < vidx.size(); ++k) {
      const size_t pix = static_cast<size_t>(ri.py[vidx[k]]) * 2048 + ri.px[vidx[k]];
      for (int j = 0; j < K; ++j) pl[j] = lp[j * HW + pix];
      probs[k] = lidar_to_ontology(pl.data(), K, train_to_onto_, bucket_);
    }
    // Camera frame nearest to when the lidar swept the camera wedge.
    t1 = Clock::now();
    PoseTrack odom;
    { std::lock_guard<std::mutex> lk(odom_mutex_); odom = odom_; }
    bool cam_used = false;
    double cam_minus_wedge = std::nan("");
    std::vector<uint8_t> from_cam(vidx.size(), 0);
    Camera cam;
    bool have_info;
    std::vector<std::shared_ptr<const CamFrame>> frames;
    {
      std::lock_guard<std::mutex> lk(cam_mutex_);
      cam = cam_;
      have_info = have_info_;
      frames.assign(frames_.begin(), frames_.end());
    }
    double t_match = 0, t_fuse = 0;
    if (have_info && !frames.empty() && !odom.empty()) {
      std::vector<uint32_t> local(vidx.size());
      for (size_t k = 0; k < local.size(); ++k) local[k] = static_cast<uint32_t>(k);
      const auto hits0 = project_to_camera(vpts, local, cam, T_cam_lidar_);
      std::vector<double> th;
      for (const auto& h : hits0) th.push_back(vt[h.idx]);
      if (!th.empty()) {
        std::nth_element(th.begin(), th.begin() + th.size() / 2, th.end());
        const double t_wedge = th[th.size() / 2];
        const CamFrame* best = nullptr;
        for (const auto& f : frames)
          if (!best || std::abs(f->stamp - t_wedge) < std::abs(best->stamp - t_wedge)) best = f.get();
        cam_minus_wedge = (best->stamp - t_wedge) * 1e3;
        t_match = ms_since(t1);
        if (std::abs(best->stamp - t_wedge) <= window_) {
          t1 = Clock::now();
          const auto hits = project_to_camera(deskew(vpts, vt, odom, T_base_lidar_, best->stamp), local, cam, T_cam_lidar_);
          const int C = static_cast<int>(cam_ids_.size()), h4 = static_cast<int>(camera_->probs_shape()[2]),
                    w4 = static_cast<int>(camera_->probs_shape()[3]);
          std::vector<float> pc(C);
          for (const auto& h : hits) {
            const int u = std::clamp(static_cast<int>(h.u * w4 / cam.width), 0, w4 - 1);
            const int v = std::clamp(static_cast<int>(h.v * h4 / cam.height), 0, h4 - 1);
            for (int c = 0; c < C; ++c) pc[c] = best->probs[(static_cast<size_t>(c) * h4 + v) * w4 + u];
            probs[h.idx] = fuse(probs[h.idx], camera_to_ontology(pc.data(), cam_ids_), static_cast<float>(alpha_), loglinear_);
            from_cam[h.idx] = 1;
          }
          cam_used = true;
          t_fuse = ms_since(t1);
        }
      }
    }
    // Deskew to the scan stamp in base_link, drop the machine's own returns, publish.
    t1 = Clock::now();
    const auto lidar_at_stamp = odom.empty() ? vpts : deskew(vpts, vt, odom, T_base_lidar_, stamp);
    auto out = std::make_unique<sensor_msgs::msg::PointCloud2>();
    out->header.stamp = m->header.stamp;  // contract: measurement time, never publish time
    out->header.frame_id = base_frame_;
    out->height = 1;
    const char* names[] = {"x", "y", "z", "conf", "label", "cam"};
    const uint8_t types[] = {7, 7, 7, 7, 2, 2};  // FLOAT32 x4, UINT8 x2
    const uint32_t offs[] = {0, 4, 8, 12, 16, 17};
    for (int f = 0; f < 6; ++f) {
      sensor_msgs::msg::PointField pf;
      pf.name = names[f]; pf.offset = offs[f]; pf.datatype = types[f]; pf.count = 1;
      out->fields.push_back(pf);
    }
    out->point_step = 20;
    out->is_dense = true;
    out->data.reserve(vidx.size() * 20);
    size_t n_out = 0;
    for (size_t k = 0; k < vidx.size(); ++k) {
      const Vec3 b = apply(T_base_lidar_, lidar_at_stamp[k]);
      if (ego_.contains(b)) continue;
      const int cls = static_cast<int>(std::max_element(probs[k].begin(), probs[k].end()) - probs[k].begin());
      const float rec[4] = {static_cast<float>(b.x), static_cast<float>(b.y), static_cast<float>(b.z), probs[k][cls]};
      const uint8_t tail[4] = {static_cast<uint8_t>(kOntologyIds[cls]), from_cam[k], 0, 0};
      const auto* r8 = reinterpret_cast<const uint8_t*>(rec);
      out->data.insert(out->data.end(), r8, r8 + 16);
      out->data.insert(out->data.end(), tail, tail + 4);
      ++n_out;
    }
    out->width = static_cast<uint32_t>(n_out);
    out->row_step = out->point_step * out->width;
    const double t_base = ms_since(t1);
    t1 = Clock::now();
    pub_->publish(std::move(out));
    const double t_pub = ms_since(t1);
    lidar_log_.row(std::to_string(stamp), receive_delay, t_decode, t_proj, lidar_gpu, t_lidar, t_match, t_fuse, t_base, t_pub,
                   ms_since(t0), cam_used ? 1 : 0, cam_minus_wedge, (now().seconds() - stamp) * 1e3);
  }

  std::unique_ptr<TrtEngine> lidar_, camera_;
  double alpha_, window_;
  bool loglinear_;
  std::string base_frame_;
  Mat4 T_cam_lidar_, T_base_lidar_;
  bool have_base_{false}, have_info_{false};
  std::vector<int> train_to_onto_, bucket_, cam_ids_;
  EgoBox ego_;
  Camera cam_{};
  std::deque<std::shared_ptr<const CamFrame>> frames_;
  PoseTrack odom_;
  std::mutex cam_mutex_, odom_mutex_;
  CsvLog lidar_log_, camera_log_;
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
  rclcpp::CallbackGroup::SharedPtr cam_group_, lidar_group_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr info_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr img_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
};

}  // namespace offroad

RCLCPP_COMPONENTS_REGISTER_NODE(offroad::SegmentationFusionNode)
