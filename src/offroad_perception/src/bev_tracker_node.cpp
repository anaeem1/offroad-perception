// bev_tracker: fused points -> BEV traversability grid + clustering + tracker.
// Publishes the contract: /perception/traversability (TraversabilityGrid), /perception/obstacles (ObstacleArray).
// Both carry the source scan's stamp in base_link. Tracking runs in the odom frame (static stays static).
#include <mutex>

#include <nav_msgs/msg/odometry.hpp>
#include <perception_msgs/msg/obstacle_array.hpp>
#include <perception_msgs/msg/traversability_grid.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include "offroad_perception/core.hpp"
#include "offroad_perception/ros_util.hpp"
#include "offroad_perception/tracker.hpp"

namespace offroad {

class BevTrackerNode : public rclcpp::Node {
 public:
  explicit BevTrackerNode(const rclcpp::NodeOptions& opts) : Node("bev_tracker", opts) {
    gp_.extent_m = declare_parameter<double>("grid_extent_m", 40.0);
    gp_.resolution_m = declare_parameter<double>("grid_resolution_m", 0.2);
    gp_.obstacle_height_m = declare_parameter<double>("obstacle_height_m", 0.3);
    memory_ = std::make_unique<GridMemory>(gp_.resolution_m, declare_parameter<double>("memory_max_age_s", 5.0));
    // Cost table in ontology-id order (kOntologyIds), generated from config/class_costs.yaml.
    const auto cost = declare_parameter<std::vector<int64_t>>("cost_lut");
    const auto obst = declare_parameter<std::vector<bool>>("obstacle_lut");
    if (cost.size() != kNumClasses || obst.size() != kNumClasses) throw std::runtime_error("cost/obstacle lut must have 20 entries");
    for (int k = 0; k < kNumClasses; ++k) { cost_lut_[k] = static_cast<int>(cost[k]); obstacle_lut_[k] = obst[k]; }
    timing_.open(declare_parameter<std::string>("timing_csv", ""),
                 "stamp,receive_delay_ms,grid_cluster_ms,memory_ms,track_ms,publish_ms,total_ms,n_points,n_dets,n_tracks,"
                 "remembered_cells,age_at_publish_ms");
    grid_pub_ = create_publisher<perception_msgs::msg::TraversabilityGrid>("/perception/traversability", 5);
    obs_pub_ = create_publisher<perception_msgs::msg::ObstacleArray>("/perception/obstacles", 5);
    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>("/odometry/filtered", 100,
        [this](nav_msgs::msg::Odometry::ConstSharedPtr m) {
          std::lock_guard<std::mutex> lk(mutex_);
          const auto& p = m->pose.pose;
          const double t = to_sec(m->header.stamp);
          odom_.add(t, {p.position.x, p.position.y, p.position.z}, {p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w});
          odom_.trim_before(t - 5.0);
        });
    pts_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>("/perception/fused_points", 1,  // newest wins (ADR-0011)
        [this](sensor_msgs::msg::PointCloud2::ConstSharedPtr m) { on_points(m); });
  }

 private:
  void on_points(sensor_msgs::msg::PointCloud2::ConstSharedPtr m) {
    const auto t0 = Clock::now();
    const double stamp = to_sec(m->header.stamp);
    const double receive_delay = (now().seconds() - stamp) * 1e3;
    const size_t N = static_cast<size_t>(m->width) * m->height;
    const int ox = field_offset(*m, "x"), oc = field_offset(*m, "conf"), ol = field_offset(*m, "label"), ocam = field_offset(*m, "cam");
    std::vector<Vec3> pts(N);
    std::vector<Dist> probs(N, Dist{});
    std::vector<uint8_t> cam(N);
    for (size_t i = 0; i < N; ++i) {
      const uint8_t* p = &m->data[i * m->point_step];
      pts[i] = {read_at<float>(p + ox), read_at<float>(p + ox + 4), read_at<float>(p + ox + 8)};
      // label + its probability carry everything the grid and clustering use (argmax and max prob).
      probs[i][onto_index(p[ol])] = read_at<float>(p + oc);
      cam[i] = p[ocam];
    }
    PoseTrack odom;
    { std::lock_guard<std::mutex> lk(mutex_); odom = odom_; }
    if (odom.empty()) { RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "no odometry yet"); return; }
    const Mat4 T_ob = odom.at(stamp), T_bo = inverse_rigid(T_ob);
    auto t1 = Clock::now();
    Grid g = build_grid(pts, probs, cam, gp_, cost_lut_);
    const auto dets = cluster(pts, probs, g.geometric, gp_, obstacle_lut_);  // current sweep only
    const double t_grid = ms_since(t1);
    t1 = Clock::now();
    const int n_remembered = memory_->fill(g, gp_, T_ob, stamp);  // ADR-0008: blind cells from earlier sweeps
    memory_->update(g, gp_, T_ob, stamp);
    if (++frames_ % 50 == 0) memory_->prune(stamp);
    const double t_memory = ms_since(t1);
    t1 = Clock::now();
    std::vector<TrackInput> in;
    for (const auto& d : dets) in.push_back({apply(T_ob, d.xyz), d.dims, d.semantic_class, d.class_confidence});
    const auto& tracks = tracker_.step(stamp, in);
    const double t_track = ms_since(t1);

    t1 = Clock::now();
    perception_msgs::msg::TraversabilityGrid grid;
    grid.header = m->header;  // stamp = source lidar scan time, frame = base_link
    grid.resolution = static_cast<float>(gp_.resolution_m);
    grid.width = grid.height = static_cast<uint32_t>(g.n);
    grid.origin.position.x = grid.origin.position.y = -gp_.extent_m / 2;
    grid.origin.orientation.w = 1.0;
    grid.cost = g.cost;
    grid.confidence = g.confidence;
    grid.semantic_class = g.semantic_class;
    uint32_t flags = 0;  // 1 lidar, 2 camera, 4 remembered (ADR-0008)
    for (uint8_t f : g.source_flags) flags |= f;
    grid.source_flags = flags;
    grid_pub_->publish(grid);

    perception_msgs::msg::ObstacleArray arr;
    arr.header = m->header;
    for (const auto& tr : tracks) {
      perception_msgs::msg::Obstacle o;
      o.track_id = tr.id;
      o.semantic_class = static_cast<uint8_t>(tr.semantic_class());
      o.class_confidence = static_cast<float>(tr.class_conf);
      o.existence_probability = static_cast<float>(tr.existence);
      const Vec3 pb = apply(T_bo, {tr.x[0], tr.x[1], tr.z});
      o.pose.position.x = pb.x; o.pose.position.y = pb.y; o.pose.position.z = pb.z;
      o.pose.orientation.w = 1.0;
      o.dimensions.x = tr.dims.x; o.dimensions.y = tr.dims.y; o.dimensions.z = tr.dims.z;
      // Velocity: rotate the odom-frame estimate into base_link.
      o.velocity.x = T_bo(0, 0) * tr.x[2] + T_bo(0, 1) * tr.x[3];
      o.velocity.y = T_bo(1, 0) * tr.x[2] + T_bo(1, 1) * tr.x[3];
      o.pose_covariance.fill(0.0);
      o.pose_covariance[0] = tr.P[0]; o.pose_covariance[1] = tr.P[1];
      o.pose_covariance[6] = tr.P[4]; o.pose_covariance[7] = tr.P[5];
      o.pose_covariance[14] = 0.25;  // z: not estimated
      o.pose_covariance[21] = o.pose_covariance[28] = o.pose_covariance[35] = 1e3;  // orientation: not estimated
      o.age_frames = tr.age;
      o.track_state = static_cast<uint8_t>(tr.state);
      arr.obstacles.push_back(o);
    }
    obs_pub_->publish(arr);
    const double t_pub = ms_since(t1);
    timing_.row(std::to_string(stamp), receive_delay, t_grid, t_memory, t_track, t_pub, ms_since(t0), N, dets.size(),
                tracks.size(), n_remembered, (now().seconds() - stamp) * 1e3);
  }

  GridParams gp_;
  std::array<int, kNumClasses> cost_lut_{};
  std::array<bool, kNumClasses> obstacle_lut_{};
  Tracker tracker_;
  std::unique_ptr<GridMemory> memory_;
  uint64_t frames_{0};
  PoseTrack odom_;
  std::mutex mutex_;
  CsvLog timing_;
  rclcpp::Publisher<perception_msgs::msg::TraversabilityGrid>::SharedPtr grid_pub_;
  rclcpp::Publisher<perception_msgs::msg::ObstacleArray>::SharedPtr obs_pub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pts_sub_;
};

}  // namespace offroad

RCLCPP_COMPONENTS_REGISTER_NODE(offroad::BevTrackerNode)
