// planner_consumer: a minimal downstream planner that consumes the perception contract and enforces its
// rules (docs/interface_contract.md):
//   1. cost 255 (UNKNOWN) is never free: it blocks the corridor like a lethal cell
//   2. obstacles count only above an existence_probability gate; COASTING tracks get a wider margin
//   3. MODE_SAFE_STOP (or stale / missing output) => zero speed; MODE_LIDAR_ONLY => speed cap
// Output: a speed command along +x in base_link (/planner/cmd_vel) and a CSV of every decision.
#include <mutex>

#include <geometry_msgs/msg/twist.hpp>
#include <perception_msgs/msg/obstacle_array.hpp>
#include <perception_msgs/msg/perception_health.hpp>
#include <perception_msgs/msg/traversability_grid.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>

#include "offroad_perception/ros_util.hpp"

namespace offroad {

using perception_msgs::msg::Obstacle;
using perception_msgs::msg::PerceptionHealth;

class PlannerConsumerNode : public rclcpp::Node {
 public:
  explicit PlannerConsumerNode(const rclcpp::NodeOptions& opts) : Node("planner_consumer", opts) {
    v_max_ = declare_parameter<double>("v_max", 2.0);
    lidar_only_factor_ = declare_parameter<double>("lidar_only_speed_factor", 0.5);
    half_width_ = declare_parameter<double>("corridor_half_width_m", 0.9);
    lookahead_ = declare_parameter<double>("lookahead_m", 15.0);
    stop_dist_ = declare_parameter<double>("stop_distance_m", 2.0);
    gain_ = declare_parameter<double>("speed_gain", 0.5);
    existence_gate_ = declare_parameter<double>("existence_gate", 0.5);
    max_age_ = declare_parameter<double>("max_output_age_s", 0.25);
    log_.open(declare_parameter<std::string>("decision_csv", ""),
              "stamp,age_at_consume_ms,mode,d_blocked_m,d_obstacle_m,speed_factor,cmd_v,reason");
    cmd_pub_ = create_publisher<geometry_msgs::msg::Twist>("/planner/cmd_vel", 10);
    health_sub_ = create_subscription<PerceptionHealth>("/perception/health", 10,
        [this](PerceptionHealth::ConstSharedPtr m) { std::lock_guard<std::mutex> lk(mu_); mode_ = m->mode; have_health_ = true; });
    obs_sub_ = create_subscription<perception_msgs::msg::ObstacleArray>("/perception/obstacles", 5,
        [this](perception_msgs::msg::ObstacleArray::ConstSharedPtr m) { std::lock_guard<std::mutex> lk(mu_); obstacles_ = m; });
    grid_sub_ = create_subscription<perception_msgs::msg::TraversabilityGrid>("/perception/traversability", 5,
        [this](perception_msgs::msg::TraversabilityGrid::ConstSharedPtr m) { on_grid(m); });
  }

 private:
  void on_grid(perception_msgs::msg::TraversabilityGrid::ConstSharedPtr g) {
    const double stamp = to_sec(g->header.stamp);
    const double age_ms = (now().seconds() - stamp) * 1e3;
    uint8_t mode;
    perception_msgs::msg::ObstacleArray::ConstSharedPtr obs;
    bool have_health;
    { std::lock_guard<std::mutex> lk(mu_); mode = mode_; obs = obstacles_; have_health = have_health_; }

    // Rule 1: walk the corridor; the first lethal or UNKNOWN cell blocks it, mid costs slow us down.
    double d_blocked = lookahead_, worst_cost = 0;
    const double res = g->resolution, ox = g->origin.position.x, oy = g->origin.position.y;
    for (uint32_t r = 0; r < g->height; ++r)
      for (uint32_t c = 0; c < g->width; ++c) {
        const double x = ox + (c + 0.5) * res, y = oy + (r + 0.5) * res;
        if (x < 0 || x > lookahead_ || std::abs(y) > half_width_) continue;
        const uint8_t cost = g->cost[static_cast<size_t>(r) * g->width + c];
        if (cost >= 254) d_blocked = std::min(d_blocked, x);  // 254 lethal, 255 UNKNOWN: both block
        else worst_cost = std::max(worst_cost, static_cast<double>(cost));
      }
    // Rule 2: obstacles above the existence gate, COASTING ones with doubled lateral margin.
    double d_obstacle = lookahead_;
    if (obs && obs->header.stamp == g->header.stamp) {
      for (const Obstacle& o : obs->obstacles) {
        if (o.existence_probability < existence_gate_) continue;
        const double margin = (o.track_state == Obstacle::STATE_COASTING ? 2.0 : 1.0) * (half_width_ + o.dimensions.y / 2);
        if (o.pose.position.x > 0 && std::abs(o.pose.position.y) < margin)
          d_obstacle = std::min(d_obstacle, o.pose.position.x - o.dimensions.x / 2);
      }
    }
    // Rule 3: health mode and output age.
    std::string reason = "ok";
    double factor = 1.0 - worst_cost / 254.0;
    if (!have_health) { factor = 0; reason = "no health"; }
    else if (mode == PerceptionHealth::MODE_SAFE_STOP) { factor = 0; reason = "SAFE_STOP"; }
    else if (age_ms > max_age_ * 1e3) { factor = 0; reason = "stale output"; }
    else if (mode == PerceptionHealth::MODE_LIDAR_ONLY) { factor *= lidar_only_factor_; reason = "LIDAR_ONLY cap"; }
    const double d_free = std::min(d_blocked, d_obstacle);
    const double v = std::clamp(gain_ * (d_free - stop_dist_), 0.0, v_max_) * factor;
    if (v == 0 && reason == "ok") reason = d_obstacle < d_blocked ? "obstacle ahead" : "blocked/unknown ahead";
    geometry_msgs::msg::Twist cmd;
    cmd.linear.x = v;
    cmd_pub_->publish(cmd);
    log_.row(std::to_string(stamp), age_ms, static_cast<int>(mode), d_blocked, d_obstacle, factor, v, reason);
  }

  double v_max_, lidar_only_factor_, half_width_, lookahead_, stop_dist_, gain_, existence_gate_, max_age_;
  uint8_t mode_{PerceptionHealth::MODE_SAFE_STOP};
  bool have_health_{false};
  perception_msgs::msg::ObstacleArray::ConstSharedPtr obstacles_;
  std::mutex mu_;
  CsvLog log_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_pub_;
  rclcpp::Subscription<PerceptionHealth>::SharedPtr health_sub_;
  rclcpp::Subscription<perception_msgs::msg::ObstacleArray>::SharedPtr obs_sub_;
  rclcpp::Subscription<perception_msgs::msg::TraversabilityGrid>::SharedPtr grid_sub_;
};

}  // namespace offroad

RCLCPP_COMPONENTS_REGISTER_NODE(offroad::PlannerConsumerNode)
