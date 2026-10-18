// health_monitor: per-sensor status + degraded mode, published at 10 Hz on /perception/health.
// Modes (contract): NOMINAL both sensors OK; LIDAR_ONLY camera stale/degraded; SAFE_STOP lidar stale, output
// stale, or output older than max_output_age. This implementation has no camera-only geometry (the grid is
// built from lidar points, ADR-0002), so a lidar failure maps to SAFE_STOP rather than MODE_CAMERA_ONLY.
#include <mutex>
#include <optional>

#include <perception_msgs/msg/perception_health.hpp>
#include <perception_msgs/msg/traversability_grid.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include "offroad_perception/ros_util.hpp"

namespace offroad {

using perception_msgs::msg::PerceptionHealth;
using perception_msgs::msg::SensorStatus;

class HealthMonitorNode : public rclcpp::Node {
 public:
  explicit HealthMonitorNode(const rclcpp::NodeOptions& opts) : Node("health_monitor", opts) {
    timeout_ = declare_parameter<double>("sensor_timeout_s", 0.25);
    degraded_age_ = declare_parameter<double>("camera_degraded_age_s", 0.15);
    max_output_age_ = declare_parameter<double>("max_output_age_s", 0.25);  // ADR-0005 (proposed bound)
    min_valid_fraction_ = declare_parameter<double>("lidar_min_valid_fraction", 0.2);
    pub_ = create_publisher<PerceptionHealth>("/perception/health", 10);
    lidar_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>("/os1_cloud_node/points", rclcpp::SensorDataQoS(),
        [this](sensor_msgs::msg::PointCloud2::ConstSharedPtr m) {
          // Cheap density check: share of returns with nonzero range in a sample of points.
          size_t valid = 0, n = 0;
          int orng = -1;
          for (const auto& f : m->fields) if (f.name == "range") orng = static_cast<int>(f.offset);
          const size_t N = static_cast<size_t>(m->width) * m->height;
          for (size_t i = 0; orng >= 0 && i < N; i += 64, ++n) valid += read_at<uint32_t>(&m->data[i * m->point_step + orng]) > 0;
          std::lock_guard<std::mutex> lk(mu_);
          lidar_rx_ = now().seconds();
          lidar_valid_fraction_ = n ? static_cast<double>(valid) / n : 0.0;
        });
    cam_sub_ = create_subscription<sensor_msgs::msg::Image>("/pylon_camera_node/image_raw", rclcpp::SensorDataQoS(),
        [this](sensor_msgs::msg::Image::ConstSharedPtr) { std::lock_guard<std::mutex> lk(mu_); cam_rx_ = now().seconds(); });
    grid_sub_ = create_subscription<perception_msgs::msg::TraversabilityGrid>("/perception/traversability", 5,
        [this](perception_msgs::msg::TraversabilityGrid::ConstSharedPtr m) {
          std::lock_guard<std::mutex> lk(mu_);
          grid_rx_ = now().seconds();
          last_age_ms_ = (grid_rx_ - to_sec(m->header.stamp)) * 1e3;
          grid_stamp_ = m->header.stamp;
        });
    timer_ = create_wall_timer(std::chrono::milliseconds(100), [this] { tick(); });
  }

 private:
  SensorStatus status(const std::string& id, std::optional<double> rx, double now_s, double degraded_age, bool degraded,
                      const std::string& degraded_reason) {
    SensorStatus s;
    s.sensor_id = id;
    if (!rx) { s.status = SensorStatus::FAILED; s.data_age_ms = -1; s.reason = "no data received"; return s; }
    const double age = now_s - *rx;
    s.data_age_ms = static_cast<float>(age * 1e3);
    if (age > timeout_) { s.status = SensorStatus::FAILED; s.reason = "stale: no message for " + std::to_string(int(age * 1e3)) + " ms"; }
    else if (age > degraded_age || degraded) { s.status = SensorStatus::DEGRADED; s.reason = degraded ? degraded_reason : "late frame"; }
    else s.status = SensorStatus::OK;
    return s;
  }

  void tick() {
    std::lock_guard<std::mutex> lk(mu_);
    const double t = now().seconds();
    PerceptionHealth h;
    h.header.stamp = grid_stamp_;  // health refers to the latest published output
    h.header.frame_id = "base_link";
    const auto lidar = status("os1_lidar", lidar_rx_, t, timeout_, lidar_valid_fraction_ < min_valid_fraction_, "low point density");
    const auto cam = status("pylon_camera", cam_rx_, t, degraded_age_, false, "");
    h.sensors = {lidar, cam};
    const bool output_stale = !grid_rx_ || (t - *grid_rx_) > timeout_ || last_age_ms_ > max_output_age_ * 1e3;
    if (lidar.status == SensorStatus::FAILED || output_stale) h.mode = PerceptionHealth::MODE_SAFE_STOP;
    else if (cam.status != SensorStatus::OK || lidar.status != SensorStatus::OK) h.mode = PerceptionHealth::MODE_LIDAR_ONLY;
    else h.mode = PerceptionHealth::MODE_NOMINAL;
    h.end_to_end_latency_ms = static_cast<float>(last_age_ms_);
    pub_->publish(h);
  }

  double timeout_, degraded_age_, max_output_age_, min_valid_fraction_;
  std::optional<double> lidar_rx_, cam_rx_, grid_rx_;
  double lidar_valid_fraction_{1.0}, last_age_ms_{0};
  builtin_interfaces::msg::Time grid_stamp_;
  std::mutex mu_;
  rclcpp::Publisher<PerceptionHealth>::SharedPtr pub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr cam_sub_;
  rclcpp::Subscription<perception_msgs::msg::TraversabilityGrid>::SharedPtr grid_sub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace offroad

RCLCPP_COMPONENTS_REGISTER_NODE(offroad::HealthMonitorNode)
