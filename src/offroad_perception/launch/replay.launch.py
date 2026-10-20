"""Replay a converted RELLIS-3D bag through the perception stack on the Jetson.

  ros2 launch offroad_perception replay.launch.py bag:=<dir> seq:=00001 engines:=<dir> logs:=<dir>

All four nodes run as components in one multi-threaded container (intra-process, and the segmentation
node's camera and lidar callback groups can run in parallel). Every node writes per-stage timing CSVs
into `logs`; the README latency table is computed from them.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    share = get_package_share_directory("offroad_perception")
    bag, seq, engines, logs, rate = (LaunchConfiguration(n) for n in ("bag", "seq", "engines", "logs", "rate"))
    params = PathJoinSubstitution([share, "config", PythonExpression(["'perception_' + '", seq, "' + '.yaml'"])])
    common = {"use_sim_time": True}

    def node(plugin, name, extra):
        return ComposableNode(package="offroad_perception", plugin=plugin, name=name,
                              parameters=[params, common, extra], extra_arguments=[{"use_intra_process_comms": True}])

    container = ComposableNodeContainer(
        name="perception", namespace="", package="rclcpp_components", executable="component_container_mt",
        output="screen",
        composable_node_descriptions=[
            node("offroad::SegmentationFusionNode", "segmentation_fusion", {
                "lidar_engine": PathJoinSubstitution([engines, LaunchConfiguration("lidar_engine_file")]),
                "camera_engine": PathJoinSubstitution([engines, LaunchConfiguration("camera_engine_file")]),
                "lidar_timing_csv": PathJoinSubstitution([logs, "segmentation_lidar.csv"]),
                "camera_timing_csv": PathJoinSubstitution([logs, "segmentation_camera.csv"])}),
            node("offroad::BevTrackerNode", "bev_tracker", {"timing_csv": PathJoinSubstitution([logs, "bev_tracker.csv"])}),
            node("offroad::HealthMonitorNode", "health_monitor", {}),
            node("offroad::PlannerConsumerNode", "planner_consumer",
                 {"decision_csv": PathJoinSubstitution([logs, "planner.csv"])}),
        ])
    play = ExecuteProcess(cmd=["ros2", "bag", "play", bag, "--clock", "--rate", rate, "--qos-profile-overrides-path",
                               os.path.join(share, "config", "qos_overrides.yaml")], output="screen")
    return LaunchDescription([
        DeclareLaunchArgument("bag"), DeclareLaunchArgument("seq", default_value="00001"),
        DeclareLaunchArgument("engines"), DeclareLaunchArgument("logs"),
        DeclareLaunchArgument("rate", default_value="1.0"),
        # Deployment configuration (ADR-0010): FP16 lidar (INT8 failed its accuracy check), 480x768 camera.
        DeclareLaunchArgument("lidar_engine_file", default_value="salsanext_fp16.engine"),
        DeclareLaunchArgument("camera_engine_file", default_value="segformer480_fp16.engine"),
        container,
        TimerAction(period=8.0, actions=[play]),  # let engines load and subscriptions come up first
    ])
