# ROS 2 nodes (Jetson). Included from CMakeLists.txt only when ament_cmake is found.
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(perception_msgs REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc)
find_package(CUDAToolkit REQUIRED)
find_library(NVINFER_LIB nvinfer REQUIRED)
find_path(NVINFER_INCLUDE NvInfer.h PATHS /usr/include/aarch64-linux-gnu /usr/include REQUIRED)

add_library(offroad_nodes SHARED
  src/trt_engine.cpp
  src/segmentation_fusion_node.cpp
  src/bev_tracker_node.cpp
  src/health_monitor_node.cpp
  src/planner_consumer_node.cpp)
target_include_directories(offroad_nodes PRIVATE ${NVINFER_INCLUDE} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(offroad_nodes offroad_core ${OpenCV_LIBS} CUDA::cudart ${NVINFER_LIB})
ament_target_dependencies(offroad_nodes rclcpp rclcpp_components sensor_msgs nav_msgs geometry_msgs tf2_ros perception_msgs)

rclcpp_components_register_node(offroad_nodes PLUGIN "offroad::SegmentationFusionNode" EXECUTABLE segmentation_fusion)
rclcpp_components_register_node(offroad_nodes PLUGIN "offroad::BevTrackerNode" EXECUTABLE bev_tracker)
rclcpp_components_register_node(offroad_nodes PLUGIN "offroad::HealthMonitorNode" EXECUTABLE health_monitor)
rclcpp_components_register_node(offroad_nodes PLUGIN "offroad::PlannerConsumerNode" EXECUTABLE planner_consumer)

install(TARGETS offroad_nodes ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
install(DIRECTORY launch config DESTINATION share/${PROJECT_NAME})
