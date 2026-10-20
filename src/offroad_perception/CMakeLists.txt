cmake_minimum_required(VERSION 3.16)
project(offroad_perception CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Core algorithms: no ROS, no third-party deps. Builds on the server for parity tests against the Python
# reference, and on the Jetson inside the ROS 2 nodes. -ffp-contract=off keeps GCC on aarch64 from
# fusing multiply-adds, which would make float results drift from NumPy.
add_library(offroad_core STATIC src/projection.cpp src/core.cpp src/tracker.cpp)
target_include_directories(offroad_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)
target_compile_options(offroad_core PUBLIC -ffp-contract=off -Wall -Wextra)
find_package(OpenMP)  # parallel per-point loops; results are identical with or without it
if(OpenMP_CXX_FOUND)
  target_link_libraries(offroad_core PUBLIC OpenMP::OpenMP_CXX)
endif()
set_target_properties(offroad_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_executable(parity test/parity.cpp)
add_executable(bench test/bench.cpp)
target_link_libraries(bench offroad_core)
target_link_libraries(parity offroad_core)

# ROS 2 nodes are only built where ament is available (the Jetson).
find_package(ament_cmake QUIET)
if(ament_cmake_FOUND)
  include(cmake/ros_nodes.cmake OPTIONAL)
  install(TARGETS offroad_core parity ARCHIVE DESTINATION lib RUNTIME DESTINATION lib/${PROJECT_NAME})
  install(DIRECTORY include/ DESTINATION include)
  ament_package()
endif()
