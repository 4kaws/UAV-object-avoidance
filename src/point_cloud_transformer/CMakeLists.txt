cmake_minimum_required(VERSION 3.5)
project(point_cloud_transformer)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_sensor_msgs REQUIRED)

# Include directories
include_directories(include)

# Add executable
add_executable(point_cloud_transformer_node src/point_cloud_transformer_node.cpp)
ament_target_dependencies(point_cloud_transformer_node rclcpp sensor_msgs tf2 tf2_ros tf2_sensor_msgs)

# Install executables
install(TARGETS
  point_cloud_transformer_node
  DESTINATION lib/${PROJECT_NAME})

ament_package()

