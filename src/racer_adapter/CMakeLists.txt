cmake_minimum_required(VERSION 3.8)
project(racer_adapter)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(eigen3_cmake_module REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(swarm_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)

add_library(racer_transform_utils
  src/transform_utils.cpp
)
target_include_directories(racer_transform_utils
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(racer_transform_utils Eigen3::Eigen)
ament_target_dependencies(racer_transform_utils
  geometry_msgs
  sensor_msgs
)

add_executable(racer_frame_adapter
  src/racer_frame_adapter.cpp
)
target_link_libraries(racer_frame_adapter
  racer_transform_utils
  Eigen3::Eigen
)
ament_target_dependencies(racer_frame_adapter
  geometry_msgs
  nav_msgs
  rclcpp
  sensor_msgs
  std_msgs
  swarm_msgs
  tf2
  tf2_ros
)

install(
  TARGETS racer_transform_utils racer_frame_adapter
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_transform_utils test/test_transform_utils.cpp)
  target_link_libraries(test_transform_utils
    racer_transform_utils
    Eigen3::Eigen
  )
  ament_target_dependencies(test_transform_utils
    geometry_msgs
    sensor_msgs
  )
endif()

ament_export_include_directories(include)
ament_export_libraries(racer_transform_utils)
ament_package()
