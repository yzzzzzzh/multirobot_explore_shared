#pragma once

#include <string>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/transform.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

namespace racer_adapter
{

struct RigidTransform
{
  Eigen::Quaterniond rotation{Eigen::Quaterniond::Identity()};
  Eigen::Vector3d translation{Eigen::Vector3d::Zero()};
};

RigidTransform fromMsg(const geometry_msgs::msg::Transform & transform);

RigidTransform composeTransforms(
  const RigidTransform & target_from_middle,
  const RigidTransform & middle_from_source);

RigidTransform invertTransform(const RigidTransform & target_from_source);

geometry_msgs::msg::Pose transformPose(
  const RigidTransform & transform,
  const geometry_msgs::msg::Pose & pose);

Eigen::Vector3d transformVector(
  const RigidTransform & transform,
  const Eigen::Vector3d & vector);

double rotationDistance(
  const Eigen::Quaterniond & first,
  const Eigen::Quaterniond & second);

bool transformPointCloudXYZ(
  const sensor_msgs::msg::PointCloud2 & input,
  sensor_msgs::msg::PointCloud2 & output,
  const RigidTransform & transform,
  std::string & error);

bool filterPointCloudToPlanarSlice(
  const sensor_msgs::msg::PointCloud2 & input,
  sensor_msgs::msg::PointCloud2 & output,
  double plane_z,
  double half_thickness,
  std::string & error);

}  // namespace racer_adapter
