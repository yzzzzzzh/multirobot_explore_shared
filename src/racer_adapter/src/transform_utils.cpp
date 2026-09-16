#include "racer_adapter/transform_utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

#include <sensor_msgs/msg/point_field.hpp>

namespace racer_adapter
{

RigidTransform fromMsg(const geometry_msgs::msg::Transform & transform)
{
  RigidTransform result;
  result.translation = Eigen::Vector3d(
    transform.translation.x,
    transform.translation.y,
    transform.translation.z);
  result.rotation = Eigen::Quaterniond(
    transform.rotation.w,
    transform.rotation.x,
    transform.rotation.y,
    transform.rotation.z);
  if (result.rotation.norm() < 1e-12) {
    result.rotation = Eigen::Quaterniond::Identity();
  } else {
    result.rotation.normalize();
  }
  return result;
}

RigidTransform composeTransforms(
  const RigidTransform & target_from_middle,
  const RigidTransform & middle_from_source)
{
  RigidTransform result;
  result.rotation =
    target_from_middle.rotation.normalized() *
    middle_from_source.rotation.normalized();
  result.rotation.normalize();
  result.translation =
    target_from_middle.rotation.normalized() * middle_from_source.translation +
    target_from_middle.translation;
  return result;
}

RigidTransform invertTransform(const RigidTransform & target_from_source)
{
  RigidTransform result;
  result.rotation = target_from_source.rotation.normalized().conjugate();
  result.translation =
    -(result.rotation * target_from_source.translation);
  return result;
}

geometry_msgs::msg::Pose transformPose(
  const RigidTransform & transform,
  const geometry_msgs::msg::Pose & pose)
{
  const Eigen::Vector3d source_position(
    pose.position.x,
    pose.position.y,
    pose.position.z);
  Eigen::Quaterniond source_rotation(
    pose.orientation.w,
    pose.orientation.x,
    pose.orientation.y,
    pose.orientation.z);
  if (source_rotation.norm() < 1e-12) {
    source_rotation = Eigen::Quaterniond::Identity();
  } else {
    source_rotation.normalize();
  }

  const Eigen::Vector3d target_position =
    transform.rotation * source_position + transform.translation;
  Eigen::Quaterniond target_rotation = transform.rotation * source_rotation;
  target_rotation.normalize();

  geometry_msgs::msg::Pose result;
  result.position.x = target_position.x();
  result.position.y = target_position.y();
  result.position.z = target_position.z();
  result.orientation.x = target_rotation.x();
  result.orientation.y = target_rotation.y();
  result.orientation.z = target_rotation.z();
  result.orientation.w = target_rotation.w();
  return result;
}

Eigen::Vector3d transformVector(
  const RigidTransform & transform,
  const Eigen::Vector3d & vector)
{
  return transform.rotation * vector;
}

double rotationDistance(
  const Eigen::Quaterniond & first,
  const Eigen::Quaterniond & second)
{
  Eigen::Quaterniond delta = first.normalized().conjugate() * second.normalized();
  delta.normalize();
  const double absolute_w = std::clamp(std::abs(delta.w()), 0.0, 1.0);
  return 2.0 * std::acos(absolute_w);
}

namespace
{

bool findFloat32Offset(
  const sensor_msgs::msg::PointCloud2 & cloud,
  const std::string & name,
  std::uint32_t & offset)
{
  for (const auto & field : cloud.fields) {
    if (field.name == name &&
      field.datatype == sensor_msgs::msg::PointField::FLOAT32 &&
      field.count == 1)
    {
      offset = field.offset;
      return true;
    }
  }
  return false;
}

float loadFloat(const std::uint8_t * data)
{
  float value;
  std::memcpy(&value, data, sizeof(float));
  return value;
}

void storeFloat(std::uint8_t * data, float value)
{
  std::memcpy(data, &value, sizeof(float));
}

}  // namespace

bool transformPointCloudXYZ(
  const sensor_msgs::msg::PointCloud2 & input,
  sensor_msgs::msg::PointCloud2 & output,
  const RigidTransform & transform,
  std::string & error)
{
  if (input.is_bigendian) {
    error = "big-endian PointCloud2 is not supported";
    return false;
  }
  if (input.point_step == 0) {
    error = "PointCloud2 point_step is zero";
    return false;
  }

  std::uint32_t x_offset = 0;
  std::uint32_t y_offset = 0;
  std::uint32_t z_offset = 0;
  if (!findFloat32Offset(input, "x", x_offset) ||
    !findFloat32Offset(input, "y", y_offset) ||
    !findFloat32Offset(input, "z", z_offset))
  {
    error = "PointCloud2 requires float32 x, y, and z fields";
    return false;
  }
  const std::uint32_t largest_offset = std::max({x_offset, y_offset, z_offset});
  if (largest_offset + sizeof(float) > input.point_step) {
    error = "PointCloud2 xyz field extends beyond point_step";
    return false;
  }

  output = input;
  if (input.row_step <
    static_cast<std::size_t>(input.width) * input.point_step)
  {
    error = "PointCloud2 row_step is shorter than width*point_step";
    return false;
  }
  if (input.width > 0 && input.height > 0) {
    const std::size_t required_size =
      static_cast<std::size_t>(input.height - 1) * input.row_step +
      static_cast<std::size_t>(input.width) * input.point_step;
    if (required_size > output.data.size()) {
      error = "PointCloud2 data is shorter than its organized strides";
      return false;
    }

    for (std::size_t row = 0; row < input.height; ++row) {
      for (std::size_t column = 0; column < input.width; ++column) {
        std::uint8_t * point =
          output.data.data() + row * input.row_step + column * input.point_step;
        const float x = loadFloat(point + x_offset);
        const float y = loadFloat(point + y_offset);
        const float z = loadFloat(point + z_offset);
        if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
          continue;
        }

        const Eigen::Vector3d source(x, y, z);
        const Eigen::Vector3d target =
          transform.rotation * source + transform.translation;
        storeFloat(point + x_offset, static_cast<float>(target.x()));
        storeFloat(point + y_offset, static_cast<float>(target.y()));
        storeFloat(point + z_offset, static_cast<float>(target.z()));
      }
    }
  }
  error.clear();
  return true;
}

bool filterPointCloudToPlanarSlice(
  const sensor_msgs::msg::PointCloud2 & input,
  sensor_msgs::msg::PointCloud2 & output,
  double plane_z,
  double half_thickness,
  std::string & error)
{
  if (input.is_bigendian) {
    error = "big-endian PointCloud2 is not supported";
    return false;
  }
  if (input.point_step == 0) {
    error = "PointCloud2 point_step is zero";
    return false;
  }
  if (!std::isfinite(plane_z) || !std::isfinite(half_thickness) ||
    half_thickness <= 0.0)
  {
    error = "planar slice parameters must be finite and thickness positive";
    return false;
  }

  std::uint32_t x_offset = 0;
  std::uint32_t y_offset = 0;
  std::uint32_t z_offset = 0;
  if (!findFloat32Offset(input, "x", x_offset) ||
    !findFloat32Offset(input, "y", y_offset) ||
    !findFloat32Offset(input, "z", z_offset))
  {
    error = "PointCloud2 requires float32 x, y, and z fields";
    return false;
  }
  const std::uint32_t largest_offset = std::max({x_offset, y_offset, z_offset});
  if (largest_offset + sizeof(float) > input.point_step) {
    error = "PointCloud2 xyz field extends beyond point_step";
    return false;
  }
  if (input.row_step <
    static_cast<std::size_t>(input.width) * input.point_step)
  {
    error = "PointCloud2 row_step is shorter than width*point_step";
    return false;
  }
  if (input.width > 0 && input.height > 0) {
    const std::size_t required_size =
      static_cast<std::size_t>(input.height - 1) * input.row_step +
      static_cast<std::size_t>(input.width) * input.point_step;
    if (required_size > input.data.size()) {
      error = "PointCloud2 data is shorter than its organized strides";
      return false;
    }
  }

  output = input;
  output.height = 1;
  output.width = 0;
  output.row_step = 0;
  output.is_dense = true;
  output.data.clear();
  output.data.reserve(
    static_cast<std::size_t>(input.width) * input.height * input.point_step);

  const float projected_z = static_cast<float>(plane_z);
  for (std::size_t row = 0; row < input.height; ++row) {
    for (std::size_t column = 0; column < input.width; ++column) {
      const std::uint8_t * point =
        input.data.data() + row * input.row_step + column * input.point_step;
      const float x = loadFloat(point + x_offset);
      const float y = loadFloat(point + y_offset);
      const float z = loadFloat(point + z_offset);
      if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z) ||
        std::abs(static_cast<double>(z) - plane_z) > half_thickness)
      {
        continue;
      }
      const std::size_t destination = output.data.size();
      output.data.insert(
        output.data.end(), point, point + input.point_step);
      storeFloat(output.data.data() + destination + z_offset, projected_z);
      ++output.width;
    }
  }
  output.row_step = output.width * output.point_step;
  error.clear();
  return true;
}

}  // namespace racer_adapter
