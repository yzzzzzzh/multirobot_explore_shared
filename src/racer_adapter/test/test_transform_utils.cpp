#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>

#include <gtest/gtest.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>

#include "racer_adapter/transform_utils.hpp"

namespace
{

sensor_msgs::msg::PointCloud2 onePointCloud(float x, float y, float z, float intensity)
{
  sensor_msgs::msg::PointCloud2 cloud;
  cloud.height = 1;
  cloud.width = 1;
  cloud.is_bigendian = false;
  cloud.is_dense = true;
  cloud.point_step = 16;
  cloud.row_step = 16;
  cloud.fields.resize(4);
  const char * names[] = {"x", "y", "z", "intensity"};
  for (std::uint32_t i = 0; i < 4; ++i) {
    cloud.fields[i].name = names[i];
    cloud.fields[i].offset = i * 4;
    cloud.fields[i].datatype = sensor_msgs::msg::PointField::FLOAT32;
    cloud.fields[i].count = 1;
  }
  cloud.data.resize(16);
  const float values[] = {x, y, z, intensity};
  std::memcpy(cloud.data.data(), values, sizeof(values));
  return cloud;
}

sensor_msgs::msg::PointCloud2 fourPointCloud()
{
  auto cloud = onePointCloud(1.0F, 2.0F, 0.50F, 10.0F);
  cloud.width = 4;
  cloud.row_step = 64;
  cloud.data.resize(64);
  const float values[] = {
    1.0F, 2.0F, 0.50F, 10.0F,
    3.0F, 4.0F, 0.62F, 20.0F,
    5.0F, 6.0F, 0.90F, 30.0F,
    NAN, 8.0F, 0.55F, 40.0F};
  std::memcpy(cloud.data.data(), values, sizeof(values));
  return cloud;
}

float valueAt(const sensor_msgs::msg::PointCloud2 & cloud, std::size_t offset)
{
  float result;
  std::memcpy(&result, cloud.data.data() + offset, sizeof(float));
  return result;
}

}  // namespace

TEST(TransformUtils, AppliesFullSe3AndPreservesIntensity)
{
  racer_adapter::RigidTransform transform;
  transform.rotation = Eigen::Quaterniond(
    Eigen::AngleAxisd(M_PI / 2.0, Eigen::Vector3d::UnitZ()));
  transform.translation = Eigen::Vector3d(10.0, -2.0, 3.5);

  const auto input = onePointCloud(1.0F, 2.0F, -1.0F, 1234.0F);
  sensor_msgs::msg::PointCloud2 output;
  std::string error;
  ASSERT_TRUE(
    racer_adapter::transformPointCloudXYZ(input, output, transform, error))
    << error;

  EXPECT_NEAR(valueAt(output, 0), 8.0, 1e-6);
  EXPECT_NEAR(valueAt(output, 4), -1.0, 1e-6);
  EXPECT_NEAR(valueAt(output, 8), 2.5, 1e-6);
  EXPECT_FLOAT_EQ(valueAt(output, 12), 1234.0F);
}

TEST(TransformUtils, TransformsPoseOrientationAndPosition)
{
  racer_adapter::RigidTransform transform;
  transform.rotation = Eigen::Quaterniond(
    Eigen::AngleAxisd(M_PI / 2.0, Eigen::Vector3d::UnitX()));
  transform.translation = Eigen::Vector3d(1.0, 2.0, 3.0);

  geometry_msgs::msg::Pose source;
  source.position.x = 0.0;
  source.position.y = 1.0;
  source.position.z = 0.0;
  source.orientation.w = 1.0;
  const auto target = racer_adapter::transformPose(transform, source);

  EXPECT_NEAR(target.position.x, 1.0, 1e-9);
  EXPECT_NEAR(target.position.y, 2.0, 1e-9);
  EXPECT_NEAR(target.position.z, 4.0, 1e-9);
  const Eigen::Quaterniond target_rotation(
    target.orientation.w,
    target.orientation.x,
    target.orientation.y,
    target.orientation.z);
  EXPECT_NEAR(
    racer_adapter::rotationDistance(target_rotation, transform.rotation),
    0.0, 1e-9);
}

TEST(TransformUtils, RotationDistanceUsesShortestQuaternionArc)
{
  const Eigen::Quaterniond rotation(
    Eigen::AngleAxisd(0.25, Eigen::Vector3d::UnitY()));
  Eigen::Quaterniond negated = rotation;
  negated.coeffs() *= -1.0;
  EXPECT_NEAR(racer_adapter::rotationDistance(rotation, negated), 0.0, 1e-9);
}

TEST(TransformUtils, ComposesFullSe3InCorrectOrder)
{
  racer_adapter::RigidTransform target_from_middle;
  target_from_middle.rotation = Eigen::Quaterniond(
    Eigen::AngleAxisd(M_PI / 2.0, Eigen::Vector3d::UnitZ()));
  target_from_middle.translation = Eigen::Vector3d(10.0, 0.0, 1.0);

  racer_adapter::RigidTransform middle_from_source;
  middle_from_source.rotation = Eigen::Quaterniond(
    Eigen::AngleAxisd(M_PI / 2.0, Eigen::Vector3d::UnitX()));
  middle_from_source.translation = Eigen::Vector3d(2.0, 0.0, 3.0);

  const auto composed = racer_adapter::composeTransforms(
    target_from_middle, middle_from_source);
  EXPECT_NEAR(composed.translation.x(), 10.0, 1e-9);
  EXPECT_NEAR(composed.translation.y(), 2.0, 1e-9);
  EXPECT_NEAR(composed.translation.z(), 4.0, 1e-9);

  const Eigen::Vector3d source(0.5, -1.0, 2.0);
  const Eigen::Vector3d sequential =
    target_from_middle.rotation *
    (middle_from_source.rotation * source + middle_from_source.translation) +
    target_from_middle.translation;
  const Eigen::Vector3d direct =
    composed.rotation * source + composed.translation;
  EXPECT_NEAR((sequential - direct).norm(), 0.0, 1e-9);
}

TEST(TransformUtils, InvertsFullSe3)
{
  racer_adapter::RigidTransform target_from_source;
  target_from_source.rotation = Eigen::Quaterniond(
    Eigen::AngleAxisd(0.31, Eigen::Vector3d::UnitX()) *
    Eigen::AngleAxisd(-0.22, Eigen::Vector3d::UnitY()) *
    Eigen::AngleAxisd(0.73, Eigen::Vector3d::UnitZ()));
  target_from_source.translation = Eigen::Vector3d(1.2, -3.4, 0.8);

  const auto source_from_target =
    racer_adapter::invertTransform(target_from_source);
  const auto identity = racer_adapter::composeTransforms(
    source_from_target, target_from_source);

  EXPECT_NEAR(identity.translation.norm(), 0.0, 1e-9);
  EXPECT_NEAR(
    racer_adapter::rotationDistance(
      identity.rotation, Eigen::Quaterniond::Identity()),
    0.0, 1e-9);
}

TEST(TransformUtils, HonorsOrganizedCloudRowPadding)
{
  auto input = onePointCloud(1.0F, 2.0F, 3.0F, 4.0F);
  input.height = 2;
  input.width = 1;
  input.row_step = 24;
  input.data.resize(40, 0x5A);
  const float second_values[] = {5.0F, 6.0F, 7.0F, 8.0F};
  std::memcpy(input.data.data() + 24, second_values, sizeof(second_values));

  racer_adapter::RigidTransform transform;
  transform.translation = Eigen::Vector3d(10.0, 20.0, 30.0);
  sensor_msgs::msg::PointCloud2 output;
  std::string error;
  ASSERT_TRUE(
    racer_adapter::transformPointCloudXYZ(input, output, transform, error))
    << error;

  EXPECT_NEAR(valueAt(output, 0), 11.0, 1e-6);
  EXPECT_NEAR(valueAt(output, 4), 22.0, 1e-6);
  EXPECT_NEAR(valueAt(output, 8), 33.0, 1e-6);
  EXPECT_EQ(output.data[16], 0x5A);
  EXPECT_NEAR(valueAt(output, 24), 15.0, 1e-6);
  EXPECT_NEAR(valueAt(output, 28), 26.0, 1e-6);
  EXPECT_NEAR(valueAt(output, 32), 37.0, 1e-6);
  EXPECT_FLOAT_EQ(valueAt(output, 36), 8.0F);
}

TEST(TransformUtils, KeepsOnlyFinitePointsInsidePlanarSlice)
{
  const auto input = fourPointCloud();
  sensor_msgs::msg::PointCloud2 output;
  std::string error;
  ASSERT_TRUE(
    racer_adapter::filterPointCloudToPlanarSlice(
      input, output, 0.55, 0.08, error))
    << error;

  ASSERT_EQ(output.height, 1U);
  ASSERT_EQ(output.width, 2U);
  ASSERT_EQ(output.row_step, 32U);
  ASSERT_EQ(output.data.size(), 32U);
  EXPECT_NEAR(valueAt(output, 0), 1.0, 1e-6);
  EXPECT_NEAR(valueAt(output, 8), 0.55, 1e-6);
  EXPECT_FLOAT_EQ(valueAt(output, 12), 10.0F);
  EXPECT_NEAR(valueAt(output, 16), 3.0, 1e-6);
  EXPECT_NEAR(valueAt(output, 24), 0.55, 1e-6);
  EXPECT_FLOAT_EQ(valueAt(output, 28), 20.0F);
}
