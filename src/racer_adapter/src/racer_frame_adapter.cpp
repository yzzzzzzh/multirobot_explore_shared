#include <chrono>
#include <cmath>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/string.hpp>
#include <swarm_msgs/msg/global_extrinsic_status.hpp>
#include <tf2/exceptions.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include "racer_adapter/transform_utils.hpp"

namespace racer_adapter
{

using namespace std::chrono_literals;

struct TimedTransform
{
  std::chrono::steady_clock::time_point received;
  RigidTransform transform;
};

struct Channel
{
  std::string robot_name;
  std::string local_world_frame;
  std::deque<TimedTransform> samples;
  bool stable{false};
  bool has_latched_transform{false};
  RigidTransform latched_transform;
  sensor_msgs::msg::Imu::ConstSharedPtr imu;
  bool state_valid{false};
  // Latest LiDAR height in the robot's own SLAM frame (from odometry), used
  // to cut the planar slice before the cross-robot transform is applied.
  double source_lidar_z{0.0};
  bool has_source_lidar_z{false};
  int recovery_samples{0};
  bool has_accepted_pose{false};
  Eigen::Vector3d last_accepted_position{Eigen::Vector3d::Zero()};
  rclcpp::Time last_accepted_stamp{0, 0, RCL_ROS_TIME};

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_pub;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr lidar_pose_pub;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr validity_pub;
};

class RacerFrameAdapter : public rclcpp::Node
{
public:
  RacerFrameAdapter()
  : Node("racer_frame_adapter"),
    tf_buffer_(this->get_clock()),
    tf_listener_(tf_buffer_)
  {
    const auto ids = declare_parameter<std::vector<std::int64_t>>(
      "uav_ids", std::vector<std::int64_t>{1});
    root_id_ = declare_parameter<int>("root_id", 1);
    robot_prefix_ = declare_parameter<std::string>("robot_prefix", "bot");
    root_frame_ = declare_parameter<std::string>("root_frame", "map_origin");
    output_frame_ = declare_parameter<std::string>("output_frame", "world");
    latch_transforms_ = declare_parameter<bool>("latch_transforms", true);
    require_bootstrap_complete_ =
      declare_parameter<bool>("require_bootstrap_complete", true);
    stability_window_seconds_ =
      declare_parameter<double>("stability_window_seconds", 3.0);
    max_translation_drift_ =
      declare_parameter<double>("max_translation_drift", 0.05);
    max_rotation_drift_rad_ =
      declare_parameter<double>("max_rotation_drift_deg", 0.5) * M_PI / 180.0;
    validate_slam_state_ =
      declare_parameter<bool>("validate_slam_state", true);
    max_slam_speed_ =
      declare_parameter<double>("max_slam_speed", 3.0);
    max_pose_jump_speed_ =
      declare_parameter<double>("max_pose_jump_speed", 3.0);
    pose_jump_margin_ =
      declare_parameter<double>("pose_jump_margin", 0.5);
    max_gravity_disagreement_rad_ =
      declare_parameter<double>("max_gravity_disagreement_deg", 25.0) *
      M_PI / 180.0;
    max_imu_stamp_delta_ =
      declare_parameter<double>("max_imu_stamp_delta", 0.5);
    recovery_samples_required_ =
      declare_parameter<int>("recovery_samples_required", 5);
    planar_mode_ = declare_parameter<bool>("planar_mode", false);
    planar_body_z_ = declare_parameter<double>("planar_body_z", 0.0);
    planar_cloud_z_ = declare_parameter<double>(
      "planar_cloud_z", planar_body_z_);
    planar_cloud_half_thickness_ = declare_parameter<double>(
      "planar_cloud_half_thickness", 0.15);
    // When true the slice is taken in the robot's own SLAM frame around its
    // current LiDAR height, before the latched cross-robot transform.  A z or
    // tilt error of that transform (or LIO z drift) then cannot move the
    // robot's nearby returns out of the slice; the flattened output z is
    // still planar_cloud_z.
    planar_band_in_source_frame_ = declare_parameter<bool>(
      "planar_band_in_source_frame", false);
    // Offset of the source-frame slice centre above the LiDAR height, so the
    // slice can be biased upward (e.g. to include obstacles above the LiDAR
    // that the raw-scan safety barrier also sees).
    planar_source_band_center_offset_ = declare_parameter<double>(
      "planar_source_band_center_offset", 0.0);

    const auto output_translation = declare_parameter<std::vector<double>>(
      "output_transform_translation", std::vector<double>{0.0, 0.0, 0.0});
    const auto output_rotation = declare_parameter<std::vector<double>>(
      "output_transform_rotation_xyzw", std::vector<double>{0.0, 0.0, 0.0, 1.0});
    if (output_translation.size() != 3 || output_rotation.size() != 4) {
      throw std::runtime_error(
              "Output transform requires translation[3] and rotation_xyzw[4]");
    }
    output_from_root_.translation = Eigen::Vector3d(
      output_translation[0], output_translation[1], output_translation[2]);
    output_from_root_.rotation = Eigen::Quaterniond(
      output_rotation[3], output_rotation[0], output_rotation[1], output_rotation[2]);
    output_from_root_.rotation.normalize();

    const auto lidar_translation = declare_parameter<std::vector<double>>(
      "lidar_extrinsic_translation", std::vector<double>{0.0, 0.0, 0.095});
    const auto lidar_rotation = declare_parameter<std::vector<double>>(
      "lidar_extrinsic_rotation_xyzw", std::vector<double>{0.0, 0.0, 0.0, 1.0});
    if (lidar_translation.size() != 3 || lidar_rotation.size() != 4) {
      throw std::runtime_error(
              "LiDAR extrinsic requires translation[3] and rotation_xyzw[4]");
    }
    body_to_lidar_.translation = Eigen::Vector3d(
      lidar_translation[0], lidar_translation[1], lidar_translation[2]);
    body_to_lidar_.rotation = Eigen::Quaterniond(
      lidar_rotation[3], lidar_rotation[0], lidar_rotation[1], lidar_rotation[2]);
    body_to_lidar_.rotation.normalize();

    const auto ready_qos = rclcpp::QoS(1).reliable().transient_local();
    ready_pub_ = create_publisher<std_msgs::msg::Bool>("/racer/ready", ready_qos);
    status_pub_ = create_publisher<std_msgs::msg::String>("/racer/status", ready_qos);
    bootstrap_sub_ = create_subscription<std_msgs::msg::Bool>(
      "/racer/bootstrap_complete", ready_qos,
      [this](std_msgs::msg::Bool::ConstSharedPtr message) {
        bootstrapCallback(*message);
      });
    global_extrinsic_sub_ =
      create_subscription<swarm_msgs::msg::GlobalExtrinsicStatus>(
      "/global_extrinsic_to_teammate", rclcpp::SystemDefaultsQoS(),
      [this](swarm_msgs::msg::GlobalExtrinsicStatus::ConstSharedPtr message) {
        globalExtrinsicCallback(*message);
      });

    for (const auto id64 : ids) {
      const int id = static_cast<int>(id64);
      Channel channel;
      channel.robot_name = robot_prefix_ + std::to_string(id);
      channel.local_world_frame = channel.robot_name + "/world";

      const std::string input_base = "/" + channel.robot_name;
      const std::string output_base = "/racer/" + channel.robot_name;
      channel.cloud_pub = create_publisher<sensor_msgs::msg::PointCloud2>(
        output_base + "/cloud_world", rclcpp::SensorDataQoS());
      channel.odom_pub = create_publisher<nav_msgs::msg::Odometry>(
        output_base + "/odom_world", rclcpp::QoS(20));
      channel.lidar_pose_pub = create_publisher<geometry_msgs::msg::PoseStamped>(
        output_base + "/lidar_pose", rclcpp::QoS(20));
      channel.validity_pub = create_publisher<std_msgs::msg::Bool>(
        output_base + "/slam_valid", rclcpp::QoS(20));

      channels_.emplace(id, std::move(channel));
      auto & stored = channels_.at(id);
      stored.cloud_sub = create_subscription<sensor_msgs::msg::PointCloud2>(
        input_base + "/cloud_registered_sparse",
        rclcpp::SensorDataQoS(),
        [this, id](sensor_msgs::msg::PointCloud2::ConstSharedPtr message) {
          cloudCallback(id, std::move(message));
        });
      stored.odom_sub = create_subscription<nav_msgs::msg::Odometry>(
        input_base + "/lidar_slam/odom",
        rclcpp::QoS(20),
        [this, id](nav_msgs::msg::Odometry::ConstSharedPtr message) {
          odomCallback(id, std::move(message));
        });
      stored.imu_sub = create_subscription<sensor_msgs::msg::Imu>(
        input_base + "/imu",
        rclcpp::SensorDataQoS(),
        [this, id](sensor_msgs::msg::Imu::ConstSharedPtr message) {
          channels_.at(id).imu = std::move(message);
        });
    }
    common_transforms_[root_id_] = RigidTransform{};

    transform_timer_ = create_wall_timer(100ms, [this]() {updateTransforms();});
    publishReady(false, "waiting for stable common-frame transforms");
    RCLCPP_INFO(
      get_logger(),
      "LiDAR-only RACER adapter: %zu UAVs, root=%s, output=%s, latch=%s, "
      "bootstrap_gate=%s, state_validation=%s, planar=%s",
      channels_.size(), root_frame_.c_str(), output_frame_.c_str(),
      latch_transforms_ ? "true" : "false",
      require_bootstrap_complete_ ? "true" : "false",
      validate_slam_state_ ? "true" : "false",
      planar_mode_ ? "true" : "false");
  }

private:
  void bootstrapCallback(const std_msgs::msg::Bool & message)
  {
    std::lock_guard<std::mutex> lock(transform_mutex_);
    const bool complete = message.data;
    if (complete && !bootstrap_complete_) {
      for (auto & [id, channel] : channels_) {
        (void)id;
        channel.samples.clear();
        channel.stable = false;
      }
      RCLCPP_INFO(
        get_logger(),
        "Bootstrap completion received; starting post-excitation transform "
        "stability window");
    }
    bootstrap_complete_ = complete;
  }

  void globalExtrinsicCallback(
    const swarm_msgs::msg::GlobalExtrinsicStatus & message)
  {
    std::lock_guard<std::mutex> lock(transform_mutex_);
    common_transforms_[root_id_] = RigidTransform{};
    for (const auto & extrinsic : message.extrinsic) {
      const int teammate_id = extrinsic.teammate_id;

      constexpr double kDegreesToRadians = M_PI / 180.0;
      const Eigen::AngleAxisd roll(
        static_cast<double>(extrinsic.rot_deg[0]) * kDegreesToRadians,
        Eigen::Vector3d::UnitX());
      const Eigen::AngleAxisd pitch(
        static_cast<double>(extrinsic.rot_deg[1]) * kDegreesToRadians,
        Eigen::Vector3d::UnitY());
      const Eigen::AngleAxisd yaw(
        static_cast<double>(extrinsic.rot_deg[2]) * kDegreesToRadians,
        Eigen::Vector3d::UnitZ());

      RigidTransform transform;
      transform.rotation = Eigen::Quaterniond(yaw * pitch * roll);
      transform.rotation.normalize();
      transform.translation = Eigen::Vector3d(
        extrinsic.trans[0], extrinsic.trans[1], extrinsic.trans[2]);

      if (message.drone_id == root_id_) {
        if (channels_.find(teammate_id) != channels_.end()) {
          // Swarm-LIO publishes T_source_from_teammate.  A root-origin
          // message therefore already contains T_root_from_teammate.
          common_transforms_[teammate_id] = transform;
        }
      } else if (
        teammate_id == root_id_ &&
        channels_.find(message.drone_id) != channels_.end())
      {
        // A direct match can be discovered by either robot.  If only the
        // non-root observer publishes T_robot_from_root, invert that full
        // SE(3) edge instead of waiting indefinitely for graph infection to
        // republish the opposite direction.
        common_transforms_[message.drone_id] = invertTransform(transform);
      }
    }
  }

  bool getActiveTransform(int id, RigidTransform & transform)
  {
    std::lock_guard<std::mutex> lock(transform_mutex_);
    const auto iterator = channels_.find(id);
    if (iterator == channels_.end() || !ready_) {
      return false;
    }
    const auto & channel = iterator->second;
    if (!channel.has_latched_transform) {
      return false;
    }
    transform = channel.latched_transform;
    return true;
  }

  void cloudCallback(
    int id,
    sensor_msgs::msg::PointCloud2::ConstSharedPtr message)
  {
    RigidTransform transform;
    if (!getActiveTransform(id, transform)) {
      return;
    }
    if (validate_slam_state_ && !channels_.at(id).state_valid) {
      return;
    }

    sensor_msgs::msg::PointCloud2 transformed;
    std::string error;
    sensor_msgs::msg::PointCloud2 source_slice;
    const sensor_msgs::msg::PointCloud2 * cloud_in = message.get();
    if (planar_mode_ && planar_band_in_source_frame_) {
      const auto & channel = channels_.at(id);
      if (!channel.has_source_lidar_z) {
        return;
      }
      if (!filterPointCloudToPlanarSlice(
          *message, source_slice,
          channel.source_lidar_z + planar_source_band_center_offset_,
          planar_cloud_half_thickness_, error))
      {
        RCLCPP_ERROR_THROTTLE(
          get_logger(), *get_clock(), 2000,
          "Cannot slice %s cloud in its source frame: %s",
          channel.robot_name.c_str(), error.c_str());
        return;
      }
      cloud_in = &source_slice;
    }
    if (!transformPointCloudXYZ(*cloud_in, transformed, transform, error)) {
      RCLCPP_ERROR_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "Cannot transform %s cloud: %s",
        channels_.at(id).robot_name.c_str(), error.c_str());
      return;
    }
    if (planar_mode_) {
      sensor_msgs::msg::PointCloud2 planar;
      // Second pass: either the real slice (legacy behaviour) or, when the
      // slice was already cut in the source frame, only the z flattening.
      const double second_half_thickness =
        planar_band_in_source_frame_ ? 1.0e9 : planar_cloud_half_thickness_;
      if (!filterPointCloudToPlanarSlice(
          transformed, planar, planar_cloud_z_,
          second_half_thickness, error))
      {
        RCLCPP_ERROR_THROTTLE(
          get_logger(), *get_clock(), 2000,
          "Cannot filter %s cloud to the configured planar slice: %s",
          channels_.at(id).robot_name.c_str(), error.c_str());
        return;
      }
      transformed = std::move(planar);
    }
    transformed.header.frame_id = output_frame_;
    channels_.at(id).cloud_pub->publish(std::move(transformed));
  }

  void odomCallback(
    int id,
    nav_msgs::msg::Odometry::ConstSharedPtr message)
  {
    RigidTransform world_transform;
    if (!getActiveTransform(id, world_transform)) {
      return;
    }

    channels_.at(id).source_lidar_z = message->pose.pose.position.z;
    channels_.at(id).has_source_lidar_z = true;
    nav_msgs::msg::Odometry transformed = *message;
    transformed.header.frame_id = output_frame_;
    transformed.child_frame_id = channels_.at(id).robot_name + "/base_link";
    transformed.pose.pose = transformPose(world_transform, message->pose.pose);

    const Eigen::Vector3d source_linear(
      message->twist.twist.linear.x,
      message->twist.twist.linear.y,
      message->twist.twist.linear.z);
    const Eigen::Vector3d target_linear =
      transformVector(world_transform, source_linear);
    transformed.twist.twist.linear.x = target_linear.x();
    transformed.twist.twist.linear.y = target_linear.y();
    transformed.twist.twist.linear.z = target_linear.z();
    if (planar_mode_) {
      transformed.pose.pose.position.z = planar_body_z_;
      const auto & q = transformed.pose.pose.orientation;
      const double yaw = std::atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z));
      transformed.pose.pose.orientation.x = 0.0;
      transformed.pose.pose.orientation.y = 0.0;
      transformed.pose.pose.orientation.z = std::sin(0.5 * yaw);
      transformed.pose.pose.orientation.w = std::cos(0.5 * yaw);
      transformed.twist.twist.linear.z = 0.0;
      transformed.twist.twist.angular.x = 0.0;
      transformed.twist.twist.angular.y = 0.0;
    }

    auto & channel = channels_.at(id);
    std::string invalid_reason;
    const bool candidate_valid =
      !validate_slam_state_ ||
      validateSlamState(channel, *message, transformed, invalid_reason);
    std_msgs::msg::Bool validity;
    validity.data = candidate_valid;
    if (!candidate_valid) {
      channel.state_valid = false;
      channel.recovery_samples = 0;
      channel.validity_pub->publish(validity);
      RCLCPP_ERROR_THROTTLE(
        get_logger(), *get_clock(), 1000,
        "%s rejected divergent Swarm-LIO odometry: %s",
        channel.robot_name.c_str(), invalid_reason.c_str());
      return;
    }

    if (validate_slam_state_ && channel.has_accepted_pose &&
      !channel.state_valid &&
      channel.recovery_samples + 1 < recovery_samples_required_)
    {
      ++channel.recovery_samples;
      validity.data = false;
      channel.validity_pub->publish(validity);
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 1000,
        "%s has %d/%d consecutive recovery samples",
        channel.robot_name.c_str(), channel.recovery_samples,
        recovery_samples_required_);
      return;
    }

    channel.state_valid = true;
    channel.recovery_samples = recovery_samples_required_;
    channel.has_accepted_pose = true;
    channel.last_accepted_position = Eigen::Vector3d(
      transformed.pose.pose.position.x,
      transformed.pose.pose.position.y,
      transformed.pose.pose.position.z);
    channel.last_accepted_stamp = rclcpp::Time(transformed.header.stamp);
    validity.data = true;
    channel.validity_pub->publish(validity);
    channel.odom_pub->publish(transformed);

    const auto body_pose = transformed.pose.pose;
    const Eigen::Vector3d body_position(
      body_pose.position.x,
      body_pose.position.y,
      body_pose.position.z);
    Eigen::Quaterniond body_rotation(
      body_pose.orientation.w,
      body_pose.orientation.x,
      body_pose.orientation.y,
      body_pose.orientation.z);
    body_rotation.normalize();

    const Eigen::Vector3d lidar_position =
      body_position + body_rotation * body_to_lidar_.translation;
    Eigen::Quaterniond lidar_rotation =
      body_rotation * body_to_lidar_.rotation;
    lidar_rotation.normalize();

    geometry_msgs::msg::PoseStamped lidar_pose;
    lidar_pose.header = transformed.header;
    lidar_pose.pose.position.x = lidar_position.x();
    lidar_pose.pose.position.y = lidar_position.y();
    lidar_pose.pose.position.z = lidar_position.z();
    lidar_pose.pose.orientation.x = lidar_rotation.x();
    lidar_pose.pose.orientation.y = lidar_rotation.y();
    lidar_pose.pose.orientation.z = lidar_rotation.z();
    lidar_pose.pose.orientation.w = lidar_rotation.w();
    channel.lidar_pose_pub->publish(lidar_pose);
  }

  bool validateSlamState(
    const Channel & channel,
    const nav_msgs::msg::Odometry & local,
    const nav_msgs::msg::Odometry & transformed,
    std::string & reason) const
  {
    const auto & position = transformed.pose.pose.position;
    const auto & orientation = local.pose.pose.orientation;
    const auto & velocity = transformed.twist.twist.linear;
    const double values[] = {
      position.x, position.y, position.z,
      orientation.x, orientation.y, orientation.z, orientation.w,
      velocity.x, velocity.y, velocity.z};
    for (const double value : values) {
      if (!std::isfinite(value)) {
        reason = "non-finite pose or velocity";
        return false;
      }
    }

    Eigen::Quaterniond local_rotation(
      orientation.w, orientation.x, orientation.y, orientation.z);
    if (local_rotation.norm() < 1.0e-6) {
      reason = "zero-norm orientation";
      return false;
    }
    local_rotation.normalize();

    const double speed =
      Eigen::Vector3d(velocity.x, velocity.y, velocity.z).norm();
    if (speed > max_slam_speed_) {
      std::ostringstream stream;
      stream << "speed " << speed << " m/s exceeds " << max_slam_speed_;
      reason = stream.str();
      return false;
    }

    if (!channel.imu) {
      reason = "no onboard IMU attitude";
      return false;
    }
    const auto & imu_orientation = channel.imu->orientation;
    Eigen::Quaterniond imu_rotation(
      imu_orientation.w, imu_orientation.x,
      imu_orientation.y, imu_orientation.z);
    if (!std::isfinite(imu_rotation.norm()) ||
      imu_rotation.norm() < 1.0e-6)
    {
      reason = "invalid onboard IMU attitude";
      return false;
    }
    imu_rotation.normalize();

    const rclcpp::Time odom_stamp(local.header.stamp);
    const rclcpp::Time imu_stamp(channel.imu->header.stamp);
    const double imu_delta = std::abs((odom_stamp - imu_stamp).seconds());
    if (imu_delta > max_imu_stamp_delta_) {
      std::ostringstream stream;
      stream << "IMU stamp delta " << imu_delta << " s";
      reason = stream.str();
      return false;
    }

    const Eigen::Vector3d local_up =
      local_rotation * Eigen::Vector3d::UnitZ();
    const Eigen::Vector3d imu_up =
      imu_rotation * Eigen::Vector3d::UnitZ();
    const double gravity_cosine =
      std::max(-1.0, std::min(1.0, local_up.dot(imu_up)));
    const double gravity_disagreement = std::acos(gravity_cosine);
    if (gravity_disagreement > max_gravity_disagreement_rad_) {
      std::ostringstream stream;
      stream << "gravity disagreement "
             << gravity_disagreement * 180.0 / M_PI << " deg";
      reason = stream.str();
      return false;
    }

    if (channel.has_accepted_pose) {
      const double dt =
        (odom_stamp - channel.last_accepted_stamp).seconds();
      if (dt < -1.0e-6) {
        reason = "non-monotonic odometry stamp";
        return false;
      }
      const Eigen::Vector3d candidate(
        position.x, position.y, position.z);
      const double jump =
        (candidate - channel.last_accepted_position).norm();
      const double allowed_jump =
        pose_jump_margin_ + max_pose_jump_speed_ * std::max(0.0, dt);
      if (jump > allowed_jump) {
        std::ostringstream stream;
        stream << "pose jump " << jump << " m exceeds "
               << allowed_jump << " m";
        reason = stream.str();
        return false;
      }
    }
    return true;
  }

  void updateTransforms()
  {
    const auto now = std::chrono::steady_clock::now();
    bool all_stable = true;
    std::ostringstream status;

    std::lock_guard<std::mutex> lock(transform_mutex_);
    if (ready_ && latch_transforms_) {
      return;
    }
    if (require_bootstrap_complete_ && !bootstrap_complete_) {
      for (auto & [id, channel] : channels_) {
        (void)id;
        channel.samples.clear();
        channel.stable = false;
      }
      publishReady(false, "waiting for complete 3D bootstrap excitation");
      return;
    }

    for (auto & [id, channel] : channels_) {
      RigidTransform current;
      bool available = false;
      const auto direct = common_transforms_.find(id);
      if (direct != common_transforms_.end()) {
        current = direct->second;
        available = true;
      } else {
        try {
          const auto message = tf_buffer_.lookupTransform(
            root_frame_, channel.local_world_frame, tf2::TimePointZero);
          current = fromMsg(message.transform);
          available = true;
        } catch (const tf2::TransformException &) {
          available = false;
        }
      }

      if (!available) {
        channel.samples.clear();
        channel.stable = false;
        all_stable = false;
        status << channel.robot_name << ":missing ";
        continue;
      }

      channel.samples.push_back(TimedTransform{now, current});
      const auto window = std::chrono::duration<double>(stability_window_seconds_);
      while (!channel.samples.empty() &&
        now - channel.samples.front().received > window)
      {
        channel.samples.pop_front();
      }

      channel.stable = false;
      if (channel.samples.size() >= 2) {
        const double span = std::chrono::duration<double>(
          channel.samples.back().received -
          channel.samples.front().received).count();
        double translation_drift = 0.0;
        double rotation_drift = 0.0;
        const auto & reference = channel.samples.front().transform;
        for (const auto & sample : channel.samples) {
          translation_drift = std::max(
            translation_drift,
            (sample.transform.translation - reference.translation).norm());
          rotation_drift = std::max(
            rotation_drift,
            rotationDistance(sample.transform.rotation, reference.rotation));
        }
        channel.stable =
          span >= stability_window_seconds_ * 0.9 &&
          translation_drift <= max_translation_drift_ &&
          rotation_drift <= max_rotation_drift_rad_;
        status << channel.robot_name << ":"
               << (channel.stable ? "stable" : "settling")
               << "(dt=" << translation_drift
               << ",dr=" << rotation_drift * 180.0 / M_PI << "deg) ";
      } else {
        status << channel.robot_name << ":settling ";
      }
      all_stable = all_stable && channel.stable;
    }

    if (all_stable && !channels_.empty()) {
      for (auto & [id, channel] : channels_) {
        (void)id;
        channel.latched_transform = composeTransforms(
          output_from_root_, channel.samples.back().transform);
        channel.has_latched_transform = true;
      }
      ready_ = true;
      publishReady(true, "all common-frame transforms stable and latched");
      RCLCPP_INFO(get_logger(), "All common-frame transforms stable and latched");
    } else {
      publishReady(false, status.str());
    }
  }

  void publishReady(bool ready, const std::string & status)
  {
    std_msgs::msg::Bool ready_message;
    ready_message.data = ready;
    ready_pub_->publish(ready_message);
    std_msgs::msg::String status_message;
    status_message.data = status;
    status_pub_->publish(status_message);
  }

  int root_id_{1};
  std::string robot_prefix_;
  std::string root_frame_;
  std::string output_frame_;
  bool latch_transforms_{true};
  bool require_bootstrap_complete_{true};
  bool bootstrap_complete_{false};
  bool ready_{false};
  double stability_window_seconds_{3.0};
  double max_translation_drift_{0.05};
  double max_rotation_drift_rad_{0.5 * M_PI / 180.0};
  bool validate_slam_state_{true};
  double max_slam_speed_{3.0};
  double max_pose_jump_speed_{3.0};
  double pose_jump_margin_{0.5};
  double max_gravity_disagreement_rad_{25.0 * M_PI / 180.0};
  double max_imu_stamp_delta_{0.5};
  int recovery_samples_required_{5};
  bool planar_mode_{false};
  double planar_body_z_{0.0};
  double planar_cloud_z_{0.0};
  double planar_cloud_half_thickness_{0.15};
  bool planar_band_in_source_frame_{false};
  double planar_source_band_center_offset_{0.0};
  RigidTransform body_to_lidar_;
  RigidTransform output_from_root_;

  std::unordered_map<int, Channel> channels_;
  std::unordered_map<int, RigidTransform> common_transforms_;
  std::mutex transform_mutex_;
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  rclcpp::TimerBase::SharedPtr transform_timer_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr ready_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_pub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr bootstrap_sub_;
  rclcpp::Subscription<swarm_msgs::msg::GlobalExtrinsicStatus>::SharedPtr
    global_extrinsic_sub_;
};

}  // namespace racer_adapter

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<racer_adapter::RacerFrameAdapter>());
  rclcpp::shutdown();
  return 0;
}
