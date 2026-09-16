#pragma once

#include <Eigen/Core>
#include <Eigen/SVD>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace swarm_lio {

struct TrajectoryAlignmentResult {
    Eigen::Matrix3d rotation{Eigen::Matrix3d::Identity()};
    Eigen::Vector3d translation{Eigen::Vector3d::Zero()};
    double mean_error{std::numeric_limits<double>::infinity()};
    double coefficient{std::numeric_limits<double>::infinity()};
};

// Pair monotonically stamped trajectories without reusing a sample.  Exact
// timestamp equality is not repeatable for independently scheduled Gazebo
// lidars; the caller still applies the geometric residual acceptance gate.
inline std::size_t associateNearestTimedPositions(
    const std::vector<Eigen::Vector4d>& observed,
    const std::vector<Eigen::Vector4d>& teammate,
    const double maximum_time_delta,
    std::vector<Eigen::Vector3d>& observed_positions,
    std::vector<Eigen::Vector3d>& teammate_positions,
    double* maximum_used_time_delta = nullptr) {
    observed_positions.clear();
    teammate_positions.clear();
    if (maximum_used_time_delta != nullptr) {
        *maximum_used_time_delta = 0.0;
    }
    if (maximum_time_delta < 0.0) {
        return 0;
    }

    std::size_t observed_index = 0;
    std::size_t teammate_index = 0;
    while (observed_index < observed.size() &&
           teammate_index < teammate.size()) {
        const double observed_time = observed[observed_index](3);
        while (
            teammate_index + 1 < teammate.size() &&
            std::abs(teammate[teammate_index + 1](3) - observed_time) <
                std::abs(teammate[teammate_index](3) - observed_time)) {
            ++teammate_index;
        }

        const double time_delta =
            teammate[teammate_index](3) - observed_time;
        const double absolute_time_delta = std::abs(time_delta);
        if (absolute_time_delta <= maximum_time_delta) {
            observed_positions.push_back(
                observed[observed_index].head<3>());
            teammate_positions.push_back(
                teammate[teammate_index].head<3>());
            if (maximum_used_time_delta != nullptr) {
                *maximum_used_time_delta = std::max(
                    *maximum_used_time_delta, absolute_time_delta);
            }
            ++observed_index;
            ++teammate_index;
        } else if (observed_time < teammate[teammate_index](3)) {
            ++observed_index;
        } else {
            ++teammate_index;
        }
    }
    return observed_positions.size();
}

// Use both LIO gravity estimates to remove the weakly observable tilt of a
// short/near-planar trajectory match while preserving its yaw estimate.  This
// is still a full SO(3) rotation: roll and pitch come from the two independently
// estimated world-to-gravity frames rather than being forced to zero.
inline Eigen::Matrix3d constrainRotationWithGravity(
    const Eigen::Matrix3d& estimated_teammate_to_self,
    const Eigen::Matrix3d& self_world_to_gravity,
    const Eigen::Matrix3d& teammate_world_to_gravity) {
    const Eigen::Matrix3d gravity_frame_rotation =
        self_world_to_gravity * estimated_teammate_to_self *
        teammate_world_to_gravity.transpose();
    const double yaw = std::atan2(
        gravity_frame_rotation(1, 0), gravity_frame_rotation(0, 0));
    const double cosine = std::cos(yaw);
    const double sine = std::sin(yaw);
    Eigen::Matrix3d yaw_rotation;
    yaw_rotation << cosine, -sine, 0.0,
                    sine,   cosine, 0.0,
                    0.0,    0.0,    1.0;
    return self_world_to_gravity.transpose() * yaw_rotation *
           teammate_world_to_gravity;
}

// Full 3D Kabsch alignment used by upstream Swarm-LIO2:
// observed_i = rotation * teammate_i + translation.
inline bool solveTrajectoryAlignmentSE3(
    const std::vector<Eigen::Vector3d>& observed_positions,
    const std::vector<Eigen::Vector3d>& teammate_positions,
    TrajectoryAlignmentResult& result) {
    if (observed_positions.size() != teammate_positions.size() ||
        observed_positions.size() < 3) {
        return false;
    }

    Eigen::Vector3d observed_mean = Eigen::Vector3d::Zero();
    Eigen::Vector3d teammate_mean = Eigen::Vector3d::Zero();
    for (std::size_t i = 0; i < observed_positions.size(); ++i) {
        observed_mean += observed_positions[i];
        teammate_mean += teammate_positions[i];
    }
    observed_mean /= static_cast<double>(observed_positions.size());
    teammate_mean /= static_cast<double>(teammate_positions.size());

    Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
    for (std::size_t i = 0; i < observed_positions.size(); ++i) {
        covariance += (teammate_positions[i] - teammate_mean) *
                      (observed_positions[i] - observed_mean).transpose();
    }

    Eigen::JacobiSVD<Eigen::Matrix3d> svd(
        covariance, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d reflection = Eigen::Matrix3d::Identity();
    reflection(2, 2) =
        (svd.matrixV() * svd.matrixU().transpose()).determinant();

    result.rotation =
        svd.matrixV() * reflection * svd.matrixU().transpose();
    result.translation = observed_mean - result.rotation * teammate_mean;

    double total_error = 0.0;
    for (std::size_t i = 0; i < observed_positions.size(); ++i) {
        total_error +=
            (observed_positions[i] -
             result.rotation * teammate_positions[i] -
             result.translation)
                .norm();
    }
    result.mean_error =
        total_error / static_cast<double>(observed_positions.size());

    const auto singular_values = svd.singularValues();
    const double denominator = std::max(
        1e-12, singular_values(0) * singular_values(1));
    result.coefficient = result.mean_error / denominator * 1000.0;
    return result.rotation.allFinite() && result.translation.allFinite() &&
           std::isfinite(result.mean_error) &&
           std::isfinite(result.coefficient);
}

// Original ROS2-port behavior, retained as an explicit compatibility mode.
// It estimates only yaw + x/y and intentionally forces translation z to zero.
inline bool solveTrajectoryAlignmentSE2Legacy(
    const std::vector<Eigen::Vector3d>& observed_positions,
    const std::vector<Eigen::Vector3d>& teammate_positions,
    TrajectoryAlignmentResult& result) {
    if (observed_positions.size() != teammate_positions.size() ||
        observed_positions.size() < 3) {
        return false;
    }

    Eigen::Vector2d observed_mean = Eigen::Vector2d::Zero();
    Eigen::Vector2d teammate_mean = Eigen::Vector2d::Zero();
    for (std::size_t i = 0; i < observed_positions.size(); ++i) {
        observed_mean += observed_positions[i].head<2>();
        teammate_mean += teammate_positions[i].head<2>();
    }
    observed_mean /= static_cast<double>(observed_positions.size());
    teammate_mean /= static_cast<double>(teammate_positions.size());

    Eigen::Matrix2d covariance = Eigen::Matrix2d::Zero();
    for (std::size_t i = 0; i < observed_positions.size(); ++i) {
        covariance +=
            (teammate_positions[i].head<2>() - teammate_mean) *
            (observed_positions[i].head<2>() - observed_mean).transpose();
    }

    Eigen::JacobiSVD<Eigen::Matrix2d> svd(
        covariance, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix2d reflection = Eigen::Matrix2d::Identity();
    reflection(1, 1) =
        (svd.matrixV() * svd.matrixU().transpose()).determinant();
    const Eigen::Matrix2d rotation_xy =
        svd.matrixV() * reflection * svd.matrixU().transpose();

    const double yaw = std::atan2(rotation_xy(1, 0), rotation_xy(0, 0));
    const double c = std::cos(yaw);
    const double s = std::sin(yaw);
    result.rotation << c, -s, 0.0,
                       s,  c, 0.0,
                     0.0, 0.0, 1.0;

    const Eigen::Vector2d translation_xy =
        observed_mean - rotation_xy * teammate_mean;
    result.translation =
        Eigen::Vector3d(translation_xy(0), translation_xy(1), 0.0);

    double total_error = 0.0;
    for (std::size_t i = 0; i < observed_positions.size(); ++i) {
        total_error +=
            (observed_positions[i].head<2>() -
             rotation_xy * teammate_positions[i].head<2>() -
             translation_xy)
                .norm();
    }
    result.mean_error =
        total_error / static_cast<double>(observed_positions.size());

    const auto singular_values = svd.singularValues();
    const double denominator = std::max(
        1e-12, singular_values(0) * singular_values(1));
    result.coefficient = result.mean_error / denominator * 1000.0;
    return result.rotation.allFinite() && result.translation.allFinite() &&
           std::isfinite(result.mean_error) &&
           std::isfinite(result.coefficient);
}

// Upstream Swarm-LIO2 gates on the second singular value: two independent
// trajectory directions are sufficient to determine a proper 3D rigid rotation.
inline double trajectoryExcitationSE3(
    const std::vector<Eigen::Vector3d>& positions) {
    if (positions.size() < 3) {
        return 0.0;
    }
    Eigen::Vector3d mean = Eigen::Vector3d::Zero();
    for (const auto& position : positions) {
        mean += position;
    }
    mean /= static_cast<double>(positions.size());

    Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
    for (const auto& position : positions) {
        const Eigen::Vector3d centered = position - mean;
        covariance += centered * centered.transpose();
    }
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(
        covariance, Eigen::ComputeFullU | Eigen::ComputeFullV);
    return svd.singularValues()(1);
}

inline double trajectoryExcitationSE2Legacy(
    const std::vector<Eigen::Vector3d>& positions) {
    if (positions.size() < 3) {
        return 0.0;
    }
    Eigen::Vector2d mean = Eigen::Vector2d::Zero();
    for (const auto& position : positions) {
        mean += position.head<2>();
    }
    mean /= static_cast<double>(positions.size());

    Eigen::Matrix2d covariance = Eigen::Matrix2d::Zero();
    for (const auto& position : positions) {
        const Eigen::Vector2d centered = position.head<2>() - mean;
        covariance += centered * centered.transpose();
    }
    Eigen::JacobiSVD<Eigen::Matrix2d> svd(
        covariance, Eigen::ComputeFullU | Eigen::ComputeFullV);
    return svd.singularValues()(1);
}

}  // namespace swarm_lio
