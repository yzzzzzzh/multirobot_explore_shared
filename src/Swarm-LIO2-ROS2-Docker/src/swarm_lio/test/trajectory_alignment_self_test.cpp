#include "trajectory_alignment.hpp"

#include <Eigen/Geometry>

#include <cmath>
#include <iostream>
#include <vector>

namespace {

double rotationError(const Eigen::Matrix3d& estimate,
                     const Eigen::Matrix3d& truth) {
    Eigen::AngleAxisd error(estimate.transpose() * truth);
    return std::abs(error.angle());
}

}  // namespace

int main() {
    const Eigen::Matrix3d truth_rotation =
        (Eigen::AngleAxisd(0.61, Eigen::Vector3d::UnitZ()) *
         Eigen::AngleAxisd(-0.23, Eigen::Vector3d::UnitY()) *
         Eigen::AngleAxisd(0.19, Eigen::Vector3d::UnitX()))
            .toRotationMatrix();
    const Eigen::Vector3d truth_translation(1.25, -0.72, 2.40);

    const std::vector<Eigen::Vector3d> teammate_positions{
        {-1.0, -0.2, 0.4}, {0.3, 1.2, -0.5}, {1.8, -0.7, 0.9},
        {-0.4, 0.6, 1.7},  {1.1, 1.5, 2.2},   {-1.3, 0.8, -1.1},
    };
    std::vector<Eigen::Vector3d> observed_positions;
    observed_positions.reserve(teammate_positions.size());
    for (const auto& position : teammate_positions) {
        observed_positions.push_back(
            truth_rotation * position + truth_translation);
    }

    swarm_lio::TrajectoryAlignmentResult se3;
    if (!swarm_lio::solveTrajectoryAlignmentSE3(
            observed_positions, teammate_positions, se3)) {
        std::cerr << "SE3 solver rejected valid input\n";
        return 1;
    }

    const double rotation_error =
        rotationError(se3.rotation, truth_rotation);
    const double translation_error =
        (se3.translation - truth_translation).norm();
    if (rotation_error > 1e-10 || translation_error > 1e-10 ||
        se3.mean_error > 1e-10) {
        std::cerr << "SE3 recovery failed: rotation_error=" << rotation_error
                  << " translation_error=" << translation_error
                  << " mean_error=" << se3.mean_error << '\n';
        return 2;
    }

    swarm_lio::TrajectoryAlignmentResult legacy;
    if (!swarm_lio::solveTrajectoryAlignmentSE2Legacy(
            observed_positions, teammate_positions, legacy)) {
        std::cerr << "legacy SE2 solver rejected valid input\n";
        return 3;
    }
    if (std::abs(legacy.translation.z()) > 1e-12 ||
        std::abs(legacy.rotation(2, 0)) > 1e-12 ||
        std::abs(legacy.rotation(2, 1)) > 1e-12) {
        std::cerr << "legacy SE2 compatibility behavior changed\n";
        return 4;
    }

    const Eigen::Matrix3d self_world_to_gravity =
        (Eigen::AngleAxisd(-0.03, Eigen::Vector3d::UnitY()) *
         Eigen::AngleAxisd(0.02, Eigen::Vector3d::UnitX()))
            .toRotationMatrix();
    const Eigen::Matrix3d teammate_world_to_gravity =
        (Eigen::AngleAxisd(0.04, Eigen::Vector3d::UnitY()) *
         Eigen::AngleAxisd(-0.01, Eigen::Vector3d::UnitX()))
            .toRotationMatrix();
    const double gravity_yaw = 0.47;
    const Eigen::Matrix3d expected_gravity_rotation =
        self_world_to_gravity.transpose() *
        Eigen::AngleAxisd(gravity_yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix() *
        teammate_world_to_gravity;
    const Eigen::Matrix3d noisy_tilt_rotation =
        self_world_to_gravity.transpose() *
        (Eigen::AngleAxisd(gravity_yaw, Eigen::Vector3d::UnitZ()) *
         Eigen::AngleAxisd(0.18, Eigen::Vector3d::UnitY()) *
         Eigen::AngleAxisd(-0.12, Eigen::Vector3d::UnitX()))
            .toRotationMatrix() *
        teammate_world_to_gravity;
    const Eigen::Matrix3d constrained =
        swarm_lio::constrainRotationWithGravity(
            noisy_tilt_rotation,
            self_world_to_gravity,
            teammate_world_to_gravity);
    if (rotationError(constrained, expected_gravity_rotation) > 1e-10) {
        std::cerr << "gravity-constrained SE3 rotation recovery failed\n";
        return 5;
    }

    std::vector<Eigen::Vector4d> observed_timed;
    std::vector<Eigen::Vector4d> teammate_timed;
    for (int index = 0; index < 80; ++index) {
        const double time = 0.1 * static_cast<double>(index);
        const Eigen::Vector3d teammate_position(
            std::cos(time), std::sin(0.7 * time), 0.1 * time);
        const Eigen::Vector3d observed_position =
            truth_rotation * teammate_position + truth_translation;
        observed_timed.push_back(Eigen::Vector4d(
            observed_position.x(), observed_position.y(),
            observed_position.z(), time));
        teammate_timed.push_back(Eigen::Vector4d(
            teammate_position.x(), teammate_position.y(),
            teammate_position.z(), time + 0.04));
    }
    std::vector<Eigen::Vector3d> synchronized_observed;
    std::vector<Eigen::Vector3d> synchronized_teammate;
    double maximum_used_time_delta = 0.0;
    const std::size_t synchronized_count =
        swarm_lio::associateNearestTimedPositions(
            observed_timed, teammate_timed, 0.06,
            synchronized_observed, synchronized_teammate,
            &maximum_used_time_delta);
    if (synchronized_count != observed_timed.size() ||
        maximum_used_time_delta > 0.0400001) {
        std::cerr << "bounded timestamp association failed\n";
        return 6;
    }
    if (swarm_lio::associateNearestTimedPositions(
            observed_timed, teammate_timed, 0.02,
            synchronized_observed, synchronized_teammate) != 0) {
        std::cerr << "timestamp association accepted samples outside gate\n";
        return 7;
    }

    std::cout << "SE3 trajectory alignment PASS"
              << " rotation_error_rad=" << rotation_error
              << " translation_error_m=" << translation_error
              << " mean_error_m=" << se3.mean_error << '\n';
    std::cout << "legacy SE2 retained"
              << " translation_z_m=" << legacy.translation.z()
              << " mean_xy_error_m=" << legacy.mean_error << '\n';
    std::cout << "gravity-constrained full SO3 PASS\n";
    std::cout << "bounded trajectory timestamp association PASS\n";
    return 0;
}
