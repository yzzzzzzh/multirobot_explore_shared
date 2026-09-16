#!/usr/bin/env python3
"""Dependency-free control math for the RACER Gazebo velocity controller."""

from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence, Tuple


Vector3 = Tuple[float, float, float]
Quaternion = Tuple[float, float, float, float]  # x, y, z, w


def smoothstep(unit_value: float) -> float:
    """C1-continuous interpolation on [0, 1]."""
    value = clamp(float(unit_value), 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def bootstrap_excitation_offset(
    elapsed: float,
    radius: float,
    climb_height: float,
    vertical_amplitude: float,
    climb_duration: float,
    loop_duration: float,
    direction_angle: float = 0.0,
    lateral_scale: float = 1.0,
) -> Vector3:
    """Return a closed, non-planar initialization trajectory.

    The trajectory starts at the local SLAM origin, climbs, executes one
    horizontal circle with a vertical second harmonic, then descends to the
    origin.  It therefore excites x/y/z while requiring no global frame or
    simulator ground truth.
    """
    if climb_duration <= 0.0 or loop_duration <= 0.0:
        raise ValueError("bootstrap durations must be positive")
    if (
        radius < 0.0
        or climb_height < 0.0
        or vertical_amplitude < 0.0
        or lateral_scale <= 0.0
    ):
        raise ValueError("bootstrap dimensions must be non-negative")

    elapsed = float(elapsed)
    total_duration = 2.0 * climb_duration + loop_duration
    if elapsed <= 0.0 or elapsed >= total_duration:
        return 0.0, 0.0, 0.0

    if elapsed < climb_duration:
        return (
            0.0,
            0.0,
            climb_height * smoothstep(elapsed / climb_duration),
        )

    if elapsed < climb_duration + loop_duration:
        loop_time = elapsed - climb_duration
        angle = 2.0 * math.pi * loop_time / loop_duration
        # Circle centred one radius away from the origin: start/end are zero.
        local_x = radius * (1.0 - math.cos(angle))
        # Ground robots can use a long, narrow ellipse aligned with a
        # corridor.  The default remains the original circle for UAVs.
        local_y = radius * lateral_scale * math.sin(angle)
        cosine = math.cos(direction_angle)
        sine = math.sin(direction_angle)
        return (
            cosine * local_x - sine * local_y,
            sine * local_x + cosine * local_y,
            climb_height + vertical_amplitude * math.sin(2.0 * angle),
        )

    descent_time = elapsed - climb_duration - loop_duration
    return (
        0.0,
        0.0,
        climb_height
        * (1.0 - smoothstep(descent_time / climb_duration)),
    )


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def normalize_quaternion(quaternion: Sequence[float]) -> Quaternion:
    if len(quaternion) != 4:
        raise ValueError("quaternion must contain x, y, z, w")
    x, y, z, w = (float(value) for value in quaternion)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-12:
        raise ValueError("zero-norm quaternion")
    return x / norm, y / norm, z / norm, w / norm


def multiply_quaternions(first: Sequence[float], second: Sequence[float]) -> Quaternion:
    ax, ay, az, aw = normalize_quaternion(first)
    bx, by, bz, bw = normalize_quaternion(second)
    return normalize_quaternion(
        (
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        )
    )


def conjugate_quaternion(quaternion: Sequence[float]) -> Quaternion:
    x, y, z, w = normalize_quaternion(quaternion)
    return -x, -y, -z, w


def rotate_body_to_world(vector: Sequence[float], quaternion: Sequence[float]) -> Vector3:
    """Apply R(q) to a body-frame vector."""
    return rotate_world_to_body(vector, conjugate_quaternion(quaternion))


def rotate_world_to_body(vector: Sequence[float], quaternion: Sequence[float]) -> Vector3:
    """Apply R(q)^T to a world-frame vector.

    q is the body orientation in the world frame, in ROS x,y,z,w order.
    """
    if len(vector) != 3:
        raise ValueError("vector must have three elements")
    vx, vy, vz = (float(value) for value in vector)
    x, y, z, w = normalize_quaternion(quaternion)

    # Rows of R(q)^T, written explicitly to avoid a NumPy dependency.
    return (
        (1.0 - 2.0 * (y * y + z * z)) * vx
        + 2.0 * (x * y + z * w) * vy
        + 2.0 * (x * z - y * w) * vz,
        2.0 * (x * y - z * w) * vx
        + (1.0 - 2.0 * (x * x + z * z)) * vy
        + 2.0 * (y * z + x * w) * vz,
        2.0 * (x * z + y * w) * vx
        + 2.0 * (y * z - x * w) * vy
        + (1.0 - 2.0 * (x * x + y * y)) * vz,
    )


def yaw_from_quaternion(quaternion: Sequence[float]) -> float:
    x, y, z, w = normalize_quaternion(quaternion)
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def ground_unicycle_command(
    velocity_world: Sequence[float],
    current_yaw: float,
    max_forward_speed: float,
    max_yaw_rate: float,
    heading_kp: float,
    turn_in_place_angle: float,
):
    """Convert a desired planar world velocity to forward speed and yaw rate."""
    if len(velocity_world) != 3:
        raise ValueError("velocity_world must be 3D")
    if max_forward_speed < 0.0 or max_yaw_rate < 0.0 or heading_kp < 0.0:
        raise ValueError("unicycle limits and gain must be non-negative")
    if turn_in_place_angle <= 0.0 or turn_in_place_angle > math.pi:
        raise ValueError("turn_in_place_angle must be in (0, pi]")

    vx = float(velocity_world[0])
    vy = float(velocity_world[1])
    desired_speed = min(float(max_forward_speed), math.hypot(vx, vy))
    if desired_speed <= 1.0e-6:
        return 0.0, 0.0, (0.0, 0.0, 0.0)

    desired_yaw = math.atan2(vy, vx)
    heading_error = wrap_angle(desired_yaw - float(current_yaw))
    if abs(heading_error) >= turn_in_place_angle:
        forward_speed = 0.0
    else:
        forward_speed = desired_speed * max(0.0, math.cos(heading_error))
    yaw_rate = clamp(
        float(heading_kp) * heading_error,
        -float(max_yaw_rate),
        float(max_yaw_rate),
    )
    executed_world = (
        forward_speed * math.cos(float(current_yaw)),
        forward_speed * math.sin(float(current_yaw)),
        0.0,
    )
    return forward_speed, yaw_rate, executed_world


def calculate_world_velocity(
    position: Sequence[float],
    target: Sequence[float],
    feedforward: Sequence[float],
    kp: float,
    max_horizontal_speed: float,
    max_vertical_speed: float,
) -> Vector3:
    if len(position) != 3 or len(target) != 3 or len(feedforward) != 3:
        raise ValueError("position, target, and feedforward must be 3D")
    velocity = [
        float(feedforward[index])
        + float(kp) * (float(target[index]) - float(position[index]))
        for index in range(3)
    ]
    horizontal_norm = math.hypot(velocity[0], velocity[1])
    if horizontal_norm > max_horizontal_speed:
        scale = max_horizontal_speed / horizontal_norm
        velocity[0] *= scale
        velocity[1] *= scale
    velocity[2] = clamp(velocity[2], -max_vertical_speed, max_vertical_speed)
    return velocity[0], velocity[1], velocity[2]


def limit_velocity_change(
    previous_velocity: Sequence[float],
    requested_velocity: Sequence[float],
    max_acceleration: float,
    dt: float,
) -> Vector3:
    """Apply a three-dimensional acceleration envelope to a velocity command."""
    if len(previous_velocity) != 3 or len(requested_velocity) != 3:
        raise ValueError("velocity inputs must be 3D")
    if max_acceleration <= 0.0:
        raise ValueError("max_acceleration must be positive")
    if dt <= 0.0:
        raise ValueError("dt must be positive")

    previous = tuple(float(value) for value in previous_velocity)
    requested = tuple(float(value) for value in requested_velocity)
    delta = tuple(requested[index] - previous[index] for index in range(3))
    delta_norm = math.sqrt(sum(value * value for value in delta))
    maximum_delta = float(max_acceleration) * float(dt)
    if delta_norm <= maximum_delta or delta_norm <= 1e-12:
        return requested
    scale = maximum_delta / delta_norm
    return tuple(
        previous[index] + scale * delta[index] for index in range(3)
    )


def obstacle_braking_scale(
    speed: float,
    obstacle_center_clearance: Optional[float],
    hard_clearance: float,
    braking_margin: float,
    deceleration: float,
) -> float:
    """Scale speed so the UAV can stop before a raw-LiDAR obstacle."""
    speed = max(0.0, float(speed))
    if speed <= 1.0e-9 or obstacle_center_clearance is None:
        return 1.0
    available = max(
        0.0,
        float(obstacle_center_clearance)
        - max(0.0, float(hard_clearance))
        - max(0.0, float(braking_margin)),
    )
    safe_speed = math.sqrt(
        2.0 * max(1.0e-6, float(deceleration)) * available
    )
    return clamp(safe_speed / speed, 0.0, 1.0)


def apply_obstacle_velocity_barrier(
    nominal_velocity: Sequence[float],
    measured_velocity: Sequence[float],
    obstacle_vectors: Iterable[Sequence[float]],
    protected_clearance: float,
    repulsion_activation_distance: float,
    repulsion_speed: float,
    braking_acceleration: float,
) -> Vector3:
    """Apply a 3-D point-obstacle stopping and recovery barrier.

    ``obstacle_vectors`` point from the UAV centre to raw-LiDAR returns in the
    same frame as both velocities.  A nominal command toward a return is
    limited to the speed that can stop before ``protected_clearance``.  If the
    measured velocity has already violated that stopping envelope, an active
    reverse command is issued instead of merely commanding zero.  Very close
    returns also produce a bounded repulsive command, which prevents a
    planner/controller deadlock when the mapped inflation shell lags the raw
    scan.
    """
    if len(nominal_velocity) != 3 or len(measured_velocity) != 3:
        raise ValueError("velocity inputs must be 3D")
    if protected_clearance <= 0.0:
        raise ValueError("protected_clearance must be positive")
    if repulsion_activation_distance <= protected_clearance:
        raise ValueError(
            "repulsion activation must exceed protected clearance"
        )
    if repulsion_speed < 0.0:
        raise ValueError("repulsion_speed must be non-negative")
    if braking_acceleration <= 0.0:
        raise ValueError("braking_acceleration must be positive")

    velocity = [float(value) for value in nominal_velocity]
    measured = tuple(float(value) for value in measured_velocity)
    for obstacle in obstacle_vectors:
        vector = tuple(float(value) for value in obstacle)
        if len(vector) != 3:
            raise ValueError("obstacle vectors must be 3D")
        distance = math.sqrt(sum(value * value for value in vector))
        if distance <= 1.0e-9:
            continue
        unit_toward = tuple(value / distance for value in vector)
        nominal_toward = sum(
            velocity[index] * unit_toward[index] for index in range(3)
        )
        measured_toward = sum(
            measured[index] * unit_toward[index] for index in range(3)
        )
        maximum_toward = math.sqrt(
            2.0
            * braking_acceleration
            * max(0.0, distance - protected_clearance)
        )

        # A new command can approach at most the stopping-envelope speed.
        # When actual momentum is already unsafe, command a bounded velocity
        # away from the obstacle so the Gazebo flight controller uses its
        # available deceleration instead of coasting on a zero setpoint.
        allowed_toward = maximum_toward
        if measured_toward > maximum_toward:
            allowed_toward = -min(
                repulsion_speed, measured_toward - maximum_toward
            )

        if distance < repulsion_activation_distance:
            repulsion = repulsion_speed * clamp(
                (repulsion_activation_distance - distance)
                / (
                    repulsion_activation_distance
                    - protected_clearance
                ),
                0.0,
                1.0,
            )
            allowed_toward = min(allowed_toward, -repulsion)

        if nominal_toward > allowed_toward:
            correction = nominal_toward - allowed_toward
            for index in range(3):
                velocity[index] -= correction * unit_toward[index]

    return velocity[0], velocity[1], velocity[2]


def apply_separation_barrier(
    position: Sequence[float],
    nominal_velocity: Sequence[float],
    neighbor_positions: Iterable[Sequence[float]],
    activation_distance: float,
    hard_distance: float,
    repulsion_speed: float,
    braking_acceleration: float,
) -> Vector3:
    """Maintain 3D separation using only common-frame SLAM estimates.

    Each UAV limits its inward radial speed to
    ``sqrt(a * (distance - hard_distance))``. When both UAVs apply this rule,
    their relative stopping distance fits inside the available clearance.
    Repulsion adds margin for asynchronous commands and SLAM error.
    """
    if activation_distance <= 0.0 or hard_distance <= 0.0:
        raise ValueError("separation distances must be positive")
    if hard_distance >= activation_distance:
        raise ValueError("hard distance must be below activation distance")
    if repulsion_speed < 0.0:
        raise ValueError("repulsion speed must be non-negative")
    if braking_acceleration <= 0.0:
        raise ValueError("braking acceleration must be positive")

    current = tuple(float(value) for value in position)
    velocity = [float(value) for value in nominal_velocity]
    if len(current) != 3 or len(velocity) != 3:
        raise ValueError("position and velocity must be 3D")

    for neighbor in neighbor_positions:
        other = tuple(float(value) for value in neighbor)
        if len(other) != 3:
            raise ValueError("neighbor positions must be 3D")
        away = [current[index] - other[index] for index in range(3)]
        distance = math.sqrt(sum(value * value for value in away))
        if distance <= 1e-9 or distance >= activation_distance:
            continue
        unit_away = [value / distance for value in away]
        toward_speed = -sum(
            velocity[index] * unit_away[index] for index in range(3)
        )
        available_clearance = max(0.0, distance - hard_distance)
        max_toward_speed = math.sqrt(
            braking_acceleration * available_clearance
        )
        if toward_speed > max_toward_speed:
            correction = toward_speed - max_toward_speed
            for index in range(3):
                velocity[index] += correction * unit_away[index]
        scale = clamp(
            (activation_distance - distance)
            / (activation_distance - hard_distance),
            0.0,
            1.0,
        )
        for index in range(3):
            velocity[index] += repulsion_speed * scale * unit_away[index]
    return velocity[0], velocity[1], velocity[2]


def position_inside_bounds(
    position: Iterable[float],
    lower: Sequence[float],
    upper: Sequence[float],
) -> bool:
    values = tuple(float(value) for value in position)
    if len(values) != 3 or len(lower) != 3 or len(upper) != 3:
        raise ValueError("bounds check requires three-dimensional inputs")
    return all(float(lower[i]) <= values[i] <= float(upper[i]) for i in range(3))
