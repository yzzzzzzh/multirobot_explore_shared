#!/usr/bin/env python3
"""Record a reproducible RACER + Swarm-LIO2 Gazebo exploration benchmark.

The production controller never consumes Gazebo truth.  This process is a
test-only observer: it records the common-frame estimate, Gazebo pose, current
registered LiDAR scans, controller state, and simulation clock.  The resulting
NPZ is rendered and analysed offline by ``render_exploration_video.py``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Bool, String
from tf2_msgs.msg import TFMessage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from racer_control_math import (  # noqa: E402
    conjugate_quaternion,
    multiply_quaternions,
    rotate_body_to_world,
)
from racer_evaluation_math import (  # noqa: E402
    apply_rigid_transform_se3,
    fit_rigid_transform_se3,
)


Position = Tuple[float, float, float]
Quaternion = Tuple[float, float, float, float]
Pose = Tuple[Position, Quaternion]
DEFAULT_ROBOT_XY_RADIUS = 0.356
DEFAULT_ROBOT_BODY_HEIGHT = 0.13


def quaternion_matrix_xyzw(quaternion: Quaternion) -> np.ndarray:
    x, y, z, w = quaternion
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1e-12:
        return np.eye(3)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def yaw_from_quaternion_xyzw(quaternion: Quaternion) -> float:
    x, y, z, w = quaternion
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def pointcloud_xyz(message: PointCloud2, stride: int) -> np.ndarray:
    """Read x/y/z without assuming tightly packed rows or points."""
    fields = {field.name: field for field in message.fields}
    required = [fields.get(name) for name in ("x", "y", "z")]
    if any(field is None for field in required):
        return np.empty((0, 3), dtype=np.float64)
    if any(field.datatype != PointField.FLOAT32 for field in required):
        return np.empty((0, 3), dtype=np.float64)

    endian = ">f4" if message.is_bigendian else "<f4"
    raw = memoryview(message.data)
    # A truncated or inconsistent message (row_step/point_step not matching
    # the buffer) must not abort the recorder; skip that frame instead.
    if (
        message.height <= 0
        or message.width <= 0
        or message.row_step * message.height > len(raw)
        or message.point_step * message.width > message.row_step
    ):
        return np.empty((0, 3), dtype=np.float64)
    components = []
    for field in required:
        try:
            view = np.ndarray(
                shape=(message.height, message.width),
                dtype=endian,
                buffer=raw,
                offset=field.offset,
                strides=(message.row_step, message.point_step),
            )
        except (ValueError, TypeError):
            return np.empty((0, 3), dtype=np.float64)
        components.append(np.asarray(view, dtype=np.float64).reshape(-1)[::stride])
    points = np.column_stack(components)
    return points[np.isfinite(points).all(axis=1)]


def parse_robot_ids(text: str) -> Tuple[int, ...]:
    result = tuple(int(value) for value in text.split(",") if value.strip())
    if not result:
        raise argparse.ArgumentTypeError("at least one UAV id is required")
    return result


class ExplorationRecorder(Node):
    def __init__(
        self,
        robot_ids: Iterable[int],
        duration: float,
        voxel_size: float,
        cloud_stride: int,
        max_voxels: int,
        output_prefix: str,
        duration_basis: str = "wall",
        wait_for_tracking: bool = False,
        progress_interval: float = 30.0,
        divergence_error: float = 5.0,
        stop_on_divergence: bool = False,
        cloud_interval: float = 0.0,
        emergency_clearance: float = 0.0,
        platform: str = "uav",
        cloud_bounds: Tuple[float, float, float, float, float, float] = (
            -2.0,
            -2.0,
            -2.0,
            52.0,
            52.0,
            52.0,
        ),
        robot_xy_radius: float = DEFAULT_ROBOT_XY_RADIUS,
        robot_body_height: float = DEFAULT_ROBOT_BODY_HEIGHT,
    ):
        super().__init__("racer_exploration_recorder")
        self.robot_ids = tuple(robot_ids)
        self.duration = duration
        self.voxel_size = voxel_size
        self.cloud_stride = max(1, cloud_stride)
        self.max_voxels = max_voxels
        self.output_prefix = Path(output_prefix)
        self.duration_basis = duration_basis
        self.wait_for_tracking = wait_for_tracking
        self.progress_interval = max(progress_interval, 1.0)
        self.divergence_error = max(divergence_error, 0.1)
        self.stop_on_divergence = bool(stop_on_divergence)
        self.cloud_interval = max(cloud_interval, 0.0)
        self.emergency_clearance = max(emergency_clearance, 0.0)
        self.platform = str(platform)
        if len(cloud_bounds) != 6:
            raise ValueError("cloud_bounds must contain xmin ymin zmin xmax ymax zmax")
        self.cloud_bounds_lower = np.asarray(cloud_bounds[:3], dtype=np.float64)
        self.cloud_bounds_upper = np.asarray(cloud_bounds[3:], dtype=np.float64)
        if np.any(self.cloud_bounds_upper <= self.cloud_bounds_lower):
            raise ValueError("cloud upper bounds must exceed lower bounds")
        self.robot_xy_radius = max(float(robot_xy_radius), 0.0)
        self.robot_body_height = max(float(robot_body_height), 0.0)

        self.estimated: Dict[int, Pose] = {}
        self.truth: Dict[int, Pose] = {}
        # Callback arrival order is not a valid synchronization signal when
        # PointCloud2 decoding shares the Python executor.  Retain source
        # stamps and synchronize trajectories offline.
        self.estimated_history: Dict[int, list] = {
            robot_id: [] for robot_id in self.robot_ids
        }
        self.truth_history: Dict[int, list] = {
            robot_id: [] for robot_id in self.robot_ids
        }
        self.wheel_odom_history: Dict[int, list] = {
            robot_id: [] for robot_id in self.robot_ids
        }
        self.alignment_stamp: Optional[float] = None
        self.sample_t: Dict[int, list] = {robot_id: [] for robot_id in self.robot_ids}
        self.estimated_samples: Dict[int, list] = {
            robot_id: [] for robot_id in self.robot_ids
        }
        self.truth_samples: Dict[int, list] = {
            robot_id: [] for robot_id in self.robot_ids
        }
        self.distance: Dict[int, float] = {robot_id: 0.0 for robot_id in self.robot_ids}
        self.previous_truth: Dict[int, Position] = {}

        self.first_pose_rotation_q: Optional[Quaternion] = None
        self.first_pose_rotation_m: Optional[np.ndarray] = None
        self.first_pose_translation: Optional[np.ndarray] = None
        self.start_wall: Optional[float] = None
        self.start_sim: Optional[float] = None
        self.latest_sim: Optional[float] = None
        self.last_sample_wall = 0.0
        self.ready_seen = False
        self.last_progress_wall = time.monotonic()
        self.startup_wall = self.last_progress_wall
        self.estimate_last_wall: Dict[int, float] = {}
        self.slam_valid: Dict[int, Optional[bool]] = {
            robot_id: None for robot_id in self.robot_ids
        }
        self.slam_valid_last_wall: Dict[int, float] = {}
        self.slam_valid_samples: Dict[int, int] = {
            robot_id: 0 for robot_id in self.robot_ids
        }
        self.slam_invalid_samples: Dict[int, int] = {
            robot_id: 0 for robot_id in self.robot_ids
        }
        self.slam_valid_transitions: Dict[int, int] = {
            robot_id: 0 for robot_id in self.robot_ids
        }
        self.current_errors: Dict[int, float] = {}
        self.divergence_counts: Dict[int, int] = {
            robot_id: 0 for robot_id in self.robot_ids
        }
        self.monitor_warnings = []
        self.last_cloud_wall: Dict[int, float] = {}
        self.emergency_triggered = False

        self.voxel_keys = set()
        self.voxel_xyz = []
        self.voxel_first_t = []
        self.voxel_robot = []
        self.voxel_count_t = []
        self.voxel_count = []
        self.last_voxel_count_wall = 0.0

        self.status_t = []
        self.status_text = []
        self.tracking_seen = {robot_id: False for robot_id in self.robot_ids}

        self.closest_pairs = {}
        for first_index, first_id in enumerate(self.robot_ids):
            for second_id in self.robot_ids[first_index + 1 :]:
                self.closest_pairs[(first_id, second_id)] = {
                    "min_horizontal_m": float("inf"),
                    "vertical_at_min_horizontal_m": float("inf"),
                    "min_3d_m": float("inf"),
                    "contact_samples": 0,
                    "first_contact_wall_s": None,
                }

        cloud_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        latest_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        status_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=100,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        ready_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.emergency_pub = self.create_publisher(
            Bool, "/racer/emergency_stop", ready_qos
        )
        self.safety_ready_pub = self.create_publisher(
            Bool, "/racer/safety_monitor_ready", ready_qos
        )

        self.subscription_handles = [
            # Keep only the latest high-rate sample.  Deep queues made the
            # single-threaded recorder evaluate stale truth/odometry whenever
            # PointCloud2 decoding briefly occupied the Python callback loop.
            self.create_subscription(
                TFMessage, "/world/default/pose/info", self.on_truth, latest_qos
            ),
            # The standard Gazebo launch already bridges its Pose_V stream to
            # /tf.  Some test runs additionally launch a dedicated
            # /world/default/pose/info bridge; accepting both keeps recording
            # independent of that optional helper.
            self.create_subscription(TFMessage, "/tf", self.on_truth, latest_qos),
            self.create_subscription(Clock, "/clock", self.on_clock, cloud_qos),
            self.create_subscription(
                String, "/racer/controller/status", self.on_status, status_qos
            ),
            self.create_subscription(Bool, "/racer/ready", self.on_ready, ready_qos),
        ]
        for robot_id in self.robot_ids:
            self.subscription_handles.append(
                self.create_subscription(
                    Odometry,
                    f"/racer/bot{robot_id}/odom_world",
                    lambda message, rid=robot_id: self.on_estimate(rid, message),
                    latest_qos,
                )
            )
            if self.platform == "quadruped":
                self.subscription_handles.append(
                    self.create_subscription(
                        Odometry,
                        f"/bot{robot_id}/wheel_odom",
                        lambda message, rid=robot_id: self.on_wheel_odom(
                            rid, message
                        ),
                        latest_qos,
                    )
                )
            self.subscription_handles.append(
                self.create_subscription(
                    Bool,
                    f"/racer/bot{robot_id}/slam_valid",
                    lambda message, rid=robot_id: self.on_slam_valid(rid, message),
                    latest_qos,
                )
            )
        for robot_id in self.robot_ids:
            self.subscription_handles.append(
                self.create_subscription(
                    PointCloud2,
                    f"/racer/bot{robot_id}/cloud_world",
                    lambda message, rid=robot_id: self.on_cloud(rid, message),
                    cloud_qos,
                )
            )

        self.get_logger().info(
            "recording UAVs %s for %.1f %s seconds; wait_for_tracking=%s"
            % (
                self.robot_ids,
                self.duration,
                self.duration_basis,
                self.wait_for_tracking,
            )
        )

    def on_clock(self, message: Clock):
        self.latest_sim = float(message.clock.sec) + float(message.clock.nanosec) * 1e-9

    def on_ready(self, message: Bool):
        self.ready_seen = self.ready_seen or bool(message.data)

    def on_status(self, message: String):
        now = time.monotonic()
        elapsed = 0.0 if self.start_wall is None else now - self.start_wall
        self.status_t.append(elapsed)
        self.status_text.append(message.data)
        if ":tracking_racer_command" in message.data:
            try:
                robot_id = int(message.data.split(":", 1)[0].replace("bot", ""))
            except ValueError:
                return
            if robot_id in self.tracking_seen:
                self.tracking_seen[robot_id] = True

    def on_estimate(self, robot_id: int, message: Odometry):
        p = message.pose.pose.position
        q = message.pose.pose.orientation
        pose = ((p.x, p.y, p.z), (q.x, q.y, q.z, q.w))
        self.estimated[robot_id] = pose
        self.estimate_last_wall[robot_id] = time.monotonic()
        stamp = float(message.header.stamp.sec) + float(message.header.stamp.nanosec) * 1e-9
        if stamp > 0.0:
            self.estimated_history[robot_id].append((stamp, pose))

    def on_slam_valid(self, robot_id: int, message: Bool):
        value = bool(message.data)
        previous = self.slam_valid[robot_id]
        if previous is not None and previous != value:
            self.slam_valid_transitions[robot_id] += 1
        self.slam_valid[robot_id] = value
        self.slam_valid_last_wall[robot_id] = time.monotonic()

    def on_wheel_odom(self, robot_id: int, message: Odometry):
        position = message.pose.pose.position
        orientation = message.pose.pose.orientation
        stamp = float(message.header.stamp.sec) + float(
            message.header.stamp.nanosec
        ) * 1e-9
        if stamp <= 0.0:
            return
        pose = (
            (position.x, position.y, position.z),
            (
                orientation.x,
                orientation.y,
                orientation.z,
                orientation.w,
            ),
        )
        self.wheel_odom_history[robot_id].append((stamp, pose))

    def on_truth(self, message: TFMessage):
        expected = {f"bot{robot_id}": robot_id for robot_id in self.robot_ids}
        for transform in message.transforms:
            robot_id = expected.get(transform.child_frame_id.lstrip("/"))
            if robot_id is None:
                continue
            p = transform.transform.translation
            q = transform.transform.rotation
            pose = ((p.x, p.y, p.z), (q.x, q.y, q.z, q.w))
            self.truth[robot_id] = pose
            stamp = float(transform.header.stamp.sec) + float(
                transform.header.stamp.nanosec
            ) * 1e-9
            # ros_gz_bridge's Pose_V -> TFMessage conversion leaves each
            # transform stamp at zero.  Pair it with the bridged simulation
            # clock rather than callback arrival time.
            if stamp <= 0.0 and self.latest_sim is not None:
                stamp = self.latest_sim
            if stamp > 0.0:
                self.truth_history[robot_id].append((stamp, pose))

    def latch_alignment(self) -> bool:
        if self.wait_for_tracking and not all(self.tracking_seen.values()):
            return False
        root_id = self.robot_ids[0]
        estimates = self.estimated_history[root_id][-80:]
        truths = self.truth_history[root_id][-160:]
        if not estimates or not truths:
            return False
        best = min(
            ((abs(e[0] - g[0]), e, g) for e in estimates for g in truths),
            key=lambda item: item[0],
        )
        if best[0] > 0.05:
            return False
        _, estimate_entry, truth_entry = best
        estimated_position, estimated_q = estimate_entry[1]
        truth_position, truth_q = truth_entry[1]
        rotation_q = multiply_quaternions(truth_q, conjugate_quaternion(estimated_q))
        rotated = rotate_body_to_world(estimated_position, rotation_q)
        translation = np.asarray(truth_position, dtype=np.float64) - np.asarray(
            rotated, dtype=np.float64
        )
        self.first_pose_rotation_q = rotation_q
        self.first_pose_rotation_m = quaternion_matrix_xyzw(rotation_q)
        self.first_pose_translation = translation
        self.alignment_stamp = 0.5 * (estimate_entry[0] + truth_entry[0])
        self.start_wall = time.monotonic()
        self.start_sim = self.alignment_stamp
        self.last_sample_wall = self.start_wall
        self.last_voxel_count_wall = self.start_wall
        self.get_logger().info(
            "latched test-only first-pose SE(3) alignment; recording started"
        )
        return True

    def elapsed(self) -> float:
        if self.start_wall is None:
            return 0.0
        return time.monotonic() - self.start_wall

    def sim_elapsed(self) -> float:
        if self.start_sim is None or self.latest_sim is None:
            return 0.0
        return max(0.0, self.latest_sim - self.start_sim)

    def record_elapsed(self) -> float:
        if self.duration_basis == "sim":
            return self.sim_elapsed()
        return self.elapsed()

    def on_cloud(self, robot_id: int, message: PointCloud2):
        now = time.monotonic()
        if (
            self.cloud_interval > 0.0
            and now - self.last_cloud_wall.get(robot_id, 0.0)
            < self.cloud_interval
        ):
            return
        self.last_cloud_wall[robot_id] = now
        if (
            self.start_wall is None
            or self.first_pose_rotation_m is None
            or self.first_pose_translation is None
            or len(self.voxel_keys) >= self.max_voxels
        ):
            return
        points = pointcloud_xyz(message, self.cloud_stride)
        if points.size == 0:
            return
        points = points @ self.first_pose_rotation_m.T + self.first_pose_translation
        # Reject numerical or corrupt outliers using scenario-specific bounds.
        keep = np.logical_and(
            points >= self.cloud_bounds_lower,
            points <= self.cloud_bounds_upper,
        ).all(axis=1)
        points = points[keep]
        if points.size == 0:
            return

        keys = np.floor(points / self.voxel_size).astype(np.int32)
        keys = np.unique(keys, axis=0)
        first_seen = self.record_elapsed()
        for key_array in keys:
            key = (int(key_array[0]), int(key_array[1]), int(key_array[2]))
            if key in self.voxel_keys:
                continue
            self.voxel_keys.add(key)
            center = (np.asarray(key, dtype=np.float64) + 0.5) * self.voxel_size
            self.voxel_xyz.append(center)
            self.voxel_first_t.append(first_seen)
            self.voxel_robot.append(robot_id)
            if len(self.voxel_keys) >= self.max_voxels:
                break

    def update_pair_clearances(self, elapsed: float):
        for first_index, first_id in enumerate(self.robot_ids):
            if first_id not in self.truth:
                continue
            first = np.asarray(self.truth[first_id][0], dtype=np.float64)
            for second_id in self.robot_ids[first_index + 1 :]:
                if second_id not in self.truth:
                    continue
                second = np.asarray(self.truth[second_id][0], dtype=np.float64)
                delta = first - second
                horizontal = float(np.linalg.norm(delta[:2]))
                vertical = float(abs(delta[2]))
                distance_3d = float(np.linalg.norm(delta))
                stats = self.closest_pairs[(first_id, second_id)]
                if horizontal < stats["min_horizontal_m"]:
                    stats["min_horizontal_m"] = horizontal
                    stats["vertical_at_min_horizontal_m"] = vertical
                stats["min_3d_m"] = min(stats["min_3d_m"], distance_3d)
                if (
                    self.emergency_clearance > 0.0
                    and distance_3d < self.emergency_clearance
                    and not self.emergency_triggered
                ):
                    self.emergency_triggered = True
                    warning = {
                        "type": "inter_uav_emergency_clearance",
                        "bots": [first_id, second_id],
                        "distance_3d_m": distance_3d,
                        "threshold_m": self.emergency_clearance,
                        "wall_elapsed_s": self.elapsed(),
                        "sim_elapsed_s": self.sim_elapsed(),
                    }
                    self.monitor_warnings.append(warning)
                    self.emergency_pub.publish(Bool(data=True))
                    print(
                        "RACER_MONITOR_WARNING="
                        + json.dumps(warning, sort_keys=True),
                        flush=True,
                    )
                if (
                    horizontal < 2.0 * self.robot_xy_radius
                    and vertical < self.robot_body_height
                ):
                    stats["contact_samples"] += 1
                    if stats["first_contact_wall_s"] is None:
                        stats["first_contact_wall_s"] = elapsed

    def sample(self):
        now = time.monotonic()
        self.safety_ready_pub.publish(Bool(data=True))
        if self.start_wall is None:
            # Safety monitoring covers gravity/common-frame initialization as
            # well as the timed exploration.  The formal duration still starts
            # only after every controller reports tracking.
            self.update_pair_clearances(0.0)
            if self.emergency_triggered:
                self.emergency_pub.publish(Bool(data=True))
            self.latch_alignment()
            if now - self.last_progress_wall >= self.progress_interval:
                self.last_progress_wall = now
                print(
                    "RACER_MONITOR_STARTUP="
                    + json.dumps(
                        {
                            "wall_s": now - self.startup_wall,
                            "sim_clock_s": self.latest_sim,
                            "ready": self.ready_seen,
                            "tracking": self.tracking_seen,
                            "estimate_topics": sorted(self.estimated),
                            "truth_topics": sorted(self.truth),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            return
        if now - self.last_sample_wall < 0.1:
            return
        self.last_sample_wall = now
        elapsed = now - self.start_wall

        for robot_id in self.robot_ids:
            if robot_id not in self.estimated or robot_id not in self.truth:
                continue
            valid = self.slam_valid[robot_id]
            if valid is True:
                self.slam_valid_samples[robot_id] += 1
            else:
                # Missing/stale validity is not silently counted as healthy.
                self.slam_invalid_samples[robot_id] += 1
            estimated_position = self.estimated[robot_id][0]
            truth_position = self.truth[robot_id][0]
            self.sample_t[robot_id].append(elapsed)
            self.estimated_samples[robot_id].append(estimated_position)
            self.truth_samples[robot_id].append(truth_position)
            if (
                self.first_pose_rotation_m is not None
                and self.first_pose_translation is not None
            ):
                aligned = (
                    np.asarray(estimated_position, dtype=np.float64)
                    @ self.first_pose_rotation_m.T
                    + self.first_pose_translation
                )
                error = float(
                    np.linalg.norm(
                        aligned - np.asarray(truth_position, dtype=np.float64)
                    )
                )
                self.current_errors[robot_id] = error
                estimate_age = now - self.estimate_last_wall.get(robot_id, 0.0)
                if estimate_age <= 1.0 and (
                    not math.isfinite(error) or error >= self.divergence_error
                ):
                    self.divergence_counts[robot_id] += 1
                else:
                    self.divergence_counts[robot_id] = 0
                if self.divergence_counts[robot_id] == 10:
                    warning = {
                        "type": "persistent_position_error",
                        "bot": robot_id,
                        "error_m": error,
                        "wall_elapsed_s": self.elapsed(),
                        "sim_elapsed_s": self.sim_elapsed(),
                    }
                    self.monitor_warnings.append(warning)
                    if self.stop_on_divergence and not self.emergency_triggered:
                        # This is an evaluation-only ground-truth watchdog.  It
                        # prevents a known-diverged test from wasting compute,
                        # commands every controller to stop, and still lets the
                        # recorder exit normally so the partial NPZ is saved.
                        self.emergency_triggered = True
                        self.emergency_pub.publish(Bool(data=True))
                    print(
                        "RACER_MONITOR_WARNING="
                        + json.dumps(warning, sort_keys=True),
                        flush=True,
                    )
            previous = self.previous_truth.get(robot_id)
            if previous is not None:
                self.distance[robot_id] += math.sqrt(
                    sum(
                        (truth_position[axis] - previous[axis]) ** 2
                        for axis in range(3)
                    )
                )
            self.previous_truth[robot_id] = truth_position

        self.update_pair_clearances(elapsed)
        if now - self.last_voxel_count_wall >= 1.0:
            self.last_voxel_count_wall = now
            self.voxel_count_t.append(elapsed)
            self.voxel_count.append(len(self.voxel_keys))
        if now - self.last_progress_wall >= self.progress_interval:
            self.last_progress_wall = now
            stale = {
                f"bot{robot_id}": now - self.estimate_last_wall.get(robot_id, 0.0)
                for robot_id in self.robot_ids
            }
            min_pair = min(
                (
                    stats["min_3d_m"]
                    for stats in self.closest_pairs.values()
                    if math.isfinite(stats["min_3d_m"])
                ),
                default=None,
            )
            print(
                "RACER_MONITOR_PROGRESS="
                + json.dumps(
                    {
                        "wall_elapsed_s": self.elapsed(),
                        "sim_elapsed_s": self.sim_elapsed(),
                        "rtf": self.sim_elapsed() / max(self.elapsed(), 1e-9),
                        "map_voxels": len(self.voxel_keys),
                        "distance_m": self.distance,
                        "position_error_m": self.current_errors,
                        "slam_valid": {
                            f"bot{robot_id}": self.slam_valid[robot_id]
                            for robot_id in self.robot_ids
                        },
                        "slam_invalid_samples": self.slam_invalid_samples,
                        "estimate_stale_wall_s": stale,
                        "min_inter_uav_3d_m": min_pair,
                        "tracking": self.tracking_seen,
                        "warning_count": len(self.monitor_warnings),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    def finished(self) -> bool:
        return (
            self.start_wall is not None
            and (
                self.record_elapsed() >= self.duration
                or self.emergency_triggered
            )
        )

    @staticmethod
    def error_summary(aligned: np.ndarray, truth: np.ndarray) -> dict:
        count = min(len(aligned), len(truth))
        if count == 0:
            return {"samples": 0}
        errors = np.linalg.norm(aligned[:count] - truth[:count], axis=1)
        signed = aligned[:count] - truth[:count]
        return {
            "samples": int(count),
            "ate_rmse_m": float(np.sqrt(np.mean(errors**2))),
            "mean_error_m": float(np.mean(errors)),
            "max_error_m": float(np.max(errors)),
            "mean_signed_error_xyz_m": np.mean(signed, axis=0).tolist(),
        }

    def synchronized_positions(self, robot_id: int):
        """Interpolate Gazebo truth at each SLAM source timestamp."""
        if self.alignment_stamp is None:
            empty = np.empty((0, 3), dtype=np.float64)
            return np.empty((0,), dtype=np.float64), empty, empty

        def unique_samples(history):
            latest = {}
            for stamp, pose in history:
                latest[float(stamp)] = np.asarray(pose[0], dtype=np.float64)
            stamps = np.asarray(sorted(latest), dtype=np.float64)
            points = (
                np.asarray([latest[stamp] for stamp in stamps], dtype=np.float64)
                if len(stamps)
                else np.empty((0, 3), dtype=np.float64)
            )
            return stamps, points

        estimate_t, estimate = unique_samples(self.estimated_history[robot_id])
        truth_t, truth = unique_samples(self.truth_history[robot_id])
        if len(estimate_t) == 0 or len(truth_t) < 2:
            empty = np.empty((0, 3), dtype=np.float64)
            return np.empty((0,), dtype=np.float64), empty, empty

        valid = (
            (estimate_t >= self.alignment_stamp)
            & (estimate_t >= truth_t[0])
            & (estimate_t <= truth_t[-1])
        )
        estimate_t = estimate_t[valid]
        estimate = estimate[valid]
        if len(estimate_t) == 0:
            empty = np.empty((0, 3), dtype=np.float64)
            return np.empty((0,), dtype=np.float64), empty, empty

        synced_truth = np.column_stack(
            [np.interp(estimate_t, truth_t, truth[:, axis]) for axis in range(3)]
        )
        return estimate_t - self.alignment_stamp, estimate, synced_truth

    def synchronized_wheel_positions(self, robot_id: int):
        """First-pose-align wheel/leg odometry to Gazebo truth for diagnosis.

        This is evaluation-only.  The final quadruped stack does not feed
        wheel odometry into Swarm-LIO2 because skid can make it much worse
        than the LiDAR-inertial estimate.
        """
        if self.alignment_stamp is None:
            empty = np.empty((0, 3), dtype=np.float64)
            return np.empty((0,), dtype=np.float64), empty, empty

        wheel_latest = {
            float(stamp): pose
            for stamp, pose in self.wheel_odom_history[robot_id]
        }
        truth_latest = {
            float(stamp): pose for stamp, pose in self.truth_history[robot_id]
        }
        wheel_t = np.asarray(sorted(wheel_latest), dtype=np.float64)
        truth_t = np.asarray(sorted(truth_latest), dtype=np.float64)
        if len(wheel_t) == 0 or len(truth_t) < 2:
            empty = np.empty((0, 3), dtype=np.float64)
            return np.empty((0,), dtype=np.float64), empty, empty

        valid = (
            (wheel_t >= self.alignment_stamp)
            & (wheel_t >= truth_t[0])
            & (wheel_t <= truth_t[-1])
        )
        wheel_t = wheel_t[valid]
        if len(wheel_t) == 0:
            empty = np.empty((0, 3), dtype=np.float64)
            return np.empty((0,), dtype=np.float64), empty, empty

        wheel_positions = np.asarray(
            [wheel_latest[stamp][0] for stamp in wheel_t], dtype=np.float64
        )
        truth_positions = np.asarray(
            [truth_latest[stamp][0] for stamp in truth_t], dtype=np.float64
        )
        synced_truth = np.column_stack(
            [
                np.interp(wheel_t, truth_t, truth_positions[:, axis])
                for axis in range(3)
            ]
        )

        first_truth_index = int(np.argmin(np.abs(truth_t - wheel_t[0])))
        first_wheel_yaw = yaw_from_quaternion_xyzw(
            wheel_latest[wheel_t[0]][1]
        )
        first_truth_yaw = yaw_from_quaternion_xyzw(
            truth_latest[truth_t[first_truth_index]][1]
        )
        yaw = first_truth_yaw - first_wheel_yaw
        c_yaw, s_yaw = math.cos(yaw), math.sin(yaw)
        rotation_xy = np.asarray(
            [[c_yaw, -s_yaw], [s_yaw, c_yaw]], dtype=np.float64
        )
        aligned = wheel_positions.copy()
        aligned[:, :2] = (
            (wheel_positions[:, :2] - wheel_positions[0, :2])
            @ rotation_xy.T
            + synced_truth[0, :2]
        )
        aligned[:, 2] = (
            wheel_positions[:, 2]
            - wheel_positions[0, 2]
            + synced_truth[0, 2]
        )
        return wheel_t - self.alignment_stamp, aligned, synced_truth

    def write_outputs(self):
        self.output_prefix.parent.mkdir(parents=True, exist_ok=True)
        rotation = self.first_pose_rotation_m
        translation = self.first_pose_translation
        if rotation is None or translation is None:
            raise RuntimeError("no common-frame/Gazebo alignment was recorded")

        synchronized = {}
        for robot_id in self.robot_ids:
            sample_t, estimated, truth = self.synchronized_positions(robot_id)
            synchronized[robot_id] = (sample_t, estimated, truth)
            self.sample_t[robot_id] = sample_t.tolist()
            self.estimated_samples[robot_id] = estimated.tolist()
            self.truth_samples[robot_id] = truth.tolist()
            self.distance[robot_id] = (
                float(np.linalg.norm(np.diff(truth, axis=0), axis=1).sum())
                if len(truth) > 1
                else 0.0
            )

        root_id = self.robot_ids[0]
        root_estimated = self.estimated_samples[root_id]
        root_truth = self.truth_samples[root_id]
        root_values = np.asarray(root_estimated, dtype=np.float64).reshape(-1, 3)
        root_centered = (
            root_values - root_values.mean(axis=0)
            if len(root_values)
            else root_values
        )
        root_excitation_m = (
            float(np.sqrt(np.mean(np.sum(root_centered**2, axis=1))))
            if len(root_centered)
            else 0.0
        )
        root_alignment_valid = (
            len(root_estimated) >= 3
            and self.distance[root_id] >= 0.5
            and root_excitation_m >= 0.15
        )
        if root_alignment_valid:
            trajectory_rotation, trajectory_translation = fit_rigid_transform_se3(
                root_estimated, root_truth
            )
        else:
            # A Kabsch rotation fitted to a hovering root is mathematically
            # underconstrained and can make every non-root ATE meaningless.
            trajectory_rotation, trajectory_translation = rotation, translation

        summary = {
            "robot_ids": list(self.robot_ids),
            "platform": self.platform,
            "cloud_bounds_xyz": {
                "lower": self.cloud_bounds_lower.tolist(),
                "upper": self.cloud_bounds_upper.tolist(),
            },
            "collision_envelope": {
                "xy_radius_m": self.robot_xy_radius,
                "body_height_m": self.robot_body_height,
            },
            "duration_wall_s": self.elapsed(),
            "requested_duration_s": self.duration,
            "duration_basis": self.duration_basis,
            "duration_sim_s": (
                float(max((values[0][-1] for values in synchronized.values() if len(values[0])), default=0.0))
            ),
            "ready_seen": self.ready_seen,
            "root_trajectory_alignment": {
                "valid": root_alignment_valid,
                "root_distance_m": self.distance[root_id],
                "root_rms_excitation_m": root_excitation_m,
                "reason": (
                    "full-SE(3) Kabsch fit"
                    if root_alignment_valid
                    else "root trajectory was too short/weakly excited; first-pose SE(3) used"
                ),
            },
            "map_voxel_size_m": self.voxel_size,
            "mapped_surface_voxels": len(self.voxel_keys),
            "mapped_surface_voxels_per_wall_s": len(self.voxel_keys)
            / max(self.elapsed(), 1e-9),
            "tracking_seen": {
                f"bot{robot_id}": self.tracking_seen[robot_id]
                for robot_id in self.robot_ids
            },
            "monitor_warnings": self.monitor_warnings,
            "stop_on_divergence": self.stop_on_divergence,
            "emergency_stop_triggered": self.emergency_triggered,
            "robots": {},
            "inter_uav_clearance": {},
        }

        arrays = {
            "robot_ids": np.asarray(self.robot_ids, dtype=np.int32),
            "first_pose_rotation": rotation,
            "first_pose_translation": translation,
            "trajectory_rotation": np.asarray(trajectory_rotation, dtype=np.float64),
            "trajectory_translation": np.asarray(
                trajectory_translation, dtype=np.float64
            ),
            "voxel_xyz": np.asarray(self.voxel_xyz, dtype=np.float32).reshape(-1, 3),
            "voxel_first_t": np.asarray(self.voxel_first_t, dtype=np.float32),
            "voxel_robot": np.asarray(self.voxel_robot, dtype=np.int16),
            "voxel_count_t": np.asarray(self.voxel_count_t, dtype=np.float32),
            "voxel_count": np.asarray(self.voxel_count, dtype=np.int32),
            "status_t": np.asarray(self.status_t, dtype=np.float32),
            "status_text": np.asarray(self.status_text, dtype="U128"),
        }

        total_distance = 0.0
        for robot_id in self.robot_ids:
            estimated = np.asarray(
                self.estimated_samples[robot_id], dtype=np.float64
            ).reshape(-1, 3)
            truth = np.asarray(
                self.truth_samples[robot_id], dtype=np.float64
            ).reshape(-1, 3)
            first_pose_aligned = estimated @ rotation.T + translation
            trajectory_aligned = apply_rigid_transform_se3(
                estimated, trajectory_rotation, trajectory_translation
            )
            root_trajectory_summary = self.error_summary(
                np.asarray(trajectory_aligned), truth
            )
            root_trajectory_summary["alignment_valid"] = root_alignment_valid
            summary["robots"][f"bot{robot_id}"] = {
                "ground_truth_distance_m": self.distance[robot_id],
                "first_pose_se3": self.error_summary(first_pose_aligned, truth),
                "root_trajectory_se3": root_trajectory_summary,
                "slam_validity": {
                    "valid_samples": self.slam_valid_samples[robot_id],
                    "invalid_or_missing_samples": self.slam_invalid_samples[robot_id],
                    "valid_fraction": self.slam_valid_samples[robot_id]
                    / max(
                        self.slam_valid_samples[robot_id]
                        + self.slam_invalid_samples[robot_id],
                        1,
                    ),
                    "transitions": self.slam_valid_transitions[robot_id],
                    "final_valid": self.slam_valid[robot_id],
                },
            }
            if self.platform == "quadruped":
                wheel_t, wheel_aligned, wheel_truth = (
                    self.synchronized_wheel_positions(robot_id)
                )
                summary["robots"][f"bot{robot_id}"][
                    "wheel_odom_first_pose_se2"
                ] = self.error_summary(wheel_aligned, wheel_truth)
                arrays[f"wheel_t_{robot_id}"] = wheel_t.astype(np.float32)
                arrays[f"wheel_aligned_{robot_id}"] = wheel_aligned.astype(
                    np.float32
                )
                arrays[f"wheel_truth_{robot_id}"] = wheel_truth.astype(
                    np.float32
                )
            total_distance += self.distance[robot_id]
            arrays[f"t_{robot_id}"] = np.asarray(
                self.sample_t[robot_id], dtype=np.float32
            )
            arrays[f"estimate_{robot_id}"] = estimated.astype(np.float32)
            arrays[f"truth_{robot_id}"] = truth.astype(np.float32)

        summary["total_ground_truth_distance_m"] = total_distance
        summary["mapped_surface_voxels_per_gt_meter"] = len(self.voxel_keys) / max(
            total_distance, 1e-9
        )

        for pair, stats in self.closest_pairs.items():
            clean = {}
            for key, value in stats.items():
                if isinstance(value, float) and not math.isfinite(value):
                    clean[key] = None
                else:
                    clean[key] = value
            clean["contact_detected"] = bool(stats["contact_samples"] > 0)
            summary["inter_uav_clearance"][
                f"bot{pair[0]}-bot{pair[1]}"
            ] = clean

        # Do not expose a partly-written ZIP archive to the postprocessor.  A
        # trajectory file is the sole input to the video renderer, so write it
        # privately, verify every array can be read, then atomically publish it.
        output_npz = Path(str(self.output_prefix) + ".npz")
        temporary_npz = output_npz.with_name(output_npz.stem + ".tmp.npz")
        np.savez_compressed(temporary_npz, **arrays)
        with np.load(temporary_npz, allow_pickle=False) as saved:
            for key in arrays:
                _ = saved[key].shape
        os.replace(temporary_npz, output_npz)
        with open(str(self.output_prefix) + ".summary.json", "w", encoding="utf-8") as stream:
            json.dump(summary, stream, indent=2, sort_keys=True)
        print("RACER_RECORD_RESULT=" + json.dumps(summary, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bots", type=parse_robot_ids, default=(1,))
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--voxel-size", type=float, default=0.25)
    parser.add_argument("--cloud-stride", type=int, default=3)
    parser.add_argument("--max-voxels", type=int, default=250000)
    parser.add_argument("--startup-timeout", type=float, default=120.0)
    parser.add_argument(
        "--duration-basis", choices=("wall", "sim"), default="wall"
    )
    parser.add_argument("--wait-for-tracking", action="store_true")
    parser.add_argument("--progress-interval", type=float, default=30.0)
    parser.add_argument("--divergence-error", type=float, default=5.0)
    parser.add_argument("--stop-on-divergence", action="store_true")
    parser.add_argument("--cloud-interval", type=float, default=0.0)
    parser.add_argument("--emergency-clearance", type=float, default=0.0)
    parser.add_argument("--platform", choices=("uav", "quadruped"), default="uav")
    parser.add_argument(
        "--cloud-bounds",
        type=float,
        nargs=6,
        metavar=("XMIN", "YMIN", "ZMIN", "XMAX", "YMAX", "ZMAX"),
        default=(-2.0, -2.0, -2.0, 52.0, 52.0, 52.0),
    )
    parser.add_argument(
        "--robot-xy-radius", type=float, default=DEFAULT_ROBOT_XY_RADIUS
    )
    parser.add_argument(
        "--robot-body-height", type=float, default=DEFAULT_ROBOT_BODY_HEIGHT
    )
    parser.add_argument("--output-prefix", required=True)
    args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)
    node = ExplorationRecorder(
        args.bots,
        args.duration,
        args.voxel_size,
        args.cloud_stride,
        args.max_voxels,
        args.output_prefix,
        args.duration_basis,
        args.wait_for_tracking,
        args.progress_interval,
        args.divergence_error,
        args.stop_on_divergence,
        args.cloud_interval,
        args.emergency_clearance,
        args.platform,
        tuple(args.cloud_bounds),
        args.robot_xy_radius,
        args.robot_body_height,
    )
    startup_wall = time.monotonic()
    # SIGTERM (docker stop / kill <pid>) requests an early but *complete*
    # shutdown: stop sampling and write the outputs for whatever was recorded.
    stop_requested = {"flag": False}

    def _request_stop(signum, frame):
        stop_requested["flag"] = True

    import signal as _signal
    _signal.signal(_signal.SIGTERM, _request_stop)
    try:
        while rclpy.ok() and not node.finished() and not stop_requested["flag"]:
            rclpy.spin_once(node, timeout_sec=0.02)
            node.sample()
            if (
                node.start_wall is None
                and time.monotonic() - startup_wall > args.startup_timeout
            ):
                raise TimeoutError("common-frame estimate/Gazebo truth startup timed out")
        if stop_requested["flag"]:
            node.get_logger().warning("SIGTERM received: writing outputs early")
        node.write_outputs()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
