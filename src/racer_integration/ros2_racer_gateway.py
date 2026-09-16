#!/usr/bin/env python3
"""ROS 2 side of the narrow Swarm-LIO2 <-> RACER TCP bridge."""

from __future__ import annotations

import argparse
from collections import deque
import math
import socket
import struct
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy._rclpy_pybind11 import RCLError
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Bool, String

sys.path.insert(0, str(Path(__file__).resolve().parent))
import racer_wire  # noqa: E402
from no_return_rays import (  # noqa: E402
    closest_vertical_row,
    no_return_azimuths,
    planar_free_ray_endpoints,
    max_range_hits_to_free_rays,
    organized_xyz,
    spherical_free_ray_endpoints,
    spherical_no_return_directions,
)


SCAN_PREFIX = struct.Struct("<7dI")
ODOMETRY_PAYLOAD = struct.Struct("<13d")
COMMAND_PAYLOAD = struct.Struct("<11d")
EXECUTION_BLOCKED_PAYLOAD = struct.Struct("<B")


def stamp_seconds(message) -> float:
    return float(message.header.stamp.sec) + float(message.header.stamp.nanosec) * 1e-9


def quaternion_from_yaw(yaw: float):
    return 0.0, 0.0, math.sin(yaw * 0.5), math.cos(yaw * 0.5)


def extract_xyzi(message: PointCloud2) -> np.ndarray:
    offsets = {}
    for field in message.fields:
        if field.name in ("x", "y", "z", "intensity"):
            if field.datatype != PointField.FLOAT32 or field.count != 1:
                raise ValueError(f"{field.name} must be one float32")
            offsets[field.name] = int(field.offset)
    for required in ("x", "y", "z"):
        if required not in offsets:
            raise ValueError(f"PointCloud2 has no {required} field")
    if message.is_bigendian:
        raise ValueError("big-endian PointCloud2 is unsupported")
    if message.point_step <= 0 or message.row_step < message.width * message.point_step:
        raise ValueError("invalid PointCloud2 stride")

    point_count = int(message.width) * int(message.height)
    result = np.empty((point_count, 4), dtype="<f4")
    intensity_offset = offsets.get("intensity")
    index = 0
    raw = memoryview(message.data)
    for row in range(int(message.height)):
        row_base = row * int(message.row_step)
        for column in range(int(message.width)):
            base = row_base + column * int(message.point_step)
            result[index, 0] = struct.unpack_from("<f", raw, base + offsets["x"])[0]
            result[index, 1] = struct.unpack_from("<f", raw, base + offsets["y"])[0]
            result[index, 2] = struct.unpack_from("<f", raw, base + offsets["z"])[0]
            result[index, 3] = (
                struct.unpack_from("<f", raw, base + intensity_offset)[0]
                if intensity_offset is not None
                else 0.0
            )
            index += 1
    return result[np.isfinite(result[:, :3]).all(axis=1)]


def extract_xyz_ring(message: PointCloud2, row: int) -> np.ndarray:
    """Decode one organized raw-LiDAR ring without dropping non-finite rays."""
    if row < 0 or row >= int(message.height):
        raise ValueError(f"raw LiDAR row {row} outside height {message.height}")
    offsets = {}
    for field in message.fields:
        if field.name in ("x", "y", "z"):
            if field.datatype != PointField.FLOAT32 or field.count != 1:
                raise ValueError(f"{field.name} must be one float32")
            offsets[field.name] = int(field.offset)
    for required in ("x", "y", "z"):
        if required not in offsets:
            raise ValueError(f"PointCloud2 has no {required} field")
    if message.is_bigendian:
        raise ValueError("big-endian PointCloud2 is unsupported")
    if message.point_step <= 0 or message.row_step < message.width * message.point_step:
        raise ValueError("invalid PointCloud2 stride")

    result = np.empty((int(message.width), 3), dtype="<f4")
    raw = memoryview(message.data)
    row_base = row * int(message.row_step)
    for column in range(int(message.width)):
        base = row_base + column * int(message.point_step)
        result[column, 0] = struct.unpack_from("<f", raw, base + offsets["x"])[0]
        result[column, 1] = struct.unpack_from("<f", raw, base + offsets["y"])[0]
        result[column, 2] = struct.unpack_from("<f", raw, base + offsets["z"])[0]
    return result


class RacerRos2Gateway(Node):
    def __init__(
        self,
        robot_ids: Iterable[int],
        host: str,
        port: int,
        pose_tolerance: float,
        planar_no_return_rays: bool,
        raw_lidar_range: float,
        raw_lidar_horizontal_min: float,
        raw_lidar_horizontal_max: float,
        raw_lidar_vertical_min: float,
        raw_lidar_vertical_max: float,
        raw_lidar_target_elevation: float,
        raw_lidar_max_range_margin: float,
        raw_lidar_match_tolerance: float,
        spherical_no_return_rays: bool = False,
        raw_lidar_row_stride: int = 3,
        raw_lidar_column_stride: int = 3,
        max_range_hits_as_free: bool = False,
        free_ray_range: float = -1.0,
    ):
        super().__init__("racer_ros2_gateway")
        self.robot_ids = tuple(robot_ids)
        self.host = host
        self.port = port
        self.pose_tolerance = pose_tolerance
        self.planar_no_return_rays = planar_no_return_rays
        self.raw_lidar_range = raw_lidar_range
        self.raw_lidar_horizontal_min = raw_lidar_horizontal_min
        self.raw_lidar_horizontal_max = raw_lidar_horizontal_max
        self.raw_lidar_vertical_min = raw_lidar_vertical_min
        self.raw_lidar_vertical_max = raw_lidar_vertical_max
        self.raw_lidar_target_elevation = raw_lidar_target_elevation
        self.raw_lidar_max_range_margin = raw_lidar_max_range_margin
        self.raw_lidar_match_tolerance = raw_lidar_match_tolerance
        # 3-D free rays for a flying 360-degree LiDAR (see no_return_rays.py).
        self.spherical_no_return_rays = bool(spherical_no_return_rays)
        self.raw_lidar_row_stride = max(1, int(raw_lidar_row_stride))
        self.raw_lidar_column_stride = max(1, int(raw_lidar_column_stride))
        self.max_range_hits_as_free = bool(max_range_hits_as_free)
        # Default endpoint one map voxel short of the relabel band so a real
        # surface that sits in [range - margin, range] is left unknown rather
        # than marked free (registered points are 0.3 m voxel centroids).
        self.free_ray_range = (
            float(free_ray_range)
            if free_ray_range > 0.0
            else max(0.5, raw_lidar_range - 2.0 * raw_lidar_max_range_margin - 0.25)
        )
        self.latest_pose: Dict[int, PoseStamped] = {}
        self.raw_no_return_cache = {
            robot_id: deque(maxlen=12) for robot_id in self.robot_ids
        }
        self.connection: Optional[socket.socket] = None
        self.connection_lock = threading.Lock()
        self.sequence_lock = threading.Lock()
        self.sequence = 0
        self.stop_event = threading.Event()
        self.status_pub = self.create_publisher(String, "/racer/bridge/status", 10)
        self.last_clock_sent = -math.inf
        self.setpoint_publishers = {}
        self.feedforward_publishers = {}

        self.subscription_handles = []
        self.subscription_handles.append(
            self.create_subscription(
                Clock, "/clock", self.on_clock, qos_profile_sensor_data
            )
        )
        for robot_id in self.robot_ids:
            base = f"/racer/bot{robot_id}"
            self.subscription_handles.append(
                self.create_subscription(
                    PoseStamped,
                    base + "/lidar_pose",
                    lambda message, rid=robot_id: self.on_pose(rid, message),
                    20,
                )
            )
            self.subscription_handles.append(
                self.create_subscription(
                    Bool,
                    base + "/execution_blocked",
                    lambda message, rid=robot_id: self.on_execution_blocked(
                        rid, message
                    ),
                    10,
                )
            )
            self.subscription_handles.append(
                self.create_subscription(
                    PointCloud2,
                    base + "/cloud_world",
                    lambda message, rid=robot_id: self.on_cloud(rid, message),
                    qos_profile_sensor_data,
                )
            )
            if self.planar_no_return_rays or self.spherical_no_return_rays:
                self.subscription_handles.append(
                    self.create_subscription(
                        PointCloud2,
                        f"/bot{robot_id}/lidar_points/points",
                        lambda message, rid=robot_id: self.on_raw_lidar(
                            rid, message
                        ),
                        qos_profile_sensor_data,
                    )
                )
            self.subscription_handles.append(
                self.create_subscription(
                    Odometry,
                    base + "/odom_world",
                    lambda message, rid=robot_id: self.on_odometry(rid, message),
                    20,
                )
            )
            self.setpoint_publishers[robot_id] = self.create_publisher(
                PoseStamped, base + "/position_setpoint", 20
            )
            self.feedforward_publishers[robot_id] = self.create_publisher(
                TwistStamped, base + "/feedforward", 20
            )

        self.network_thread = threading.Thread(
            target=self.network_loop, name="racer-tcp-client", daemon=True
        )
        self.network_thread.start()
        self.get_logger().info(
            f"narrow bridge configured for UAVs {self.robot_ids}, "
            f"server={self.host}:{self.port}, explicit_no_return_rays="
            f"{self.planar_no_return_rays}, spherical_no_return_rays="
            f"{self.spherical_no_return_rays} (stride {self.raw_lidar_row_stride}x"
            f"{self.raw_lidar_column_stride}, free range {self.free_ray_range:.2f} m), "
            f"max_range_hits_as_free={self.max_range_hits_as_free}"
        )

    def publish_status(self, text: str):
        if not rclpy.ok() or self.stop_event.is_set():
            return
        message = String()
        message.data = text
        try:
            self.status_pub.publish(message)
        except RCLError:
            # The network thread may observe disconnect while ROS is shutting
            # down; publishing status during that small window is invalid.
            pass

    def next_sequence(self) -> int:
        with self.sequence_lock:
            self.sequence = (self.sequence + 1) & 0xFFFFFFFF
            return self.sequence

    def send(self, message_type: int, robot_id: int, stamp: float, payload: bytes):
        frame = racer_wire.encode_frame(
            message_type, robot_id, self.next_sequence(), stamp, payload
        )
        with self.connection_lock:
            connection = self.connection
            if connection is None:
                return
            try:
                connection.sendall(frame)
            except OSError as error:
                self.get_logger().warning(f"bridge send failed: {error}")
                try:
                    connection.close()
                except OSError:
                    pass
                if self.connection is connection:
                    self.connection = None

    def on_pose(self, robot_id: int, message: PoseStamped):
        self.latest_pose[robot_id] = message

    def on_clock(self, message: Clock):
        stamp = float(message.clock.sec) + float(message.clock.nanosec) * 1e-9
        # Gazebo publishes at the physics rate.  RACER only needs a smooth
        # planning clock, so bridge at 50 Hz simulation time to avoid starving
        # LiDAR/odometry frames on the same TCP socket.
        if stamp - self.last_clock_sent < 0.02:
            return
        self.last_clock_sent = stamp
        self.send(racer_wire.CLOCK, 0, stamp, b"")

    def on_raw_lidar(self, robot_id: int, message: PointCloud2):
        if self.spherical_no_return_rays:
            try:
                directions = spherical_no_return_directions(
                    organized_xyz(message),
                    self.raw_lidar_horizontal_min,
                    self.raw_lidar_horizontal_max,
                    self.raw_lidar_vertical_min,
                    self.raw_lidar_vertical_max,
                    self.raw_lidar_range,
                    self.raw_lidar_max_range_margin,
                    self.raw_lidar_row_stride,
                    self.raw_lidar_column_stride,
                )
            except ValueError as error:
                self.get_logger().error(
                    f"bot{robot_id}: cannot decode raw LiDAR 3-D no-return rays: {error}",
                    throttle_duration_sec=2.0,
                )
                return
            self.raw_no_return_cache[robot_id].append(
                (stamp_seconds(message), directions)
            )
            return
        try:
            row = closest_vertical_row(
                int(message.height),
                self.raw_lidar_vertical_min,
                self.raw_lidar_vertical_max,
                self.raw_lidar_target_elevation,
            )
            ring = extract_xyz_ring(message, row)
            azimuths = no_return_azimuths(
                ring,
                self.raw_lidar_horizontal_min,
                self.raw_lidar_horizontal_max,
                self.raw_lidar_range,
                self.raw_lidar_max_range_margin,
            )
        except ValueError as error:
            self.get_logger().error(
                f"bot{robot_id}: cannot decode raw LiDAR no-return rays: {error}",
                throttle_duration_sec=2.0,
            )
            return
        self.raw_no_return_cache[robot_id].append(
            (stamp_seconds(message), azimuths)
        )

    def matching_no_return_azimuths(self, robot_id: int, stamp: float):
        cache = self.raw_no_return_cache[robot_id]
        while (
            cache
            and cache[0][0]
            < stamp - 2.0 * self.raw_lidar_match_tolerance
        ):
            cache.popleft()
        if not cache:
            return None
        differences = [abs(item[0] - stamp) for item in cache]
        index = int(np.argmin(differences))
        tolerance = self.raw_lidar_match_tolerance
        if self.spherical_no_return_rays:
            # The registered cloud carries the raw scan's own stamp (SIM
            # lidar: end time == begin time), so the right match is exact.
            # Never fall back to the previous scan (0.1 s old) and never
            # consume entries when no exact match exists: the raw scan may
            # simply not have been delivered yet.
            tolerance = min(tolerance, 0.02)
        if differences[index] > tolerance:
            return None
        matched = cache[index][1]
        for _ in range(index + 1):
            cache.popleft()
        return matched

    def on_cloud(self, robot_id: int, message: PointCloud2):
        pose = self.latest_pose.get(robot_id)
        if pose is None:
            self.get_logger().warning(
                f"bot{robot_id}: waiting for LiDAR pose before forwarding scans",
                throttle_duration_sec=2.0,
            )
            return
        cloud_stamp = stamp_seconds(message)
        pose_stamp = stamp_seconds(pose)
        if abs(cloud_stamp - pose_stamp) > self.pose_tolerance:
            self.get_logger().warning(
                f"bot{robot_id}: cloud/pose skew {abs(cloud_stamp - pose_stamp):.3f}s "
                f"exceeds {self.pose_tolerance:.3f}s",
                throttle_duration_sec=2.0,
            )
            return
        try:
            points = extract_xyzi(message)
        except ValueError as error:
            self.get_logger().error(
                f"bot{robot_id}: cannot encode LiDAR scan: {error}",
                throttle_duration_sec=2.0,
            )
            return

        p = pose.pose.position
        q = pose.pose.orientation
        if self.max_range_hits_as_free and len(points):
            points = max_range_hits_to_free_rays(
                points,
                np.asarray([p.x, p.y, p.z]),
                self.raw_lidar_range,
                self.raw_lidar_max_range_margin,
                self.free_ray_range,
            )
        if self.spherical_no_return_rays:
            directions = self.matching_no_return_azimuths(robot_id, cloud_stamp)
            if directions is None:
                self.get_logger().warning(
                    f"bot{robot_id}: no timestamp-matched raw LiDAR scan for "
                    "3-D free-ray completion",
                    throttle_duration_sec=2.0,
                )
            else:
                free_endpoints = spherical_free_ray_endpoints(
                    directions,
                    np.asarray([p.x, p.y, p.z]),
                    np.asarray([q.x, q.y, q.z, q.w]),
                    self.free_ray_range,
                )
                if len(free_endpoints):
                    points = np.vstack((points, free_endpoints))
                self.get_logger().info(
                    f"RACER_METRIC no_return_free_rays drone={robot_id} "
                    f"rays={len(free_endpoints)} scan_points={len(points)}",
                    throttle_duration_sec=5.0,
                )
        elif self.planar_no_return_rays:
            azimuths = self.matching_no_return_azimuths(robot_id, cloud_stamp)
            if azimuths is None:
                self.get_logger().warning(
                    f"bot{robot_id}: no timestamp-matched raw LiDAR scan for "
                    "free-ray completion",
                    throttle_duration_sec=2.0,
                )
            else:
                free_endpoints = planar_free_ray_endpoints(
                    azimuths,
                    np.asarray([p.x, p.y, p.z]),
                    np.asarray([q.x, q.y, q.z, q.w]),
                    self.raw_lidar_range,
                )
                if len(free_endpoints):
                    points = np.vstack((points, free_endpoints))
                self.get_logger().info(
                    f"RACER_METRIC no_return_free_rays drone={robot_id} "
                    f"rays={len(free_endpoints)} scan_points={len(points)}",
                    throttle_duration_sec=5.0,
                )
        prefix = SCAN_PREFIX.pack(
            p.x, p.y, p.z, q.x, q.y, q.z, q.w, int(points.shape[0])
        )
        self.send(
            racer_wire.SCAN_BUNDLE,
            robot_id,
            cloud_stamp,
            prefix + points.astype("<f4", copy=False).tobytes(order="C"),
        )

    def on_odometry(self, robot_id: int, message: Odometry):
        p = message.pose.pose.position
        q = message.pose.pose.orientation
        linear = message.twist.twist.linear
        angular = message.twist.twist.angular
        payload = ODOMETRY_PAYLOAD.pack(
            p.x,
            p.y,
            p.z,
            q.x,
            q.y,
            q.z,
            q.w,
            linear.x,
            linear.y,
            linear.z,
            angular.x,
            angular.y,
            angular.z,
        )
        self.send(racer_wire.ODOMETRY, robot_id, stamp_seconds(message), payload)

    def on_execution_blocked(self, robot_id: int, message: Bool):
        self.send(
            racer_wire.EXECUTION_BLOCKED,
            robot_id,
            self.get_clock().now().nanoseconds * 1.0e-9,
            EXECUTION_BLOCKED_PAYLOAD.pack(1 if message.data else 0),
        )

    def handle_command(self, frame: racer_wire.Frame):
        if frame.robot_id not in self.setpoint_publishers:
            return
        if len(frame.payload) != COMMAND_PAYLOAD.size:
            raise ValueError(f"bad command payload size: {len(frame.payload)}")
        values = COMMAND_PAYLOAD.unpack(frame.payload)
        pose = PoseStamped()
        pose.header.frame_id = "world"
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = values[:3]
        qx, qy, qz, qw = quaternion_from_yaw(values[9])
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw
        self.setpoint_publishers[frame.robot_id].publish(pose)

        feedforward = TwistStamped()
        feedforward.header = pose.header
        feedforward.twist.linear.x = values[3]
        feedforward.twist.linear.y = values[4]
        feedforward.twist.linear.z = values[5]
        feedforward.twist.angular.z = values[10]
        self.feedforward_publishers[frame.robot_id].publish(feedforward)

    def network_loop(self):
        while rclpy.ok() and not self.stop_event.is_set():
            with self.connection_lock:
                already_connected = self.connection is not None
            if already_connected:
                time.sleep(0.2)
                continue
            try:
                connection = socket.create_connection((self.host, self.port), timeout=2.0)
                connection.settimeout(None)
                with self.connection_lock:
                    self.connection = connection
                self.publish_status(f"connected to ROS1 gateway at {self.host}:{self.port}")
                self.get_logger().info("connected to ROS1 RACER gateway")
                while rclpy.ok() and not self.stop_event.is_set():
                    frame = racer_wire.recv_frame(connection)
                    if frame.message_type == racer_wire.POSITION_COMMAND:
                        self.handle_command(frame)
            except (OSError, ConnectionError, ValueError) as error:
                if rclpy.ok() and not self.stop_event.is_set():
                    self.publish_status(f"disconnected: {error}")
                with self.connection_lock:
                    if self.connection is not None:
                        try:
                            self.connection.close()
                        except OSError:
                            pass
                        self.connection = None
                if not self.stop_event.wait(1.0):
                    continue

    def destroy_node(self):
        self.stop_event.set()
        with self.connection_lock:
            if self.connection is not None:
                try:
                    self.connection.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                self.connection.close()
                self.connection = None
        super().destroy_node()


def parse_robot_ids(text: str):
    result = tuple(int(value) for value in text.split(",") if value.strip())
    if not result:
        raise argparse.ArgumentTypeError("at least one UAV id is required")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bots", type=parse_robot_ids, default=(1,))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=47100)
    parser.add_argument("--pose-tolerance", type=float, default=0.12)
    parser.add_argument("--planar-no-return-rays", action="store_true")
    parser.add_argument("--raw-lidar-range", type=float, default=10.0)
    parser.add_argument("--raw-lidar-horizontal-min", type=float, default=-math.pi)
    parser.add_argument("--raw-lidar-horizontal-max", type=float, default=math.pi)
    parser.add_argument("--raw-lidar-vertical-min", type=float, default=-0.126)
    parser.add_argument("--raw-lidar-vertical-max", type=float, default=0.907)
    parser.add_argument("--raw-lidar-target-elevation", type=float, default=0.0)
    parser.add_argument("--raw-lidar-max-range-margin", type=float, default=0.05)
    parser.add_argument("--raw-lidar-match-tolerance", type=float, default=0.12)
    # 3-D free rays (flying 360-degree LiDAR): every no-return beam of the
    # organized raw scan, sub-sampled by the strides, is forwarded as a
    # sentinel endpoint so RACER carves the open air it crossed.
    parser.add_argument("--spherical-no-return-rays", action="store_true")
    parser.add_argument("--raw-lidar-row-stride", type=int, default=3)
    parser.add_argument("--raw-lidar-col-stride", type=int, default=3)
    parser.add_argument("--max-range-hits-as-free", action="store_true")
    parser.add_argument("--free-ray-range", type=float, default=-1.0)
    args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)
    node = RacerRos2Gateway(
        args.bots,
        args.host,
        args.port,
        args.pose_tolerance,
        args.planar_no_return_rays,
        args.raw_lidar_range,
        args.raw_lidar_horizontal_min,
        args.raw_lidar_horizontal_max,
        args.raw_lidar_vertical_min,
        args.raw_lidar_vertical_max,
        args.raw_lidar_target_elevation,
        args.raw_lidar_max_range_margin,
        args.raw_lidar_match_tolerance,
        args.spherical_no_return_rays,
        args.raw_lidar_row_stride,
        args.raw_lidar_col_stride,
        args.max_range_hits_as_free,
        args.free_ray_range,
    )
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
