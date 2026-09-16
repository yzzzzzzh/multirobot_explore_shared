#!/usr/bin/env python3
"""Print the raw point responsible for a RACER controller LiDAR brake."""

import argparse
import math
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import PointCloud2, PointField


class Inspector(Node):
    def __init__(self, robot_id):
        super().__init__("racer_lidar_safety_inspector")
        self.command = None
        self.last_print = 0.0
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(
            Twist, f"/bot{robot_id}/cmd_vel", self.on_command, 20
        )
        self.create_subscription(
            PointCloud2,
            f"/bot{robot_id}/lidar_points/points",
            self.on_scan,
            qos,
        )

    def on_command(self, message):
        self.command = np.asarray(
            [message.linear.x, message.linear.y, message.linear.z],
            dtype=np.float32,
        )

    def on_scan(self, message):
        now = time.monotonic()
        if self.command is None or now - self.last_print < 1.0:
            return
        field_by_name = {field.name: field for field in message.fields}
        xyz = [field_by_name.get(name) for name in ("x", "y", "z")]
        if any(
            field is None or field.datatype != PointField.FLOAT32
            for field in xyz
        ):
            return
        endian = ">" if message.is_bigendian else "<"
        dtype = np.dtype(
            {
                "names": ("x", "y", "z"),
                "formats": (endian + "f4",) * 3,
                "offsets": tuple(field.offset for field in xyz),
                "itemsize": int(message.point_step),
            }
        )
        count = int(message.width) * int(message.height)
        values = np.frombuffer(message.data, dtype=dtype, count=count)[::8]
        points = np.column_stack(
            (values["x"], values["y"], values["z"])
        ).astype(np.float32, copy=False)
        points = points[np.isfinite(points).all(axis=1)].copy()
        raw_count = len(points)
        points[:, 2] += 0.15
        horizontal_sq = points[:, 0] ** 2 + points[:, 1] ** 2
        self_return = np.logical_and(
            horizontal_sq <= 0.40**2, np.abs(points[:, 2]) <= 0.30
        )
        points = points[np.logical_not(self_return)]
        radial = np.linalg.norm(points, axis=1)
        points = points[radial >= 0.20]
        speed = float(np.linalg.norm(self.command))
        if not len(points):
            return
        if speed <= 1.0e-6:
            radial = np.linalg.norm(points, axis=1)
            chosen = int(np.argmin(radial))
            point = points[chosen]
            self.last_print = now
            print(
                "raw=%d kept=%d speed=0 point=(%.3f,%.3f,%.3f) "
                "min_radial=%.3f"
                % (
                    raw_count,
                    len(points),
                    point[0],
                    point[1],
                    point[2],
                    radial[chosen],
                ),
                flush=True,
            )
            return
        direction = self.command / speed
        longitudinal = points @ direction
        distance_sq = np.einsum("ij,ij->i", points, points)
        lateral_sq = np.maximum(
            0.0, distance_sq - longitudinal * longitudinal
        )
        candidates = np.flatnonzero(
            np.logical_and(longitudinal > 0.0, lateral_sq <= 0.55**2)
        )
        if not len(candidates):
            return
        chosen = candidates[np.argmin(longitudinal[candidates])]
        point = points[chosen]
        self.last_print = now
        print(
            "raw=%d kept=%d speed=%.3f point=(%.3f,%.3f,%.3f) "
            "radial=%.3f longitudinal=%.4f lateral=%.3f"
            % (
                raw_count,
                len(points),
                speed,
                point[0],
                point[1],
                point[2],
                math.sqrt(float(distance_sq[chosen])),
                longitudinal[chosen],
                math.sqrt(float(lateral_sq[chosen])),
            ),
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bot", type=int, required=True)
    parser.add_argument("--seconds", type=float, default=10.0)
    args, ros_args = parser.parse_known_args()
    rclpy.init(args=ros_args)
    node = Inspector(args.bot)
    started = time.monotonic()
    try:
        while rclpy.ok() and time.monotonic() - started < args.seconds:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
