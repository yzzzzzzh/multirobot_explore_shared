#!/usr/bin/env python3
"""ROS 1 side of the narrow Swarm-LIO2 <-> RACER TCP bridge."""

from __future__ import annotations

import argparse
import socket
import struct
import sys
import threading
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from quadrotor_msgs.msg import PositionCommand
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Bool, Header, String

sys.path.insert(0, str(Path(__file__).resolve().parent))
import racer_wire  # noqa: E402


SCAN_PREFIX = struct.Struct("<7dI")
ODOMETRY_PAYLOAD = struct.Struct("<13d")
COMMAND_PAYLOAD = struct.Struct("<11d")
EXECUTION_BLOCKED_PAYLOAD = struct.Struct("<B")


class RacerRos1Gateway:
    def __init__(self, robot_ids: Iterable[int], bind_address: str, port: int):
        self.robot_ids = tuple(robot_ids)
        self.bind_address = bind_address
        self.port = port
        self.connection: Optional[socket.socket] = None
        self.connection_lock = threading.Lock()
        self.sequence = 0
        self.stop_event = threading.Event()

        self.cloud_publishers = {}
        self.pose_publishers = {}
        self.odom_publishers = {}
        self.execution_blocked_publishers = {}
        self.command_subscribers = []
        self.status_pub = rospy.Publisher("/racer/bridge/status_ros1", String, queue_size=10)
        self.clock_pub = rospy.Publisher("/clock", Clock, queue_size=10)
        for robot_id in self.robot_ids:
            base = f"/racer/bot{robot_id}"
            self.cloud_publishers[robot_id] = rospy.Publisher(
                base + "/cloud_world", PointCloud2, queue_size=2
            )
            self.pose_publishers[robot_id] = rospy.Publisher(
                base + "/lidar_pose", PoseStamped, queue_size=10
            )
            self.odom_publishers[robot_id] = rospy.Publisher(
                base + "/odom_world", Odometry, queue_size=20
            )
            self.execution_blocked_publishers[robot_id] = rospy.Publisher(
                base + "/execution_blocked", Bool, queue_size=10
            )
            self.command_subscribers.append(
                rospy.Subscriber(
                    f"/planning/pos_cmd_{robot_id}",
                    PositionCommand,
                    self.on_command,
                    callback_args=robot_id,
                    queue_size=20,
                )
            )

        self.server_thread = threading.Thread(
            target=self.server_loop, name="racer-tcp-server", daemon=True
        )
        self.server_thread.start()
        rospy.loginfo(
            "narrow bridge configured for UAVs %s, listening on %s:%d",
            self.robot_ids,
            self.bind_address,
            self.port,
        )

    def next_sequence(self) -> int:
        self.sequence = (self.sequence + 1) & 0xFFFFFFFF
        return self.sequence

    def on_command(self, message: PositionCommand, robot_id: int):
        p = message.position
        v = message.velocity
        a = message.acceleration
        payload = COMMAND_PAYLOAD.pack(
            p.x,
            p.y,
            p.z,
            v.x,
            v.y,
            v.z,
            a.x,
            a.y,
            a.z,
            message.yaw,
            message.yaw_dot,
        )
        frame = racer_wire.encode_frame(
            racer_wire.POSITION_COMMAND,
            robot_id,
            self.next_sequence(),
            message.header.stamp.to_sec(),
            payload,
        )
        with self.connection_lock:
            connection = self.connection
            if connection is None:
                return
            try:
                connection.sendall(frame)
            except OSError as error:
                rospy.logwarn_throttle(2.0, "RACER command bridge send failed: %s", error)

    @staticmethod
    def point_fields():
        return [
            PointField("x", 0, PointField.FLOAT32, 1),
            PointField("y", 4, PointField.FLOAT32, 1),
            PointField("z", 8, PointField.FLOAT32, 1),
            PointField("intensity", 12, PointField.FLOAT32, 1),
        ]

    def handle_scan(self, frame: racer_wire.Frame):
        if len(frame.payload) < SCAN_PREFIX.size:
            raise ValueError("truncated scan bundle")
        values = SCAN_PREFIX.unpack_from(frame.payload)
        point_count = values[7]
        expected = SCAN_PREFIX.size + point_count * 16
        if len(frame.payload) != expected:
            raise ValueError(
                f"scan payload size {len(frame.payload)} != expected {expected}"
            )
        stamp = rospy.Time.from_sec(frame.stamp)
        header = Header(stamp=stamp, frame_id="world")

        cloud = PointCloud2()
        cloud.header = header
        cloud.height = 1
        cloud.width = point_count
        cloud.fields = self.point_fields()
        cloud.is_bigendian = False
        cloud.point_step = 16
        cloud.row_step = point_count * 16
        cloud.is_dense = True
        cloud.data = frame.payload[SCAN_PREFIX.size:]

        pose = PoseStamped()
        pose.header = header
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = values[:3]
        (
            pose.pose.orientation.x,
            pose.pose.orientation.y,
            pose.pose.orientation.z,
            pose.pose.orientation.w,
        ) = values[3:7]

        self.pose_publishers[frame.robot_id].publish(pose)
        self.cloud_publishers[frame.robot_id].publish(cloud)

    def handle_odometry(self, frame: racer_wire.Frame):
        if len(frame.payload) != ODOMETRY_PAYLOAD.size:
            raise ValueError(f"bad odometry payload size: {len(frame.payload)}")
        values = ODOMETRY_PAYLOAD.unpack(frame.payload)
        message = Odometry()
        message.header.stamp = rospy.Time.from_sec(frame.stamp)
        message.header.frame_id = "world"
        message.child_frame_id = f"bot{frame.robot_id}/base_link"
        message.pose.pose.position.x, message.pose.pose.position.y, message.pose.pose.position.z = (
            values[:3]
        )
        (
            message.pose.pose.orientation.x,
            message.pose.pose.orientation.y,
            message.pose.pose.orientation.z,
            message.pose.pose.orientation.w,
        ) = values[3:7]
        (
            message.twist.twist.linear.x,
            message.twist.twist.linear.y,
            message.twist.twist.linear.z,
        ) = values[7:10]
        (
            message.twist.twist.angular.x,
            message.twist.twist.angular.y,
            message.twist.twist.angular.z,
        ) = values[10:13]
        self.odom_publishers[frame.robot_id].publish(message)

    def handle_clock(self, frame: racer_wire.Frame):
        self.clock_pub.publish(Clock(clock=rospy.Time.from_sec(frame.stamp)))

    def handle_execution_blocked(self, frame: racer_wire.Frame):
        if len(frame.payload) != EXECUTION_BLOCKED_PAYLOAD.size:
            raise ValueError(
                "bad execution-blocked payload size: "
                f"{len(frame.payload)}"
            )
        (blocked,) = EXECUTION_BLOCKED_PAYLOAD.unpack(frame.payload)
        self.execution_blocked_publishers[frame.robot_id].publish(
            Bool(data=bool(blocked))
        )

    def handle_connection(self, connection: socket.socket):
        while not rospy.is_shutdown() and not self.stop_event.is_set():
            frame = racer_wire.recv_frame(connection)
            if frame.message_type == racer_wire.CLOCK:
                self.handle_clock(frame)
                continue
            if frame.robot_id not in self.cloud_publishers:
                continue
            if frame.message_type == racer_wire.SCAN_BUNDLE:
                self.handle_scan(frame)
            elif frame.message_type == racer_wire.ODOMETRY:
                self.handle_odometry(frame)
            elif frame.message_type == racer_wire.EXECUTION_BLOCKED:
                self.handle_execution_blocked(frame)

    def server_loop(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self.bind_address, self.port))
        server.listen(1)
        server.settimeout(1.0)
        try:
            while not rospy.is_shutdown() and not self.stop_event.is_set():
                try:
                    connection, address = server.accept()
                except socket.timeout:
                    continue
                rospy.loginfo("ROS2 RACER gateway connected from %s:%d", *address)
                self.status_pub.publish(String(data=f"connected from {address[0]}:{address[1]}"))
                with self.connection_lock:
                    if self.connection is not None:
                        self.connection.close()
                    self.connection = connection
                try:
                    self.handle_connection(connection)
                except (OSError, ConnectionError, ValueError) as error:
                    rospy.logwarn("RACER gateway disconnected: %s", error)
                finally:
                    with self.connection_lock:
                        if self.connection is connection:
                            self.connection = None
                    connection.close()
        finally:
            server.close()

    def close(self):
        self.stop_event.set()
        with self.connection_lock:
            if self.connection is not None:
                try:
                    self.connection.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                self.connection.close()
                self.connection = None


def parse_robot_ids(text: str):
    result = tuple(int(value) for value in text.split(",") if value.strip())
    if not result:
        raise argparse.ArgumentTypeError("at least one UAV id is required")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bots", type=parse_robot_ids, default=(1,))
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=47100)
    args, _ = parser.parse_known_args()

    rospy.init_node("racer_ros1_gateway")
    gateway = RacerRos1Gateway(args.bots, args.bind, args.port)
    rospy.on_shutdown(gateway.close)
    rospy.spin()


if __name__ == "__main__":
    main()
