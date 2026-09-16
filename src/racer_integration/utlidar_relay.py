#!/usr/bin/env python3
"""Relay the Go2 EDU head-L1 topics from the robot's DDS domain into ours.

The Go2 main board publishes the built-in L1 as ROS 2-compatible DDS topics on
domain 0 (`/utlidar/cloud` sensor_msgs/PointCloud2 with x,y,z,intensity,ring,time
and `/utlidar/imu` sensor_msgs/Imu).  This node subscribes on the source domain
(the process's ROS_DOMAIN_ID, normally 0) and republishes on `target_domain`
under the names the rest of the stack expects:
    /bot<N>/lidar_points/points   frame bot<N>/lidar
    /bot<N>/imu                   frame bot<N>/lidar_imu
It is a pure pass-through: point fields, stamps and per-point times are untouched,
so Swarm-LIO2's UNILIDAR handler (lidar_type 7) consumes them directly.
"""
import os

import rclpy
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu, PointCloud2


class UtlidarRelay(Node):
    def __init__(self, target_context):
        """target_context None -> publish in the same domain/context we subscribe in.

        Same-domain relaying is the robust fallback: no second DDS participant, no
        cross-domain discovery, and it works with rclpy versions that lack
        rclpy.init(domain_id=...).  Our topics keep the /bot<N>/ prefix, so they
        never collide with Unitree's own /utlidar/* names.
        """
        super().__init__("utlidar_relay")
        robot_id = int(self.declare_parameter("robot_id", 1).value)
        source_cloud = str(self.declare_parameter("source_cloud", "/utlidar/cloud").value)
        source_imu = str(self.declare_parameter("source_imu", "/utlidar/imu").value)
        self.cloud_frame = f"bot{robot_id}/lidar"
        self.imu_frame = f"bot{robot_id}/lidar_imu"
        # publishers live in a second node on the TARGET domain
        self.same_domain = target_context is None
        self.target_node = self if self.same_domain else Node(
            "utlidar_relay_out", context=target_context)
        self.cloud_pub = self.target_node.create_publisher(
            PointCloud2, f"/bot{robot_id}/lidar_points/points", qos_profile_sensor_data)
        self.imu_pub = self.target_node.create_publisher(
            Imu, f"/bot{robot_id}/imu", qos_profile_sensor_data)
        self.create_subscription(PointCloud2, source_cloud, self.on_cloud, qos_profile_sensor_data)
        self.create_subscription(Imu, source_imu, self.on_imu, qos_profile_sensor_data)
        self.n_cloud = self.n_imu = 0
        self.create_timer(5.0, self.report)
        self.get_logger().info(
            "relaying %s -> /bot%d/lidar_points/points, %s -> /bot%d/imu (%s)"
            % (source_cloud, robot_id, source_imu, robot_id,
               "same domain" if self.same_domain else "cross domain"))

    def on_cloud(self, msg):
        msg.header.frame_id = self.cloud_frame
        self.cloud_pub.publish(msg)
        self.n_cloud += 1

    def on_imu(self, msg):
        msg.header.frame_id = self.imu_frame
        self.imu_pub.publish(msg)
        self.n_imu += 1

    def report(self):
        self.get_logger().info("last 5 s: %d clouds (%.1f Hz), %d imu (%.0f Hz)"
                               % (self.n_cloud, self.n_cloud / 5.0, self.n_imu, self.n_imu / 5.0))
        if self.n_cloud == 0:
            self.get_logger().warning("no /utlidar cloud: is the head lidar on? "
                                      "(unitree_sdk2_python example go2_utlidar_switch.py on)")
        self.n_cloud = self.n_imu = 0


def main():
    import sys
    # target domain is a parameter, but contexts need it before node creation:
    target_domain = 10
    for i, a in enumerate(sys.argv):
        if a.startswith("target_domain:="):
            target_domain = int(a.split(":=", 1)[1])
    rclpy.init()                                   # source domain = ROS_DOMAIN_ID env (0)
    source_domain = int(os.environ.get("ROS_DOMAIN_ID", "0"))
    target_context = None
    if target_domain != source_domain:
        target_context = Context()
        rclpy.init(context=target_context, domain_id=target_domain)
    node = UtlidarRelay(target_context)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    except Exception:  # noqa: BLE001 - SIGTERM tears the context down under spin()
        if rclpy.ok():
            raise
    finally:
        if not node.same_domain:
            node.target_node.destroy_node()
        node.destroy_node()
        for ctx in ((None,) if node.same_domain else (None, target_context)):
            try:
                rclpy.shutdown(context=ctx)
            except Exception:  # noqa: BLE001
                pass


if __name__ == "__main__":
    main()
