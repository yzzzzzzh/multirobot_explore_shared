#!/usr/bin/env python3
"""cmd_vel -> Unitree Go2 sport-mode bridge (real robot).

Replaces the simulation's twist_to_control_input.py: the RL joint controller is
Unitree's own onboard sport mode, driven through unitree_sdk2 (DDS) with
SportClient.Move(vx, vy, vyaw).  Everything above this node (Swarm-LIO2,
adapter, gateway, RACER, velocity controller) is unchanged.

Safety envelope (all parameters):
  * slew-rate limit on the body velocity command (max_linear_accel /
    max_angular_accel), same semantics as the simulation bridge;
  * command timeout -> StopMove, then BalanceStand hold;
  * hard clamps vx/vy/wz; optional obstacle-avoidance switch-off so Unitree's
    own avoidance does not fight the RACER controller;
  * E-stop topic (std_msgs/Bool on <ns>/estop): true -> StopMove + Damp.

Run:  python3 go2_sport_bridge.py --ros-args -p network_interface:=eth0 -p robot_id:=1
"""
import math
import threading
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, String


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


class Go2SportBridge(Node):
    def __init__(self):
        super().__init__("go2_sport_bridge")
        self.robot_id = int(self.declare_parameter("robot_id", 1).value)
        self.network_interface = str(self.declare_parameter("network_interface", "eth0").value)
        self.dds_domain = int(self.declare_parameter("dds_domain", 0).value)
        self.rate_hz = float(self.declare_parameter("rate_hz", 20.0).value)
        self.cmd_timeout = float(self.declare_parameter("cmd_timeout", 0.5).value)
        self.vx_limit = float(self.declare_parameter("vx_limit", 0.6).value)
        self.vy_limit = float(self.declare_parameter("vy_limit", 0.4).value)
        self.wz_limit = float(self.declare_parameter("wz_limit", 0.8).value)
        self.max_linear_accel = float(self.declare_parameter("max_linear_accel", 1.8).value)
        self.max_angular_accel = float(self.declare_parameter("max_angular_accel", 1.0).value)
        self.disable_obstacle_avoidance = bool(
            self.declare_parameter("disable_obstacle_avoidance", True).value)
        self.auto_stand = bool(self.declare_parameter("auto_stand", True).value)
        self.dry_run = bool(self.declare_parameter("dry_run", False).value)
        ns = f"/bot{self.robot_id}"

        self.twist = Twist()
        self.twist_stamp = None
        self.slew = [0.0, 0.0, 0.0]
        self.estop = False
        self.moving = False
        self.lock = threading.Lock()

        self.create_subscription(Twist, ns + "/cmd_vel", self.on_twist, 10)
        self.create_subscription(Bool, ns + "/estop", self.on_estop, 10)
        self.state_pub = self.create_publisher(String, ns + "/sport_bridge_state", 10)

        self.sport = None
        if not self.dry_run:
            self._init_sdk()
        self.timer = self.create_timer(1.0 / self.rate_hz, self.tick)
        self.get_logger().info(
            "Go2 sport bridge bot%d iface=%s limits vx=%.2f vy=%.2f wz=%.2f accel=%.2f/%.2f dry_run=%s"
            % (self.robot_id, self.network_interface, self.vx_limit, self.vy_limit,
               self.wz_limit, self.max_linear_accel, self.max_angular_accel, self.dry_run))

    # ---- unitree_sdk2 ----
    def _init_sdk(self):
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        from unitree_sdk2py.go2.sport.sport_client import SportClient
        ChannelFactoryInitialize(self.dds_domain, self.network_interface)
        self.sport = SportClient()
        self.sport.SetTimeout(5.0)
        self.sport.Init()
        if self.disable_obstacle_avoidance:
            try:
                from unitree_sdk2py.go2.obstacles_avoid.obstacles_avoid_client import (
                    ObstaclesAvoidClient)
                oa = ObstaclesAvoidClient()
                oa.SetTimeout(3.0)
                oa.Init()
                oa.SwitchSet(False)
                self.get_logger().info("Unitree obstacle avoidance switched OFF")
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warning("could not switch obstacle avoidance off: %s" % exc)
        if self.auto_stand:
            self.sport.StandUp()
            time.sleep(1.5)
            self.sport.BalanceStand()
            time.sleep(0.5)
        self.get_logger().info("sport mode ready")

    # ---- callbacks ----
    def on_twist(self, msg):
        with self.lock:
            self.twist = msg
            self.twist_stamp = time.monotonic()

    def on_estop(self, msg):
        self.estop = bool(msg.data)
        if self.estop:
            self.get_logger().error("E-STOP received: StopMove + Damp")
            if self.sport is not None:
                try:
                    self.sport.StopMove()
                    self.sport.Damp()
                except Exception as exc:  # noqa: BLE001
                    self.get_logger().error("estop command failed: %s" % exc)

    def publish_state(self, text):
        msg = String()
        msg.data = text
        self.state_pub.publish(msg)

    # ---- control loop ----
    def tick(self):
        if self.estop:
            self.publish_state("estop")
            return
        now = time.monotonic()
        with self.lock:
            fresh = self.twist_stamp is not None and now - self.twist_stamp <= self.cmd_timeout
            vx = clamp(float(self.twist.linear.x), -self.vx_limit, self.vx_limit) if fresh else 0.0
            vy = clamp(float(self.twist.linear.y), -self.vy_limit, self.vy_limit) if fresh else 0.0
            wz = clamp(float(self.twist.angular.z), -self.wz_limit, self.wz_limit) if fresh else 0.0
        dt = 1.0 / self.rate_hz
        lin_step = self.max_linear_accel * dt if self.max_linear_accel > 0 else 1e9
        ang_step = self.max_angular_accel * dt if self.max_angular_accel > 0 else 1e9
        vx = clamp(vx, self.slew[0] - lin_step, self.slew[0] + lin_step)
        vy = clamp(vy, self.slew[1] - lin_step, self.slew[1] + lin_step)
        wz = clamp(wz, self.slew[2] - ang_step, self.slew[2] + ang_step)
        self.slew = [vx, vy, wz]
        active = math.hypot(vx, vy) > 0.02 or abs(wz) > 0.02
        if self.sport is not None:
            try:
                if active:
                    self.sport.Move(vx, vy, wz)
                    self.moving = True
                elif self.moving:
                    self.sport.StopMove()
                    self.moving = False
            except Exception as exc:  # noqa: BLE001
                self.get_logger().error("sport command failed: %s" % exc)
        self.publish_state("moving vx=%.2f vy=%.2f wz=%.2f" % (vx, vy, wz) if active
                           else ("idle" if fresh else "timeout"))


def main():
    rclpy.init()
    node = Go2SportBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception:  # noqa: BLE001 - SIGTERM: rclpy tears the context down under spin()
        if rclpy.ok():
            raise
    finally:
        if node.sport is not None:
            try:
                node.sport.StopMove()
            except Exception:  # noqa: BLE001
                pass
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:  # noqa: BLE001 - context may already be shut down (SIGTERM path)
            pass


if __name__ == "__main__":
    main()
