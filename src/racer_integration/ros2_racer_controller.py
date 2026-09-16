#!/usr/bin/env python3
"""Track RACER position commands with the Gazebo quadrotor velocity plugin.

Position feedback comes only from Swarm-LIO2 odometry in RACER's common world
frame. Body attitude comes from the onboard IMU, as it would in a normal flight
controller; Gazebo ground-truth pose is intentionally not subscribed.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist, TwistStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy._rclpy_pybind11 import RCLError
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import Imu, PointCloud2, PointField
from std_msgs.msg import Bool, String
from swarm_msgs.msg import ConnectedTeammateList

sys.path.insert(0, str(Path(__file__).resolve().parent))
from racer_control_math import (  # noqa: E402
    apply_obstacle_velocity_barrier,
    apply_separation_barrier,
    bootstrap_excitation_offset,
    calculate_world_velocity,
    clamp,
    ground_unicycle_command,
    limit_velocity_change,
    obstacle_braking_scale,
    position_inside_bounds,
    rotate_body_to_world,
    rotate_world_to_body,
    wrap_angle,
    yaw_from_quaternion,
)


class Channel:
    def __init__(self):
        self.odom: Optional[Odometry] = None
        self.odom_received = 0.0
        self.local_odom: Optional[Odometry] = None
        self.local_odom_received = 0.0
        self.local_origin = None
        self.imu: Optional[Imu] = None
        self.imu_received = 0.0
        self.lidar_points_body = None
        self.lidar_received = 0.0
        self.slam_valid: Optional[bool] = None
        self.slam_valid_received = 0.0
        self.target: Optional[PoseStamped] = None
        self.target_received = 0.0
        self.feedforward: Optional[TwistStamped] = None
        self.reference_yaw: Optional[float] = None
        self.last_velocity_world = (0.0, 0.0, 0.0)
        self.last_velocity_sim_time = 0.0
        self.cmd_pub = None
        self.arm_pub = None
        self.execution_blocked_pub = None
        self.barrier_active_since = 0.0
        self.execution_blocked = False
        self.last_blocked_publish_sim_time = -math.inf
        self.last_state = ""
        # Stuck-recovery bookkeeping (ground unicycle only).
        self.stuck_anchor_pos = None
        self.stuck_anchor_time = 0.0
        self.recovery_until = 0.0
        self.recovery_count = 0
        self.last_recovery_end = -math.inf


class RacerVelocityController(Node):
    def __init__(self, robot_ids: Iterable[int]):
        super().__init__("racer_velocity_controller")
        self.robot_ids = tuple(robot_ids)
        self.kp = self.declare_parameter("position_kp", 1.2).value
        self.yaw_kp = self.declare_parameter("yaw_kp", 1.0).value
        self.max_horizontal_speed = self.declare_parameter(
            "max_horizontal_speed", 1.0
        ).value
        self.max_vertical_speed = self.declare_parameter(
            "max_vertical_speed", 0.7
        ).value
        self.max_yaw_rate = self.declare_parameter("max_yaw_rate", 0.6).value
        self.max_command_acceleration = self.declare_parameter(
            "max_command_acceleration", 1.0
        ).value
        self.platform_mode = str(
            self.declare_parameter("platform_mode", "uav_holonomic").value
        ).strip().lower()
        if self.platform_mode not in (
            "uav_holonomic", "ground_unicycle", "ground_omni"
        ):
            raise ValueError(
                "platform_mode must be uav_holonomic, ground_unicycle or "
                "ground_omni"
            )
        self.ground_unicycle = self.platform_mode == "ground_unicycle"
        # ground_omni: same planar/ground gating, but the RL gait tracks body
        # (vx, vy, wz) directly, so no turn-in-place conversion is needed.
        self.ground_omni = self.platform_mode == "ground_omni"
        self.ground_robot = self.ground_unicycle or self.ground_omni
        self.ground_heading_kp = self.declare_parameter(
            "ground_heading_kp", 2.0
        ).value
        self.ground_turn_in_place_angle = math.radians(
            self.declare_parameter(
                "ground_turn_in_place_angle_deg", 55.0
            ).value
        )
        self.ground_obstacle_z_min = self.declare_parameter(
            "ground_obstacle_z_min", -0.20
        ).value
        self.ground_obstacle_z_max = self.declare_parameter(
            "ground_obstacle_z_max", 0.60
        ).value
        self.require_lidar_safety = self.declare_parameter(
            "require_lidar_safety", True
        ).value
        self.lidar_timeout = self.declare_parameter("lidar_timeout", 1.0).value
        self.lidar_point_stride = max(
            # Keep enough of the 180 x 96 scan that thin wall edges are not
            # skipped while approaching them diagonally.  Stride 8 allowed a
            # one-sample rotor-envelope overlap at a wall-top corner.
            1, int(self.declare_parameter("lidar_point_stride", 4).value)
        )
        # Optional mount rotation (roll, pitch, yaw in radians) from the LiDAR
        # frame to the body frame, applied before the z offset.  Zero for the
        # simulated trunk-mounted Mid-360; non-zero for the head-mounted,
        # forward-tilted Unitree L1 on the real Go2.
        self.lidar_to_body_rpy = [float(v) for v in self.declare_parameter(
            "lidar_to_body_rpy", [0.0, 0.0, 0.0]).value]
        self._lidar_mount_rotation = None
        if any(abs(v) > 1e-9 for v in self.lidar_to_body_rpy):
            r, p_, y = self.lidar_to_body_rpy
            cr, sr, cp, sp, cy, sy = (math.cos(r), math.sin(r), math.cos(p_),
                                      math.sin(p_), math.cos(y), math.sin(y))
            rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
            ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
            rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
            self._lidar_mount_rotation = (rz @ ry @ rx).astype(np.float32)
        self.lidar_to_body_z = self.declare_parameter(
            "lidar_to_body_z", 0.15
        ).value
        self.lidar_self_filter_radius = self.declare_parameter(
            "lidar_self_filter_radius", 0.40
        ).value
        self.lidar_self_filter_z_min = self.declare_parameter(
            # A Gazebo raw-scan probe places the quadrotor's own returns at
            # body-frame z=0.018..0.074 m.  The former symmetric +/-0.30 m
            # filter also deleted nearby floors and wall tops, creating a
            # close-range vertical blind zone.
            "lidar_self_filter_z_min", -0.01
        ).value
        self.lidar_self_filter_z_max = self.declare_parameter(
            "lidar_self_filter_z_max", 0.10
        ).value
        self.obstacle_corridor_radius = self.declare_parameter(
            # Physical rotor radius is 0.356 m.  The wider swept corridor adds
            # scan noise and tracking allowance without changing RACER's
            # planned trajectory or map inflation.
            "obstacle_corridor_radius", 0.70
        ).value
        self.obstacle_vertical_corridor_expansion = self.declare_parameter(
            "obstacle_vertical_corridor_expansion", 1.3
        ).value
        self.obstacle_hard_clearance = self.declare_parameter(
            "obstacle_hard_clearance", 0.60
        ).value
        self.obstacle_braking_margin = self.declare_parameter(
            "obstacle_braking_margin", 0.35
        ).value
        self.obstacle_deceleration = self.declare_parameter(
            "obstacle_deceleration", 1.0
        ).value
        self.obstacle_repulsion_activation = self.declare_parameter(
            # Begin the raw-scan recovery before a horizontal wall top enters
            # the Mid-360 downward blind cone.  At 1.6 m the controller could
            # reverse actual vertical momentum, but only after the conservative
            # rotor/body envelope had already grazed the wall top.
            "obstacle_repulsion_activation", 1.90
        ).value
        self.obstacle_repulsion_speed = self.declare_parameter(
            "obstacle_repulsion_speed", 1.2
        ).value
        self.attitude_timeout = self.declare_parameter(
            "attitude_timeout", 1.0
        ).value
        self.separation_activation_distance = self.declare_parameter(
            "separation_activation_distance", 4.8
        ).value
        self.separation_hard_distance = self.declare_parameter(
            # Swarm-LIO relative position error reached about 1.2 m in the
            # six-UAV Gazebo run.  Keep enough estimated-frame margin that
            # the 0.712 m physical rotor envelopes do not touch.
            "separation_hard_distance", 2.8
        ).value
        self.separation_repulsion_speed = self.declare_parameter(
            "separation_repulsion_speed", 1.0
        ).value
        # Stuck recovery: the planner works on a map built from SLAM poses
        # while the raw-LiDAR barrier sees the true geometry.  With 0.3-0.7 m
        # of pose error a path can hug a wall that the barrier will not let
        # the robot approach, and a forward-only unicycle then stands still
        # forever (v110: 300 s at one spot).  When RACER keeps commanding
        # motion but the robot has not moved stuck_recovery_displacement in
        # stuck_recovery_dwell seconds, back up along the body axis (the space
        # behind was just traversed) for stuck_recovery_duration seconds with
        # an alternating yaw bias, then resume tracking.
        self.stuck_recovery_dwell = self.declare_parameter(
            "stuck_recovery_dwell", 6.0
        ).value
        self.stuck_recovery_displacement = self.declare_parameter(
            "stuck_recovery_displacement", 0.4
        ).value
        self.stuck_recovery_duration = self.declare_parameter(
            "stuck_recovery_duration", 2.5
        ).value
        self.stuck_recovery_speed = self.declare_parameter(
            "stuck_recovery_speed", 0.35
        ).value
        self.stuck_recovery_yaw_rate = self.declare_parameter(
            "stuck_recovery_yaw_rate", 0.35
        ).value
        self.stuck_recovery_cooldown = self.declare_parameter(
            "stuck_recovery_cooldown", 4.0
        ).value
        self.stuck_recovery_min_command_speed = self.declare_parameter(
            "stuck_recovery_min_command_speed", 0.15
        ).value
        self.execution_blocked_dwell = self.declare_parameter(
            "execution_blocked_dwell", 2.0
        ).value
        self.track_racer_yaw = self.declare_parameter(
            "track_racer_yaw", False
        ).value
        self.command_timeout = self.declare_parameter("command_timeout", 0.5).value
        self.odom_timeout = self.declare_parameter("odom_timeout", 0.5).value
        self.bootstrap_enabled = self.declare_parameter(
            "bootstrap_enabled", True
        ).value
        self.require_safety_monitor = self.declare_parameter(
            "require_safety_monitor", False
        ).value
        self.bootstrap_hover_seconds = self.declare_parameter(
            "bootstrap_hover_seconds", 3.0
        ).value
        self.bootstrap_climb_seconds = self.declare_parameter(
            "bootstrap_climb_seconds", 4.0
        ).value
        self.bootstrap_loop_seconds = self.declare_parameter(
            "bootstrap_loop_seconds", 16.0
        ).value
        # Do not accept the trajectory-match quality gate before this much
        # excitation has been executed (0 = accept as soon as the graph is
        # connected, the original behaviour).  Yaw observability of the
        # cross-robot transform scales with the executed loop size, so a gate
        # reached 6 s into a 40 s loop latches a poor yaw.
        self.bootstrap_min_excitation_seconds = self.declare_parameter(
            "bootstrap_min_excitation_seconds", 0.0
        ).value
        self.bootstrap_settle_seconds = self.declare_parameter(
            "bootstrap_settle_seconds", 5.0
        ).value
        self.bootstrap_radius = self.declare_parameter(
            "bootstrap_radius", 0.8
        ).value
        self.bootstrap_radius_step = self.declare_parameter(
            # Movers are excited sequentially, so they do not need shrinking
            # circles for collision avoidance.  Keeping the full radius also
            # keeps Swarm-LIO2's trajectory covariance above its SE(3)
            # excitation gate for bot3..bot6.
            "bootstrap_radius_step", 0.0
        ).value
        self.bootstrap_ground_lateral_scale = self.declare_parameter(
            "bootstrap_ground_lateral_scale", 0.25
        ).value
        if self.bootstrap_ground_lateral_scale <= 0.0:
            raise ValueError("bootstrap_ground_lateral_scale must be positive")
        self.bootstrap_climb_height = self.declare_parameter(
            "bootstrap_climb_height", 0.9
        ).value
        self.bootstrap_vertical_amplitude = self.declare_parameter(
            "bootstrap_vertical_amplitude", 0.3
        ).value
        self.bootstrap_farthest_first = self.declare_parameter(
            "bootstrap_farthest_first", True
        ).value
        self.bootstrap_include_root = self.declare_parameter(
            "bootstrap_include_root", True
        ).value
        self.bootstrap_root_only = self.declare_parameter(
            # A root-centred, line-of-sight start layout lets every non-root
            # vehicle estimate one direct edge to the same moving root.  This
            # avoids accumulating trajectory-alignment error along the
            # spanning tree produced by Swarm-LIO2's infection model.
            "bootstrap_root_only", False
        ).value
        non_root_movers = self.robot_ids[1:]
        non_root_order = (
            tuple(reversed(non_root_movers))
            if self.bootstrap_farthest_first
            else tuple(non_root_movers)
        )
        # A merely connected N-1 edge graph can accumulate a large rotation
        # error along a chain.  Excite the root before infection has created
        # indirect trackers: once a tracker exists, Swarm-LIO2 intentionally
        # skips a second direct trajectory match.  Moving the root first lets
        # several stationary neighbors independently estimate root edges.
        if self.bootstrap_root_only and len(self.robot_ids) > 1:
            bootstrap_movers = (self.robot_ids[0],)
        else:
            bootstrap_movers = (
                (self.robot_ids[0],) + non_root_order
                if self.bootstrap_include_root and len(self.robot_ids) > 1
                else non_root_order
            )
        self.bootstrap_movers = bootstrap_movers
        default_directions = [
            360.0 * index / max(1, len(self.bootstrap_movers))
            for index in range(len(self.bootstrap_movers))
        ]
        self.bootstrap_radii = tuple(
            float(value)
            for value in self.declare_parameter(
                "bootstrap_radii",
                [float(self.bootstrap_radius)] * len(self.bootstrap_movers),
            ).value
        )
        self.bootstrap_directions_rad = tuple(
            math.radians(float(value))
            for value in self.declare_parameter(
                "bootstrap_directions_deg", default_directions
            ).value
        )
        if (
            len(self.bootstrap_radii) < len(self.bootstrap_movers)
            or len(self.bootstrap_directions_rad) < len(self.bootstrap_movers)
        ):
            raise ValueError(
                "bootstrap_radii and bootstrap_directions_deg must provide "
                "one value per bootstrap mover"
            )
        self.bootstrap_min_direct_edges = int(
            self.declare_parameter(
                "bootstrap_min_direct_edges",
                max(0, len(self.robot_ids) - 1),
            ).value
        )
        self.bootstrap_min_root_degree = int(
            self.declare_parameter(
                "bootstrap_min_root_degree",
                (
                    max(0, len(self.robot_ids) - 1)
                    if self.bootstrap_root_only
                    else min(2, max(0, len(self.robot_ids) - 1))
                ),
            ).value
        )
        self.bootstrap_started = None
        self.bootstrap_quality_ready_since = None
        self.bootstrap_complete = not self.bootstrap_enabled
        self.bootstrap_wait_logged = False
        self.trajectory_matches_by_robot = {
            robot_id: set() for robot_id in self.robot_ids
        }
        self.last_match_graph = None
        self.emergency_stop = False
        self.safety_monitor_ready = not self.require_safety_monitor
        self.bounds_lower = tuple(
            self.declare_parameter(
                "bounds_lower", [-25.0, -25.0, 0.0]
            ).value
        )
        self.bounds_upper = tuple(
            self.declare_parameter(
                "bounds_upper", [25.0, 25.0, 50.0]
            ).value
        )
        self.channels: Dict[int, Channel] = {}
        self.subscription_handles = []
        latched_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.status_pub = self.create_publisher(String, "/racer/controller/status", 20)
        self.bootstrap_complete_pub = self.create_publisher(
            Bool, "/racer/bootstrap_complete", latched_qos
        )
        self.subscription_handles.append(
            self.create_subscription(
                Bool, "/racer/emergency_stop", self.on_emergency_stop, 10
            )
        )
        self.subscription_handles.append(
            self.create_subscription(
                Bool,
                "/racer/safety_monitor_ready",
                self.on_safety_monitor_ready,
                10,
            )
        )
        for robot_id in self.robot_ids:
            self.subscription_handles.append(
                self.create_subscription(
                    ConnectedTeammateList,
                    f"/bot{robot_id}/teammate_id_with_traj_matching",
                    self.on_trajectory_matches,
                    20,
                )
            )

        for robot_id in self.robot_ids:
            channel = Channel()
            self.channels[robot_id] = channel
            racer_base = f"/racer/bot{robot_id}"
            robot_base = f"/bot{robot_id}"
            channel.cmd_pub = self.create_publisher(Twist, robot_base + "/cmd_vel", 20)
            channel.arm_pub = self.create_publisher(Bool, robot_base + "/enable", 10)
            channel.execution_blocked_pub = self.create_publisher(
                Bool, racer_base + "/execution_blocked", 10
            )
            self.subscription_handles.extend(
                [
                    self.create_subscription(
                        Odometry,
                        racer_base + "/odom_world",
                        lambda message, rid=robot_id: self.on_odom(rid, message),
                        20,
                    ),
                    self.create_subscription(
                        Odometry,
                        robot_base + "/lidar_slam/odom",
                        lambda message, rid=robot_id: self.on_local_odom(
                            rid, message
                        ),
                        20,
                    ),
                    self.create_subscription(
                        Imu,
                        robot_base + "/imu",
                        lambda message, rid=robot_id: self.on_imu(rid, message),
                        50,
                    ),
                    self.create_subscription(
                        PointCloud2,
                        robot_base + "/lidar_points/points",
                        lambda message, rid=robot_id: self.on_lidar(
                            rid, message
                        ),
                        sensor_qos,
                    ),
                    self.create_subscription(
                        Bool,
                        racer_base + "/slam_valid",
                        lambda message, rid=robot_id: self.on_slam_valid(
                            rid, message
                        ),
                        20,
                    ),
                    self.create_subscription(
                        PoseStamped,
                        racer_base + "/position_setpoint",
                        lambda message, rid=robot_id: self.on_target(rid, message),
                        20,
                    ),
                    self.create_subscription(
                        TwistStamped,
                        racer_base + "/feedforward",
                        lambda message, rid=robot_id: self.on_feedforward(rid, message),
                        20,
                    ),
                ]
            )

        self.timer = self.create_timer(0.02, self.control_tick)
        self.get_logger().info(
            "RACER controller configured for %s %s; position=Swarm-LIO2, "
            "attitude=onboard IMU, horizontal<=%.2fm/s vertical<=%.2fm/s, "
            "acceleration<=%.2fm/s^2, separation=%.2f/%.2fm, "
            "raw-LiDAR brake=%s, yaw=%s"
            % (
                "ground robots" if self.ground_robot else "UAVs",
                self.robot_ids,
                self.max_horizontal_speed,
                self.max_vertical_speed,
                self.max_command_acceleration,
                self.separation_activation_distance,
                self.separation_hard_distance,
                "on" if self.require_lidar_safety else "off",
                (
                    "velocity heading"
                    if self.ground_robot
                    else "RACER trajectory"
                    if self.track_racer_yaw
                    else "initial heading hold"
                ),
            )
        )

    def now_monotonic(self) -> float:
        return time.monotonic()

    def now_simulation(self) -> float:
        """Return ROS simulation time in seconds.

        Freshness watchdogs deliberately use monotonic wall time, while the
        bootstrap motion must use simulation time so a low Gazebo real-time
        factor does not shorten the physical excitation.
        """
        return self.get_clock().now().nanoseconds * 1.0e-9

    def on_emergency_stop(self, message: Bool):
        self.emergency_stop = self.emergency_stop or bool(message.data)
        if self.emergency_stop:
            self.get_logger().error(
                "simulation safety monitor requested emergency hover",
                throttle_duration_sec=1.0,
            )

    def on_safety_monitor_ready(self, message: Bool):
        self.safety_monitor_ready = self.safety_monitor_ready or bool(message.data)

    def on_trajectory_matches(self, message: ConnectedTeammateList):
        source = int(message.drone_id)
        if source not in self.trajectory_matches_by_robot:
            return
        current = {
            int(robot_id)
            for robot_id in message.connected_teammate_id
            if int(robot_id) in self.trajectory_matches_by_robot
            and int(robot_id) != source
        }
        if current != self.trajectory_matches_by_robot[source]:
            self.trajectory_matches_by_robot[source] = current
            connected = self.trajectory_match_connected_component()
            edges = self.trajectory_match_edges()
            graph_state = (tuple(sorted(edges)), tuple(sorted(connected)))
            if graph_state == self.last_match_graph:
                return
            self.last_match_graph = graph_state
            self.get_logger().info(
                "trajectory-match graph: %d direct edges %s, root component %s"
                % (
                    len(edges),
                    tuple(sorted(edges)),
                    tuple(sorted(connected)),
                )
            )

    def trajectory_match_edges(self):
        edges = set()
        for source, teammates in self.trajectory_matches_by_robot.items():
            for target in teammates:
                edges.add(tuple(sorted((source, target))))
        return edges

    def trajectory_match_connected_component(self):
        edges = self.trajectory_match_edges()
        connected = {self.robot_ids[0]}
        changed = True
        while changed:
            changed = False
            for first, second in edges:
                if first in connected and second not in connected:
                    connected.add(second)
                    changed = True
                elif second in connected and first not in connected:
                    connected.add(first)
                    changed = True
        return connected

    def on_odom(self, robot_id: int, message: Odometry):
        channel = self.channels[robot_id]
        channel.odom = message
        channel.odom_received = self.now_monotonic()

    def on_local_odom(self, robot_id: int, message: Odometry):
        channel = self.channels[robot_id]
        channel.local_odom = message
        channel.local_odom_received = self.now_monotonic()
        if channel.local_origin is None:
            position = message.pose.pose.position
            orientation = message.pose.pose.orientation
            channel.local_origin = (
                position.x,
                position.y,
                position.z,
                yaw_from_quaternion(
                    (
                        orientation.x,
                        orientation.y,
                        orientation.z,
                        orientation.w,
                    )
                ),
            )

    def on_imu(self, robot_id: int, message: Imu):
        channel = self.channels[robot_id]
        channel.imu = message
        channel.imu_received = self.now_monotonic()
        if channel.reference_yaw is None:
            orientation = message.orientation
            channel.reference_yaw = yaw_from_quaternion(
                (
                    orientation.x,
                    orientation.y,
                    orientation.z,
                    orientation.w,
                )
            )

    def on_slam_valid(self, robot_id: int, message: Bool):
        channel = self.channels[robot_id]
        channel.slam_valid = bool(message.data)
        channel.slam_valid_received = self.now_monotonic()

    def on_lidar(self, robot_id: int, message: PointCloud2):
        """Keep a bounded, downsampled raw scan in the body-aligned frame."""
        if message.point_step <= 0:
            return
        field_by_name = {field.name: field for field in message.fields}
        xyz_fields = [field_by_name.get(axis) for axis in ("x", "y", "z")]
        if any(
            field is None or field.datatype != PointField.FLOAT32
            for field in xyz_fields
        ):
            self.get_logger().error(
                "bot%d raw LiDAR has unsupported XYZ fields" % robot_id,
                throttle_duration_sec=5.0,
            )
            return
        count = int(message.width) * int(message.height)
        if count <= 0:
            return
        endian = ">" if message.is_bigendian else "<"
        dtype = np.dtype(
            {
                "names": ("x", "y", "z"),
                "formats": (endian + "f4",) * 3,
                "offsets": tuple(field.offset for field in xyz_fields),
                "itemsize": int(message.point_step),
            }
        )
        try:
            values = np.frombuffer(message.data, dtype=dtype, count=count)[
                :: self.lidar_point_stride
            ]
        except (TypeError, ValueError):
            return
        points = np.column_stack((values["x"], values["y"], values["z"])).astype(
            np.float32, copy=False
        )
        points = points[np.isfinite(points).all(axis=1)]
        if not len(points):
            return
        # lidar_link is aligned with base_link and mounted 0.15 m above it.
        points = points.copy()
        if self._lidar_mount_rotation is not None:
            points = points @ self._lidar_mount_rotation.T
        points[:, 2] += float(self.lidar_to_body_z)
        # Gazebo's GPU lidar sees the X3 body and rotor collision geometry.
        # Remove only the physical UAV envelope; otherwise those returns at
        # 0--0.25 m permanently activate the emergency brake.  An external
        # return inside this envelope already overlaps the vehicle, so it
        # cannot provide a useful pre-collision braking measurement.
        horizontal_sq = points[:, 0] ** 2 + points[:, 1] ** 2
        self_return = np.logical_and(
            horizontal_sq
            <= float(self.lidar_self_filter_radius)
            * float(self.lidar_self_filter_radius),
            np.logical_and(
                points[:, 2] >= float(self.lidar_self_filter_z_min),
                points[:, 2] <= float(self.lidar_self_filter_z_max),
            ),
        )
        points = points[np.logical_not(self_return)]
        if self.ground_robot:
            # A ground-mounted 3-D lidar always sees the floor at short range.
            # Floor returns are support geometry, not forward obstacles; if
            # retained they keep the generic 3-D recovery barrier permanently
            # active. Keep the vertical band that intersects the quadruped
            # body/wheel swept envelope and reject the horizontal floor.
            points = points[
                np.logical_and(
                    points[:, 2] >= float(self.ground_obstacle_z_min),
                    points[:, 2] <= float(self.ground_obstacle_z_max),
                )
            ]
        ranges_sq = np.einsum("ij,ij->i", points, points)
        points = points[ranges_sq >= 0.20 * 0.20]
        channel = self.channels[robot_id]
        channel.lidar_points_body = points
        channel.lidar_received = self.now_monotonic()

    def lidar_braking_scale(self, channel: Channel, velocity_body):
        points = channel.lidar_points_body
        speed = math.sqrt(
            sum(component * component for component in velocity_body)
        )
        if points is None or not len(points) or speed <= 1.0e-6:
            return 1.0, None, None, None
        direction = np.asarray(velocity_body, dtype=np.float32) / speed
        longitudinal = points @ direction
        distance_sq = np.einsum("ij,ij->i", points, points)
        lateral_sq = np.maximum(
            0.0, distance_sq - longitudinal * longitudinal
        )
        # Mid-360 sees down to about -40 degrees rather than straight down.
        # Widen the swept cylinder as the commanded/measured direction becomes
        # vertical, so an oblique return from a horizontal wall top represents
        # its true vertical clearance instead of falling outside a narrow
        # body-radius corridor.
        corridor_radius = (
            float(self.obstacle_corridor_radius)
            + float(self.obstacle_vertical_corridor_expansion)
            * abs(float(direction[2]))
        )
        swept = np.logical_and(
            longitudinal > 0.0,
            lateral_sq
            <= corridor_radius * corridor_radius,
        )
        if not np.any(swept):
            return 1.0, None, None, None
        swept_indices = np.flatnonzero(swept)
        closest_index = int(
            swept_indices[np.argmin(longitudinal[swept_indices])]
        )
        clearance = float(longitudinal[closest_index])
        scale = obstacle_braking_scale(
            speed,
            clearance,
            self.obstacle_hard_clearance,
            self.obstacle_braking_margin,
            self.obstacle_deceleration,
        )
        # The collision distance is measured along the swept velocity axis.
        # Return the equivalent point on that axis rather than the oblique
        # LiDAR ray. This is essential for Mid-360's downward blind cone:
        # otherwise a wall-top return spreads a required vertical braking
        # command into an unrelated horizontal component.
        projected_obstacle = direction * clearance
        raw_obstacle = points[closest_index]
        raw_range = math.sqrt(float(np.dot(raw_obstacle, raw_obstacle)))
        escape_obstacle = (
            raw_obstacle / raw_range * clearance
            if raw_range > 1.0e-6
            else projected_obstacle
        )
        return scale, clearance, projected_obstacle, escape_obstacle

    def lidar_velocity_barrier(
        self, channel: Channel, nominal_body, measured_body
    ):
        """Fuse mapped planning with raw-scan stopping and active recovery."""
        points = channel.lidar_points_body
        if points is None or not len(points):
            return tuple(nominal_body), False, None

        (
            nominal_scale,
            nominal_clearance,
            nominal_point,
            nominal_escape_point,
        ) = (
            self.lidar_braking_scale(channel, nominal_body)
        )
        (
            measured_scale,
            measured_clearance,
            measured_point,
            measured_escape_point,
        ) = (
            self.lidar_braking_scale(channel, measured_body)
        )
        risk_points = []
        # Side-wall returns inside the soft repulsion band are normal for a
        # differential-drive platform in a corridor.  A unicycle cannot
        # execute the lateral component of radial repulsion, so feeding these
        # points to the generic 3-D barrier produces alternating steer/stop
        # commands.  For ground robots, retain radial escape only inside the
        # protected envelope; the swept-corridor scale still brakes any
        # commanded or measured velocity directed toward a wall.
        escape_recovery_distance = (
            float(self.obstacle_hard_clearance)
            + float(self.obstacle_braking_margin)
            if self.ground_robot
            else self.obstacle_repulsion_activation
        )
        if nominal_point is not None and nominal_scale < 0.999:
            risk_points.append(tuple(float(value) for value in nominal_point))
        if (
            nominal_escape_point is not None
            and nominal_clearance < escape_recovery_distance
        ):
            risk_points.append(
                tuple(float(value) for value in nominal_escape_point)
            )
        if measured_point is not None and measured_scale < 0.999:
            risk_points.append(tuple(float(value) for value in measured_point))
        if (
            measured_escape_point is not None
            and measured_clearance < escape_recovery_distance
        ):
            risk_points.append(
                tuple(float(value) for value in measured_escape_point)
            )

        ranges_sq = np.einsum("ij,ij->i", points, points)
        nearest_index = int(np.argmin(ranges_sq))
        nearest_range = math.sqrt(float(ranges_sq[nearest_index]))
        protected_clearance = escape_recovery_distance if self.ground_robot else (
            float(self.obstacle_hard_clearance)
            + float(self.obstacle_braking_margin)
        )
        # A ground unicycle cannot execute the lateral velocity generated by
        # generic radial repulsion.  Repelling from every nearby side-wall
        # return therefore makes it steer back and forth even while the
        # requested and measured velocities are safely parallel to the wall.
        # Directional swept-corridor checks above still brake any motion toward
        # a wall.  Keep unconditional nearest-point recovery only inside the
        # protected envelope for ground robots; UAVs retain the 3-D behavior.
        # A doorway can be physically traversable even when both jambs are
        # inside the longitudinal braking envelope.  For a ground unicycle,
        # use the physical hard radius for direction-independent recovery;
        # the swept-corridor checks above still use the larger protected
        # stopping distance for any point actually in the velocity corridor.
        nearest_recovery_distance = (
            float(self.obstacle_hard_clearance)
            if self.ground_robot
            else self.obstacle_repulsion_activation
        )
        if nearest_range < nearest_recovery_distance:
            risk_points.append(
                tuple(float(value) for value in points[nearest_index])
            )

        if not risk_points:
            return tuple(nominal_body), False, nearest_range
        safe_body = apply_obstacle_velocity_barrier(
            nominal_body,
            measured_body,
            risk_points,
            protected_clearance,
            self.obstacle_repulsion_activation,
            self.obstacle_repulsion_speed,
            self.obstacle_deceleration,
        )
        limited = math.sqrt(
            sum(
                (safe_body[index] - float(nominal_body[index])) ** 2
                for index in range(3)
            )
        ) > 1.0e-4
        corridor_clearances = [
            value
            for value in (nominal_clearance, measured_clearance)
            if value is not None
        ]
        diagnostic_clearance = min(
            [nearest_range] + corridor_clearances
        )
        return safe_body, limited, diagnostic_clearance

    def on_target(self, robot_id: int, message: PoseStamped):
        p = message.pose.position
        if not position_inside_bounds(
            (p.x, p.y, p.z), self.bounds_lower, self.bounds_upper
        ):
            self.get_logger().error(
                "bot%d rejected out-of-bounds RACER target (%.2f, %.2f, %.2f)"
                % (robot_id, p.x, p.y, p.z),
                throttle_duration_sec=1.0,
            )
            return
        channel = self.channels[robot_id]
        channel.target = message
        channel.target_received = self.now_monotonic()

    def on_feedforward(self, robot_id: int, message: TwistStamped):
        self.channels[robot_id].feedforward = message

    def publish_state(self, robot_id: int, state: str):
        channel = self.channels[robot_id]
        if state == channel.last_state:
            return
        channel.last_state = state
        message = String()
        message.data = f"bot{robot_id}:{state}"
        self.status_pub.publish(message)
        self.get_logger().info(message.data)

    @staticmethod
    def publish_hover(channel: Channel, command: Twist, sim_now: float):
        """Publish zero velocity and reset the acceleration-limiter state."""
        channel.last_velocity_world = (0.0, 0.0, 0.0)
        channel.last_velocity_sim_time = sim_now
        channel.barrier_active_since = 0.0
        if channel.execution_blocked:
            channel.execution_blocked = False
            channel.execution_blocked_pub.publish(Bool(data=False))
        channel.last_blocked_publish_sim_time = sim_now
        channel.cmd_pub.publish(command)

    def update_execution_blocked(
        self, channel: Channel, barrier_limited: bool, sim_now: float
    ) -> bool:
        if barrier_limited:
            if channel.barrier_active_since <= 0.0:
                channel.barrier_active_since = sim_now
            blocked = (
                sim_now - channel.barrier_active_since
                >= self.execution_blocked_dwell
            )
        else:
            channel.barrier_active_since = 0.0
            blocked = False
        if (
            blocked != channel.execution_blocked
            or sim_now - channel.last_blocked_publish_sim_time >= 0.5
        ):
            channel.execution_blocked = blocked
            channel.last_blocked_publish_sim_time = sim_now
            channel.execution_blocked_pub.publish(Bool(data=blocked))
        return blocked

    def stuck_recovery(self, channel: Channel, robot_id: int, current, nominal_world, sim_now: float):
        """Detect a standing robot that RACER keeps commanding and back it up.

        Returns (recovering, command).  While recovering, the returned Twist
        is the reverse command (already passed through the raw-LiDAR
        barrier); otherwise the caller continues with normal tracking."""
        command = Twist()
        planar = (float(current[0]), float(current[1]))
        if channel.stuck_anchor_pos is None or channel.stuck_anchor_time <= 0.0:
            channel.stuck_anchor_pos = planar
            channel.stuck_anchor_time = sim_now
        elif (
            math.hypot(planar[0] - channel.stuck_anchor_pos[0], planar[1] - channel.stuck_anchor_pos[1])
            > self.stuck_recovery_displacement
        ):
            channel.stuck_anchor_pos = planar
            channel.stuck_anchor_time = sim_now
        commanded_speed = math.hypot(float(nominal_world[0]), float(nominal_world[1]))
        if sim_now >= channel.recovery_until:
            if channel.recovery_until > 0.0 and channel.last_recovery_end < channel.recovery_until:
                channel.last_recovery_end = sim_now
                channel.stuck_anchor_pos = planar
                channel.stuck_anchor_time = sim_now
            stuck_for = sim_now - channel.stuck_anchor_time
            if (
                commanded_speed >= self.stuck_recovery_min_command_speed
                and stuck_for >= self.stuck_recovery_dwell
                and sim_now - channel.last_recovery_end >= self.stuck_recovery_cooldown
            ):
                channel.recovery_count += 1
                channel.recovery_until = sim_now + self.stuck_recovery_duration
                self.get_logger().warning(
                    "bot%d stuck recovery #%d: no displacement >%.2fm for %.1fs "
                    "while commanded %.2fm/s; reversing %.1fs"
                    % (
                        robot_id,
                        channel.recovery_count,
                        self.stuck_recovery_displacement,
                        stuck_for,
                        commanded_speed,
                        self.stuck_recovery_duration,
                    )
                )
            else:
                return False, command
        # Reverse along the body axis, protected by the raw-scan barrier.
        attitude = channel.imu.orientation
        attitude_quaternion = (attitude.x, attitude.y, attitude.z, attitude.w)
        measured_linear = channel.odom.twist.twist.linear
        measured_body = rotate_world_to_body(
            (measured_linear.x, measured_linear.y, measured_linear.z), attitude_quaternion
        )
        reverse_body = (-float(self.stuck_recovery_speed), 0.0, 0.0)
        safe_body, _limited, _clearance = self.lidar_velocity_barrier(
            channel, reverse_body, measured_body
        )
        forward = clamp(float(safe_body[0]), -float(self.stuck_recovery_speed), 0.0)
        command.linear.x = forward
        command.linear.y = 0.0
        command.linear.z = 0.0
        command.angular.z = (
            float(self.stuck_recovery_yaw_rate)
            if channel.recovery_count % 2 == 1
            else -float(self.stuck_recovery_yaw_rate)
        )
        channel.last_velocity_world = rotate_body_to_world(
            (forward, 0.0, 0.0), attitude_quaternion
        )
        channel.last_velocity_sim_time = sim_now
        return True, command

    def ground_command_from_world_velocity(
        self, velocity_world, current_yaw: float
    ):
        """Project a planar world velocity onto a forward-only unicycle."""
        return ground_unicycle_command(
            velocity_world,
            current_yaw,
            self.max_horizontal_speed,
            self.max_yaw_rate,
            self.ground_heading_kp,
            self.ground_turn_in_place_angle,
        )

    def control_tick(self):
        now = self.now_monotonic()
        sim_now = self.now_simulation()
        common_positions = {}
        for robot_id, channel in self.channels.items():
            if (
                channel.odom is not None
                and now - channel.odom_received <= self.odom_timeout
            ):
                position = channel.odom.pose.pose.position
                common_positions[robot_id] = (
                    position.x,
                    position.y,
                    position.z,
                )
        local_odometry_ready = all(
            channel.local_odom is not None
            and now - channel.local_odom_received <= self.odom_timeout
            for channel in self.channels.values()
        )
        if (
            self.bootstrap_enabled
            and self.bootstrap_started is None
            and local_odometry_ready
            and self.safety_monitor_ready
        ):
            self.bootstrap_started = sim_now
            self.get_logger().info(
                "All local Swarm-LIO2 odometry streams ready; starting "
                "ground-truth-free 3D common-frame initialization on "
                "simulation time; "
                "excitation order=%s" % (self.bootstrap_movers,)
            )

        bootstrap_in_progress = False
        if self.bootstrap_enabled and self.bootstrap_started is not None:
            elapsed = max(0.0, sim_now - self.bootstrap_started)
            slot_duration = (
                2.0 * self.bootstrap_climb_seconds
                + self.bootstrap_loop_seconds
            )
            minimum_duration = (
                self.bootstrap_hover_seconds
                + len(self.bootstrap_movers) * slot_duration
                + self.bootstrap_settle_seconds
            )
            connected_robots = self.trajectory_match_connected_component()
            edges = self.trajectory_match_edges()
            root_id = self.robot_ids[0]
            root_degree = sum(root_id in edge for edge in edges)
            matches_ready = (
                connected_robots >= set(self.robot_ids)
                and len(edges) >= self.bootstrap_min_direct_edges
                and root_degree >= self.bootstrap_min_root_degree
            )
            if matches_ready and elapsed < self.bootstrap_min_excitation_seconds:
                matches_ready = False
            if matches_ready:
                if self.bootstrap_quality_ready_since is None:
                    self.bootstrap_quality_ready_since = sim_now
                    self.get_logger().info(
                        "trajectory-match quality gate reached; returning all "
                        "UAVs to their local origins for %.1f simulation seconds"
                        % (self.bootstrap_settle_seconds,)
                    )
                if (
                    sim_now - self.bootstrap_quality_ready_since
                    >= self.bootstrap_settle_seconds
                ):
                    if not self.bootstrap_complete:
                        self.bootstrap_complete = True
                        self.get_logger().info(
                            "3D common-frame initialization complete after %.1f "
                            "simulation seconds; the %d-edge direct "
                            "trajectory-match graph connects all %d UAVs"
                            % (
                                elapsed,
                                len(edges),
                                len(self.robot_ids),
                            )
                        )
                else:
                    bootstrap_in_progress = True
            else:
                self.bootstrap_quality_ready_since = None
                bootstrap_in_progress = True
                if elapsed >= minimum_duration and not self.bootstrap_wait_logged:
                    missing = tuple(sorted(set(self.robot_ids) - connected_robots))
                    self.get_logger().warning(
                        "minimum 3D excitation complete, but the trajectory-"
                        "match graph is below the quality gate: missing=%s, "
                        "edges=%d/%d, root_degree=%d/%d; repeating the closed "
                        "excitation schedule"
                        % (
                            missing,
                            len(edges),
                            self.bootstrap_min_direct_edges,
                            root_degree,
                            self.bootstrap_min_root_degree,
                        )
                    )
                    self.bootstrap_wait_logged = True

        self.bootstrap_complete_pub.publish(
            Bool(data=bool(self.bootstrap_complete))
        )

        for robot_id, channel in self.channels.items():
            command = Twist()
            # The Gazebo velocity plugin must be enabled even before SLAM
            # publishes.  Otherwise mid-air spawn poses free-fall during the
            # Swarm-LIO2 build/gravity-initialization interval.
            channel.arm_pub.publish(Bool(data=True))
            if self.emergency_stop:
                self.publish_hover(channel, command, sim_now)
                self.publish_state(robot_id, "emergency_stop_hover")
                continue
            if bootstrap_in_progress:
                if (
                    channel.local_odom is None
                    or now - channel.local_odom_received > self.odom_timeout
                ):
                    self.publish_state(
                        robot_id, "failsafe_stale_local_odometry_hover"
                    )
                    self.publish_hover(channel, command, sim_now)
                    continue
                if self.require_lidar_safety and (
                    channel.lidar_points_body is None
                    or now - channel.lidar_received > self.lidar_timeout
                ):
                    self.publish_state(
                        robot_id, "failsafe_stale_lidar_hover"
                    )
                    self.publish_hover(channel, command, sim_now)
                    continue
                self.publish_bootstrap_command(
                    robot_id, channel, sim_now, command
                )
                continue
            if channel.odom is None:
                if (
                    not self.bootstrap_enabled
                    or channel.local_odom is None
                    or now - channel.local_odom_received > self.odom_timeout
                ):
                    self.publish_state(
                        robot_id, "waiting_for_local_slam_odometry"
                    )
                    self.publish_hover(channel, command, sim_now)
                    continue

                # Bootstrap uses only each UAV's own Swarm-LIO2 local frame.
                # It arms/holds the vehicle while cross-world trajectory
                # matching is pending, avoiding a ready->controller deadlock.
                if not self.safety_monitor_ready:
                    self.publish_state(
                        robot_id, "armed_hover_waiting_for_safety_monitor"
                    )
                    self.publish_hover(channel, command, sim_now)
                    continue
                if not local_odometry_ready or self.bootstrap_started is None:
                    self.publish_state(
                        robot_id, "armed_hover_waiting_for_all_local_odometry"
                    )
                    self.publish_hover(channel, command, sim_now)
                    continue
                self.publish_state(
                    robot_id, "armed_hover_waiting_for_common_frame_latch"
                )
                self.publish_hover(channel, command, sim_now)
                continue

            # Once LIO is alive, repeatedly arm. Zero velocity is the safe
            # hover command both before the first plan and on command timeout.
            if now - channel.odom_received > self.odom_timeout:
                self.publish_state(robot_id, "failsafe_stale_odometry_hover")
                self.publish_hover(channel, command, sim_now)
                continue
            if (
                channel.imu is None
                or now - channel.imu_received > self.attitude_timeout
            ):
                self.publish_state(robot_id, "failsafe_stale_imu_hover")
                self.publish_hover(channel, command, sim_now)
                continue
            if (
                channel.slam_valid is not True
                or now - channel.slam_valid_received > self.odom_timeout
            ):
                self.publish_state(robot_id, "failsafe_invalid_slam_hover")
                self.publish_hover(channel, command, sim_now)
                continue
            if self.require_lidar_safety and (
                channel.lidar_points_body is None
                or now - channel.lidar_received > self.lidar_timeout
            ):
                self.publish_state(robot_id, "failsafe_stale_lidar_hover")
                self.publish_hover(channel, command, sim_now)
                continue
            if (
                channel.target is None
                or now - channel.target_received > self.command_timeout
            ):
                self.publish_state(robot_id, "armed_hover_waiting_for_racer")
                self.publish_hover(channel, command, sim_now)
                continue

            odom_pose = channel.odom.pose.pose
            target_pose = channel.target.pose
            current = (
                odom_pose.position.x,
                odom_pose.position.y,
                odom_pose.position.z,
            )
            target = (
                target_pose.position.x,
                target_pose.position.y,
                target_pose.position.z,
            )
            if self.ground_robot:
                # RACER remains the task allocator and trajectory generator,
                # but a quadruped cannot follow vertical setpoints.
                target = (target[0], target[1], current[2])
            feedforward = (0.0, 0.0, 0.0)
            yaw_rate_ff = 0.0
            if channel.feedforward is not None:
                linear = channel.feedforward.twist.linear
                feedforward = (linear.x, linear.y, linear.z)
                yaw_rate_ff = channel.feedforward.twist.angular.z
            if self.ground_robot:
                feedforward = (feedforward[0], feedforward[1], 0.0)
            velocity_world = calculate_world_velocity(
                current,
                target,
                feedforward,
                self.kp,
                self.max_horizontal_speed,
                self.max_vertical_speed,
            )
            if self.ground_robot and self.stuck_recovery_dwell > 0.0:
                recovering, recovery_command = self.stuck_recovery(
                    channel, robot_id, current, velocity_world, sim_now
                )
                if recovering:
                    channel.cmd_pub.publish(recovery_command)
                    self.publish_state(robot_id, "stuck_recovery_reverse")
                    continue
            velocity_before_separation = velocity_world
            velocity_world = apply_separation_barrier(
                current,
                velocity_world,
                (
                    position
                    for rid, position in common_positions.items()
                    if rid != robot_id
                ),
                self.separation_activation_distance,
                self.separation_hard_distance,
                self.separation_repulsion_speed,
                self.max_command_acceleration,
            )
            separation_limited = math.sqrt(
                sum(
                    (
                        float(velocity_world[index])
                        - float(velocity_before_separation[index])
                    )
                    ** 2
                    for index in range(3)
                )
            ) > 1.0e-4
            # Several simultaneous neighbors can add repulsion; re-apply the
            # flight envelope after the separation barrier.
            velocity_world = calculate_world_velocity(
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                velocity_world,
                0.0,
                self.max_horizontal_speed,
                self.max_vertical_speed,
            )
            control_dt = sim_now - channel.last_velocity_sim_time
            if channel.last_velocity_sim_time <= 0.0 or control_dt <= 0.0:
                previous_velocity = (0.0, 0.0, 0.0)
                control_dt = 0.02
            else:
                # Preserve the last command across delayed callbacks. Resetting
                # it to zero after every planner-induced scheduling gap makes
                # the UAV restart its acceleration ramp indefinitely.
                previous_velocity = channel.last_velocity_world
                control_dt = clamp(control_dt, 0.005, 0.1)
            velocity_world = limit_velocity_change(
                previous_velocity,
                velocity_world,
                self.max_command_acceleration,
                control_dt,
            )
            channel.last_velocity_world = velocity_world
            channel.last_velocity_sim_time = sim_now

            attitude = channel.imu.orientation
            attitude_quaternion = (
                attitude.x,
                attitude.y,
                attitude.z,
                attitude.w,
            )
            velocity_body = rotate_world_to_body(
                velocity_world, attitude_quaternion
            )
            measured_linear = channel.odom.twist.twist.linear
            measured_velocity_world = (
                measured_linear.x,
                measured_linear.y,
                measured_linear.z,
            )
            measured_velocity_body = rotate_world_to_body(
                measured_velocity_world, attitude_quaternion
            )
            velocity_body, lidar_limited, obstacle_clearance = (
                self.lidar_velocity_barrier(
                    channel, velocity_body, measured_velocity_body
                )
            )
            # The raw-scan barrier may deliberately exceed the nominal
            # acceleration ramp to stop actual momentum. Keep the resulting
            # command inside RACER's velocity envelope.  Because obstacle
            # repulsion is applied after the first inter-UAV projection, it
            # can otherwise push a vehicle toward a different neighbor. Apply
            # the separation barrier once more as the final world-frame safety
            # projection; like emergency obstacle braking, this projection is
            # intentionally not weakened by the nominal acceleration limiter.
            velocity_world = rotate_body_to_world(
                velocity_body, attitude_quaternion
            )
            velocity_world = calculate_world_velocity(
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                velocity_world,
                0.0,
                self.max_horizontal_speed,
                self.max_vertical_speed,
            )
            velocity_before_final_separation = velocity_world
            velocity_world = apply_separation_barrier(
                current,
                velocity_world,
                (
                    position
                    for rid, position in common_positions.items()
                    if rid != robot_id
                ),
                self.separation_activation_distance,
                self.separation_hard_distance,
                self.separation_repulsion_speed,
                self.max_command_acceleration,
            )
            final_separation_limited = math.sqrt(
                sum(
                    (
                        float(velocity_world[index])
                        - float(velocity_before_final_separation[index])
                    )
                    ** 2
                    for index in range(3)
                )
            ) > 1.0e-4
            separation_limited = separation_limited or final_separation_limited
            velocity_world = calculate_world_velocity(
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                velocity_world,
                0.0,
                self.max_horizontal_speed,
                self.max_vertical_speed,
            )
            velocity_body = rotate_world_to_body(
                velocity_world, attitude_quaternion
            )
            execution_blocked = self.update_execution_blocked(
                channel, lidar_limited or separation_limited, sim_now
            )

            current_yaw = yaw_from_quaternion(attitude_quaternion)
            if self.ground_unicycle:
                (
                    command.linear.x,
                    command.angular.z,
                    executed_world,
                ) = self.ground_command_from_world_velocity(
                    velocity_world, current_yaw
                )
                command.linear.y = 0.0
                command.linear.z = 0.0
                channel.last_velocity_world = executed_world
            elif self.ground_omni:
                # The RL gait tracks body (vx, vy, wz) directly: send the
                # planar body velocity as-is and only *align* the heading with
                # the direction of travel (no turn-in-place gating, so lateral
                # / diagonal stepping absorbs heading transients).
                command.linear.x = float(velocity_body[0])
                command.linear.y = float(velocity_body[1])
                command.linear.z = 0.0
                planar_speed = math.hypot(
                    velocity_world[0], velocity_world[1]
                )
                yaw_error = 0.0
                if planar_speed > 0.15:
                    yaw_error = wrap_angle(
                        math.atan2(velocity_world[1], velocity_world[0])
                        - current_yaw
                    )
                command.angular.z = clamp(
                    self.ground_heading_kp * yaw_error,
                    -self.max_yaw_rate,
                    self.max_yaw_rate,
                )
                channel.last_velocity_world = (
                    float(velocity_world[0]),
                    float(velocity_world[1]),
                    0.0,
                )
            else:
                command.linear.x, command.linear.y, command.linear.z = (
                    velocity_body
                )
                channel.last_velocity_world = velocity_world
                if self.track_racer_yaw:
                    target_q = target_pose.orientation
                    target_quaternion = (
                        target_q.x,
                        target_q.y,
                        target_q.z,
                        target_q.w,
                    )
                    target_yaw = yaw_from_quaternion(target_quaternion)
                else:
                    target_yaw = channel.reference_yaw
                    yaw_rate_ff = 0.0
                command.angular.z = clamp(
                    yaw_rate_ff
                    + self.yaw_kp
                    * wrap_angle(target_yaw - current_yaw),
                    -self.max_yaw_rate,
                    self.max_yaw_rate,
                )
            channel.cmd_pub.publish(command)
            error = math.sqrt(
                sum((target[index] - current[index]) ** 2 for index in range(3))
            )
            self.publish_state(
                robot_id,
                (
                    "tracking_racer_command_execution_blocked"
                    if execution_blocked
                    else "tracking_racer_command_lidar_limited"
                    if lidar_limited or separation_limited
                    else "tracking_racer_command"
                ),
            )
            if lidar_limited:
                self.get_logger().warning(
                    "bot%d raw-LiDAR velocity barrier clearance=%.2fm "
                    "measured_speed=%.2fm/s"
                    % (
                        robot_id,
                        obstacle_clearance,
                        math.sqrt(
                            sum(
                                component * component
                                for component in measured_velocity_body
                            )
                        ),
                    ),
                    throttle_duration_sec=1.0,
                )
            self.get_logger().debug(
                "bot%d position error %.3fm" % (robot_id, error),
                throttle_duration_sec=0.5,
            )

    def publish_bootstrap_command(
        self,
        robot_id: int,
        channel: Channel,
        now: float,
        command: Twist,
    ):
        movers = self.bootstrap_movers
        elapsed = max(0.0, now - self.bootstrap_started)
        active_robot = None
        active_elapsed = 0.0
        slot_duration = (
            2.0 * self.bootstrap_climb_seconds
            + self.bootstrap_loop_seconds
        )
        if (
            movers
            and elapsed >= self.bootstrap_hover_seconds
            and self.bootstrap_quality_ready_since is None
        ):
            schedule_time = elapsed - self.bootstrap_hover_seconds
            cycle_duration = (
                len(movers) * slot_duration + self.bootstrap_settle_seconds
            )
            cycle_time = schedule_time % cycle_duration
            if cycle_time < len(movers) * slot_duration:
                active_index = int(cycle_time // slot_duration)
                active_robot = movers[active_index]
                active_elapsed = cycle_time % slot_duration

        offset = (0.0, 0.0, 0.0)
        if robot_id == active_robot:
            mover_index = movers.index(robot_id)
            radius = max(0.35, self.bootstrap_radii[mover_index])
            direction = self.bootstrap_directions_rad[mover_index]
            offset = bootstrap_excitation_offset(
                active_elapsed,
                radius,
                0.0
                if self.ground_robot
                else self.bootstrap_climb_height,
                0.0
                if self.ground_robot
                else self.bootstrap_vertical_amplitude,
                self.bootstrap_climb_seconds,
                self.bootstrap_loop_seconds,
                direction,
                (
                    self.bootstrap_ground_lateral_scale
                    if self.ground_robot
                    else 1.0
                ),
            )

        origin = channel.local_origin
        pose = channel.local_odom.pose.pose
        current = (pose.position.x, pose.position.y, pose.position.z)
        target = (
            origin[0] + offset[0],
            origin[1] + offset[1],
            (
                current[2]
                if self.ground_robot
                else origin[2] + offset[2]
            ),
        )
        velocity_world = calculate_world_velocity(
            current,
            target,
            (0.0, 0.0, 0.0),
            self.kp,
            self.max_horizontal_speed,
            self.max_vertical_speed,
        )
        if (
            channel.imu is None
            or self.now_monotonic() - channel.imu_received > self.attitude_timeout
        ):
            channel.cmd_pub.publish(command)
            self.publish_state(robot_id, "failsafe_stale_imu_hover")
            return
        orientation = channel.imu.orientation
        quaternion = (
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w,
        )
        velocity_body = rotate_world_to_body(velocity_world, quaternion)
        lidar_scale, obstacle_clearance, _, _ = self.lidar_braking_scale(
            channel, velocity_body
        )
        if lidar_scale < 1.0:
            velocity_body = tuple(
                component * lidar_scale for component in velocity_body
            )
            self.get_logger().warning(
                "bot%d bootstrap raw-LiDAR brake scale=%.2f clearance=%.2fm"
                % (robot_id, lidar_scale, obstacle_clearance),
                throttle_duration_sec=1.0,
            )
        current_yaw = yaw_from_quaternion(quaternion)
        if self.ground_unicycle:
            safe_world = rotate_body_to_world(velocity_body, quaternion)
            (
                command.linear.x,
                command.angular.z,
                channel.last_velocity_world,
            ) = self.ground_command_from_world_velocity(
                (safe_world[0], safe_world[1], 0.0), current_yaw
            )
            command.linear.y = 0.0
            command.linear.z = 0.0
        elif self.ground_omni:
            safe_world = rotate_body_to_world(velocity_body, quaternion)
            command.linear.x = float(velocity_body[0])
            command.linear.y = float(velocity_body[1])
            command.linear.z = 0.0
            planar_speed = math.hypot(safe_world[0], safe_world[1])
            yaw_error = 0.0
            if planar_speed > 0.15:
                yaw_error = wrap_angle(
                    math.atan2(safe_world[1], safe_world[0]) - current_yaw
                )
            command.angular.z = clamp(
                self.ground_heading_kp * yaw_error,
                -self.max_yaw_rate,
                self.max_yaw_rate,
            )
            channel.last_velocity_world = (
                float(safe_world[0]),
                float(safe_world[1]),
                0.0,
            )
        else:
            command.linear.x, command.linear.y, command.linear.z = (
                velocity_body
            )
            command.angular.z = clamp(
                self.yaw_kp * wrap_angle(origin[3] - current_yaw),
                -self.max_yaw_rate,
                self.max_yaw_rate,
            )
        channel.cmd_pub.publish(command)
        if robot_id == active_robot:
            self.publish_state(
                robot_id,
                (
                    "initializing_common_frame_2d_excitation"
                    if self.ground_robot
                    else "initializing_common_frame_3d_excitation"
                ),
            )
        else:
            self.publish_state(
                robot_id, "initializing_common_frame_hover"
            )


def parse_robot_ids(text: str):
    result = tuple(int(value) for value in text.split(",") if value.strip())
    if not result:
        raise argparse.ArgumentTypeError("at least one UAV id is required")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bots", type=parse_robot_ids, default=(1,))
    args, ros_args = parser.parse_known_args()
    rclpy.init(args=ros_args)
    node = RacerVelocityController(args.bots)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, RCLError):
        pass
    finally:
        if rclpy.ok():
            for channel in node.channels.values():
                channel.cmd_pub.publish(Twist())
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
