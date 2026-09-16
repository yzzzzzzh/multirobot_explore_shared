#!/usr/bin/env python3
"""Measure RACER/Swarm-LIO2 position ATE against Gazebo for test runs only.

The root alignment is a full SE(3) transform latched from bot1's first paired
estimate and ground-truth poses. No production controller consumes this topic.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
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


Pose = Tuple[Tuple[float, float, float], Tuple[float, float, float, float], float]


def message_stamp_seconds(message) -> float:
    return float(message.header.stamp.sec) + float(message.header.stamp.nanosec) * 1e-9


class RacerEvaluator(Node):
    def __init__(
        self,
        robot_ids: Iterable[int],
        duration: float,
        max_skew: float,
        alignment_mode: str,
    ):
        super().__init__("racer_evaluator")
        self.robot_ids = tuple(robot_ids)
        self.duration = duration
        self.max_skew = max_skew
        self.alignment_mode = alignment_mode
        self.start_wall = time.monotonic()
        self.estimated: Dict[int, Pose] = {}
        self.truth: Dict[int, Pose] = {}
        self.errors: Dict[int, list] = {robot_id: [] for robot_id in self.robot_ids}
        self.error_components: Dict[int, list] = {
            robot_id: [] for robot_id in self.robot_ids
        }
        self.estimated_samples: Dict[int, list] = {
            robot_id: [] for robot_id in self.robot_ids
        }
        self.truth_samples: Dict[int, list] = {
            robot_id: [] for robot_id in self.robot_ids
        }
        self.distance: Dict[int, float] = {robot_id: 0.0 for robot_id in self.robot_ids}
        self.previous_truth: Dict[int, Tuple[float, float, float]] = {}
        self.alignment_rotation: Optional[Tuple[float, float, float, float]] = None
        self.alignment_translation: Optional[Tuple[float, float, float]] = None
        self.last_sample_wall = 0.0
        self.subscription_handles = [
            self.create_subscription(TFMessage, "/world/default/pose/info", self.on_truth, 100)
        ]
        for robot_id in self.robot_ids:
            self.subscription_handles.append(
                self.create_subscription(
                    Odometry,
                    f"/racer/bot{robot_id}/odom_world",
                    lambda message, rid=robot_id: self.on_estimate(rid, message),
                    50,
                )
            )

    def stamps_compatible(self, first: float, second: float) -> bool:
        # Gazebo Pose_V messages bridged as TFMessage carry stamp zero. They
        # arrive at the physics rate, so use the latest sample in that case.
        return first <= 0.0 or second <= 0.0 or abs(first - second) <= self.max_skew

    def on_estimate(self, robot_id: int, message: Odometry):
        p = message.pose.pose.position
        q = message.pose.pose.orientation
        self.estimated[robot_id] = (
            (p.x, p.y, p.z),
            (q.x, q.y, q.z, q.w),
            message_stamp_seconds(message),
        )

    def on_truth(self, message: TFMessage):
        for transform in message.transforms:
            child = transform.child_frame_id.lstrip("/")
            for robot_id in self.robot_ids:
                if child != f"bot{robot_id}":
                    continue
                p = transform.transform.translation
                q = transform.transform.rotation
                self.truth[robot_id] = (
                    (p.x, p.y, p.z),
                    (q.x, q.y, q.z, q.w),
                    message_stamp_seconds(transform),
                )

    def latch_alignment(self) -> bool:
        root_id = self.robot_ids[0]
        if root_id not in self.estimated or root_id not in self.truth:
            return False
        est_p, est_q, est_stamp = self.estimated[root_id]
        gt_p, gt_q, gt_stamp = self.truth[root_id]
        if not self.stamps_compatible(est_stamp, gt_stamp):
            return False
        rotation = multiply_quaternions(gt_q, conjugate_quaternion(est_q))
        rotated = rotate_body_to_world(est_p, rotation)
        self.alignment_rotation = rotation
        self.alignment_translation = tuple(gt_p[i] - rotated[i] for i in range(3))
        self.get_logger().info(
            "latched full-SE(3) Gazebo evaluation alignment at skew %.3fs"
            % (abs(est_stamp - gt_stamp) if gt_stamp > 0.0 else 0.0)
        )
        return True

    def sample(self):
        now = time.monotonic()
        if now - self.last_sample_wall < 0.1:
            return
        self.last_sample_wall = now
        if self.alignment_mode == "first_pose":
            if self.alignment_rotation is None and not self.latch_alignment():
                return
            assert self.alignment_translation is not None
        elif self.robot_ids[0] not in self.estimated or self.robot_ids[0] not in self.truth:
            return
        for robot_id in self.robot_ids:
            if robot_id not in self.estimated or robot_id not in self.truth:
                continue
            est_p, _, est_stamp = self.estimated[robot_id]
            gt_p, _, gt_stamp = self.truth[robot_id]
            if not self.stamps_compatible(est_stamp, gt_stamp):
                continue
            self.estimated_samples[robot_id].append(est_p)
            self.truth_samples[robot_id].append(gt_p)
            if self.alignment_mode == "first_pose":
                rotated = rotate_body_to_world(est_p, self.alignment_rotation)
                aligned = tuple(
                    rotated[index] + self.alignment_translation[index]
                    for index in range(3)
                )
                components = tuple(
                    aligned[index] - gt_p[index] for index in range(3)
                )
                error = math.sqrt(sum(value * value for value in components))
                self.errors[robot_id].append(error)
                self.error_components[robot_id].append(components)
            previous = self.previous_truth.get(robot_id)
            if previous is not None:
                self.distance[robot_id] += math.sqrt(
                    sum((gt_p[index] - previous[index]) ** 2 for index in range(3))
                )
            self.previous_truth[robot_id] = gt_p

    def report(self):
        if self.alignment_mode == "root_trajectory":
            root_id = self.robot_ids[0]
            try:
                rotation, translation = fit_rigid_transform_se3(
                    self.estimated_samples[root_id], self.truth_samples[root_id]
                )
                for robot_id in self.robot_ids:
                    aligned = apply_rigid_transform_se3(
                        self.estimated_samples[robot_id], rotation, translation
                    )
                    truth = self.truth_samples[robot_id]
                    for estimated_point, truth_point in zip(aligned, truth):
                        components = tuple(
                            float(estimated_point[axis] - truth_point[axis])
                            for axis in range(3)
                        )
                        self.error_components[robot_id].append(components)
                        self.errors[robot_id].append(
                            math.sqrt(sum(value * value for value in components))
                        )
            except ValueError as error:
                result = {
                    "duration_wall_s": self.duration,
                    "alignment": "root trajectory full SE(3)",
                    "error": str(error),
                    "robots": {},
                }
                print(
                    "RACER_EVAL_RESULT=" + json.dumps(result, sort_keys=True),
                    flush=True,
                )
                return

        result = {
            "duration_wall_s": self.duration,
            "alignment": (
                "root trajectory full SE(3)"
                if self.alignment_mode == "root_trajectory"
                else "root first-pose full SE(3)"
            ),
            "robots": {},
        }
        for robot_id in self.robot_ids:
            values = self.errors[robot_id]
            if not values:
                result["robots"][f"bot{robot_id}"] = {"samples": 0}
                continue
            result["robots"][f"bot{robot_id}"] = {
                "samples": len(values),
                "ate_rmse_m": math.sqrt(sum(value * value for value in values) / len(values)),
                "mean_error_m": sum(values) / len(values),
                "max_error_m": max(values),
                "ground_truth_distance_m": self.distance[robot_id],
                "mean_signed_error_xyz_m": [
                    sum(component[axis] for component in self.error_components[robot_id])
                    / len(values)
                    for axis in range(3)
                ],
            }
        print("RACER_EVAL_RESULT=" + json.dumps(result, sort_keys=True), flush=True)


def parse_robot_ids(text: str):
    result = tuple(int(value) for value in text.split(",") if value.strip())
    if not result:
        raise argparse.ArgumentTypeError("at least one UAV id is required")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bots", type=parse_robot_ids, default=(1,))
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--max-skew", type=float, default=0.2)
    parser.add_argument(
        "--alignment",
        choices=("first_pose", "root_trajectory"),
        default="first_pose",
        help="online first-pose alignment or standard root-trajectory SE(3) fit",
    )
    args, ros_args = parser.parse_known_args()
    rclpy.init(args=ros_args)
    node = RacerEvaluator(
        args.bots, args.duration, args.max_skew, args.alignment
    )
    try:
        # Do not use a ROS timer for a wall-clock test.  Gazebo can jump /clock
        # during startup, and the high-rate pose/info subscription can starve a
        # timer in a single-threaded executor.  One callback per spin keeps the
        # test responsive; sample() performs the 10 Hz wall-time throttling.
        while (
            rclpy.ok()
            and time.monotonic() - node.start_wall < node.duration
        ):
            rclpy.spin_once(node, timeout_sec=0.05)
            node.sample()
        node.report()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
