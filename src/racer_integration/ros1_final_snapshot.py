#!/usr/bin/env python3
"""Capture final distributed RACER ownership and per-UAV map coverage."""

from __future__ import annotations

import argparse
import json
import math
import threading
import time

import rospy
from exploration_manager.msg import DroneState
from std_srvs.srv import Trigger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bots", default="1,2,3,4,5,6")
    parser.add_argument("--state-timeout", type=float, default=10.0)
    parser.add_argument("--service-timeout", type=float, default=20.0)
    args = parser.parse_args(rospy.myargv()[1:])
    bot_ids = tuple(int(value) for value in args.bots.split(",") if value)

    rospy.init_node("racer_final_snapshot", anonymous=True, disable_signals=True)
    lock = threading.Lock()
    latest = {}

    def state_callback(message: DroneState) -> None:
        if int(message.drone_id) not in bot_ids:
            return
        with lock:
            latest[int(message.drone_id)] = {
                "stamp": float(message.stamp),
                "grid_ids": [int(value) for value in message.grid_ids],
                "assignment_epoch": int(message.assignment_epoch),
                "grid_epochs": [int(value) for value in message.grid_epochs],
                "failed_grid_ids": [
                    int(value) for value in message.failed_grid_ids
                ],
                "failed_grid_until": [
                    float(value) for value in message.failed_grid_until
                ],
                "position_m": [float(value) for value in message.pos],
                "velocity_mps": [float(value) for value in message.vel],
                "yaw_rad": float(message.yaw),
            }

    subscriber = rospy.Subscriber(
        "/swarm_expl/drone_state", DroneState, state_callback, queue_size=100
    )
    deadline = time.monotonic() + args.state_timeout
    while not rospy.is_shutdown() and time.monotonic() < deadline:
        with lock:
            if all(bot_id in latest for bot_id in bot_ids):
                break
        rospy.sleep(0.05)
    subscriber.unregister()

    now_sim = float(rospy.Time.now().to_sec())
    with lock:
        states = dict(latest)
    for state in states.values():
        velocity = state["velocity_mps"]
        state["speed_mps"] = math.sqrt(sum(component * component for component in velocity))
        state["task_count"] = len(state["grid_ids"])
        state["state_age_sim_s"] = max(0.0, now_sim - state["stamp"])

    ownership = {}
    for bot_id, state in states.items():
        for grid_id in state["grid_ids"]:
            ownership.setdefault(grid_id, []).append(bot_id)

    coverage = {}
    for bot_id in bot_ids:
        service_name = f"/sdf_map/voxel_stats_{bot_id}"
        try:
            rospy.wait_for_service(service_name, timeout=args.service_timeout)
            response = rospy.ServiceProxy(service_name, Trigger)()
            coverage[f"bot{bot_id}"] = {
                "success": bool(response.success),
                "stats": json.loads(response.message) if response.message else {},
            }
        except Exception as error:  # ROS exceptions do not share one useful base.
            coverage[f"bot{bot_id}"] = {
                "success": False,
                "error": str(error),
            }

    result = {
        "sim_time_s": now_sim,
        "requested_bots": list(bot_ids),
        "received_state_bots": sorted(states),
        "states": {f"bot{key}": value for key, value in sorted(states.items())},
        "ownership": {
            "unique_task_count": len(ownership),
            "duplicate_task_ids": sorted(
                grid_id for grid_id, owners in ownership.items() if len(owners) > 1
            ),
            "per_bot_task_count": {
                f"bot{bot_id}": len(states.get(bot_id, {}).get("grid_ids", []))
                for bot_id in bot_ids
            },
        },
        "coverage": coverage,
    }
    print("RACER_FINAL_SNAPSHOT=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
