#!/usr/bin/env python3
"""Wait until RACER's official pairwise allocator has seeded every UAV."""

import argparse
import sys
import time

import rospy
from exploration_manager.msg import DroneState


def parse_ids(value):
    return tuple(int(item) for item in value.split(",") if item.strip())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bots", required=True, type=parse_ids)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--stable-seconds", type=float, default=2.0)
    parser.add_argument("--max-message-age", type=float, default=10.0)
    args = parser.parse_args(rospy.myargv(argv=sys.argv)[1:])

    expected = set(args.bots)
    task_counts = {}
    received_at = {}

    def callback(message):
        robot_id = int(message.drone_id)
        if robot_id not in expected:
            return
        task_counts[robot_id] = len(message.grid_ids)
        received_at[robot_id] = time.monotonic()

    rospy.init_node("racer_seed_allocation_gate", anonymous=True)
    rospy.Subscriber(
        "/swarm_expl/drone_state", DroneState, callback, queue_size=100
    )

    started = time.monotonic()
    stable_since = None
    next_report = started
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        now = time.monotonic()
        fresh = all(
            now - received_at.get(robot_id, -1.0e9) < args.max_message_age
            for robot_id in expected
        )
        allocated = all(task_counts.get(robot_id, 0) > 0 for robot_id in expected)
        if fresh and allocated:
            if stable_since is None:
                stable_since = now
            if now - stable_since >= args.stable_seconds:
                ordered = {robot_id: task_counts[robot_id] for robot_id in args.bots}
                print(
                    "[trigger] official seed allocation ready: %s" % ordered,
                    flush=True,
                )
                return 0
        else:
            stable_since = None

        if now >= next_report:
            ordered = {robot_id: task_counts.get(robot_id, 0) for robot_id in args.bots}
            print("[trigger] waiting for official seed allocation: %s" % ordered, flush=True)
            next_report = now + 10.0

        if now - started >= args.timeout:
            ordered = {robot_id: task_counts.get(robot_id, 0) for robot_id in args.bots}
            print(
                "[trigger] seed allocation timeout: %s" % ordered,
                file=sys.stderr,
                flush=True,
            )
            return 1
        rate.sleep()

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
