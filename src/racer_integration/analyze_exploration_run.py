#!/usr/bin/env python3
"""Summarize a RACER/Swarm-LIO2 recorder artifact and ROS 1 planner log."""

from __future__ import annotations

import argparse
from collections import deque
import json
import math
import re
import statistics
from pathlib import Path

import numpy as np


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
METRIC_STDOUT_RE = re.compile(
    r"\[[^\]]+,\s*([0-9.]+)\]: RACER_METRIC (\w+) (.*)"
)
METRIC_ROSOUT_RE = re.compile(
    r"^([0-9.]+)\s+\w+\s+/\S+.*\]\s+RACER_METRIC\s+(\w+)\s+(.*)"
)
FIELD_RE = re.compile(r"(\w+)=([^\s]+)")


def percentile(values, quantile):
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def distribution(values):
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "max": max(values),
    }


def longest_true_duration(times, mask):
    longest = 0.0
    start = None
    for index, value in enumerate(mask):
        if value and start is None:
            start = index
        if start is not None and (not value or index == len(mask) - 1):
            end = index if value and index == len(mask) - 1 else index - 1
            longest = max(longest, float(times[end] - times[start]))
            start = None
    return longest


def longest_confined_duration(times, positions, diameter=2.5):
    """Longest interval whose 3-D bounding-box diagonal stays within diameter."""
    if len(times) < 2:
        return 0.0
    minimums = [deque() for _ in range(3)]
    maximums = [deque() for _ in range(3)]
    left = 0
    longest = 0.0
    for right in range(len(times)):
        for axis in range(3):
            while minimums[axis] and (
                positions[minimums[axis][-1], axis] >= positions[right, axis]
            ):
                minimums[axis].pop()
            minimums[axis].append(right)
            while maximums[axis] and (
                positions[maximums[axis][-1], axis] <= positions[right, axis]
            ):
                maximums[axis].pop()
            maximums[axis].append(right)

        while left < right:
            span = np.array(
                [
                    positions[maximums[axis][0], axis]
                    - positions[minimums[axis][0], axis]
                    for axis in range(3)
                ]
            )
            if float(np.linalg.norm(span)) <= diameter:
                break
            for axis in range(3):
                if minimums[axis] and minimums[axis][0] == left:
                    minimums[axis].popleft()
                if maximums[axis] and maximums[axis][0] == left:
                    maximums[axis].popleft()
            left += 1
        longest = max(longest, float(times[right] - times[left]))
    return longest


def motion_summary(
    times,
    positions,
    smoothing_seconds=2.0,
    threshold=0.08,
    confinement_diameter=2.5,
):
    if len(times) < 3:
        return {"samples": len(times)}
    dt = np.diff(times)
    step = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    valid = dt > 1.0e-6
    speed = np.zeros_like(step)
    speed[valid] = step[valid] / dt[valid]
    median_dt = float(np.median(dt[valid])) if np.any(valid) else 0.1
    window = max(1, int(round(smoothing_seconds / max(median_dt, 1.0e-3))))
    kernel = np.ones(window, dtype=np.float64) / window
    smooth = np.convolve(speed, kernel, mode="same")
    stationary = smooth < threshold
    duration = float(times[-1] - times[0])
    distance = float(np.sum(step))
    recent_start = int(np.searchsorted(times, times[-1] - 60.0, side="left"))
    return {
        "samples": len(times),
        "duration_s": duration,
        "distance_m": distance,
        "mean_speed_mps": distance / max(duration, 1.0e-9),
        "motion_fraction": float(np.mean(~stationary)),
        "longest_stationary_s": longest_true_duration(times[1:], stationary),
        "longest_confined_s": longest_confined_duration(
            times, positions, confinement_diameter
        ),
        "confinement_diameter_m": confinement_diameter,
        "last_60s_displacement_m": float(
            np.linalg.norm(positions[-1] - positions[recent_start])
        ),
        "stationary_speed_threshold_mps": threshold,
        "smoothing_window_s": smoothing_seconds,
    }


def parse_metrics(log_path):
    text = ANSI_RE.sub("", log_path.read_text(errors="replace"))
    records = []
    for line in text.splitlines():
        match = METRIC_STDOUT_RE.search(line)
        if match is None:
            match = METRIC_ROSOUT_RE.search(line)
        if match is None:
            continue
        records.append(
            (
                float(match.group(1)),
                match.group(2),
                dict(FIELD_RE.findall(match.group(3))),
            )
        )
    return text, records


def planner_summary(records):
    plans = [(stamp, fields) for stamp, kind, fields in records if kind == "plan"]
    if not plans:
        return {}
    first_stamp = min(stamp for stamp, _ in plans)
    result = {"first_plan_sim_s": first_stamp}
    for kind, field in (
        ("plan", "wall_ms"),
        ("full_plan", "total_ms"),
        ("route", "total_ms"),
        ("grid_tour", "total_ms"),
        ("frontier_tour", "total_ms"),
        ("local_traj", "total_ms"),
    ):
        values = [
            float(fields[field])
            for _, record_kind, fields in records
            if record_kind == kind and field in fields
        ]
        result[kind] = distribution(values)
    result["frontier_update"] = distribution(
        [
            float(fields["total_ms"])
            for _, kind, fields in records
            if kind == "frontier" and "total_ms" in fields
        ]
    )
    result["viewpoint_evaluation"] = distribution(
        [
            float(fields["view_ms"])
            for _, kind, fields in records
            if kind == "frontier" and "view_ms" in fields
        ]
    )

    by_drone = {}
    for drone_id in range(1, 7):
        drone_plans = [
            (stamp, fields)
            for stamp, fields in plans
            if fields.get("drone") == str(drone_id)
        ]
        if not drone_plans:
            continue
        stamps = [stamp for stamp, _ in drone_plans]
        span = max(stamps) - min(stamps)
        by_drone[f"bot{drone_id}"] = {
            "count": len(drone_plans),
            "frequency_hz": (len(drone_plans) - 1) / span if span > 0 else 0.0,
            "wall_ms": distribution(
                [float(fields["wall_ms"]) for _, fields in drone_plans]
            ),
            "results": {
                code: sum(fields.get("result") == code for _, fields in drone_plans)
                for code in sorted({fields.get("result") for _, fields in drone_plans})
            },
        }
    result["by_drone"] = by_drone
    return result


def viewpoint_information_summary(records):
    """Summarize selected-view expected unknown gain from planner logs."""
    gain_records = [
        fields for _, kind, fields in records if kind == "viewpoint_gain"
    ]
    if not gain_records:
        return {}

    def summarize(items):
        if not items:
            return {}
        selected_views = sum(int(fields["views"]) for fields in items)
        weighted_cells = sum(
            int(fields["views"]) * float(fields["mean_cells"])
            for fields in items
        )
        weighted_area = sum(
            int(fields["views"]) * float(fields["mean_planar_area_m2"])
            for fields in items
        )
        return {
            "updates": len(items),
            "selected_viewpoints": selected_views,
            "unknown_cells_per_selected_view_mean": (
                weighted_cells / selected_views if selected_views else None
            ),
            "unknown_cells_per_selected_view_min": min(
                int(fields["min_cells"]) for fields in items
            ),
            "unknown_cells_per_selected_view_max": max(
                int(fields["max_cells"]) for fields in items
            ),
            "predicted_planar_area_m2_per_selected_view_mean": (
                weighted_area / selected_views if selected_views else None
            ),
            "predicted_planar_area_m2_per_selected_view_max": max(
                float(fields["max_planar_area_m2"]) for fields in items
            ),
        }

    result = summarize(gain_records)
    selected_targets = [
        fields for _, kind, fields in records if kind == "next_view"
    ]
    if selected_targets:
        hgrid_gains = [
            float(fields["hgrid_gain"])
            for fields in selected_targets
            if "hgrid_gain" in fields
        ]
        view_gains = [
            float(fields["view_gain"])
            for fields in selected_targets
            if "view_gain" in fields
        ]
        result["selected_targets"] = {
            "count": len(selected_targets),
            "hgrid_unknown_gain_cells": distribution(hgrid_gains),
            "viewpoint_unknown_gain_cells": distribution(view_gains),
            "hgrid_gain_nonzero_fraction": (
                sum(value > 0.0 for value in hgrid_gains) / len(hgrid_gains)
                if hgrid_gains
                else None
            ),
            "viewpoint_gain_nonzero_fraction": (
                sum(value > 0.0 for value in view_gains) / len(view_gains)
                if view_gains
                else None
            ),
        }
    result["by_drone"] = {}
    for drone_id in sorted(
        {int(fields["drone"]) for fields in gain_records if "drone" in fields}
    ):
        result["by_drone"][f"bot{drone_id}"] = summarize(
            [fields for fields in gain_records if fields.get("drone") == str(drone_id)]
        )
    return result


def allocation_summary(records):
    allocations = [
        fields for _, kind, fields in records if kind == "pair_alloc"
    ]
    if not allocations:
        return {}
    accepted = [fields for fields in allocations if fields.get("accepted") == "1"]
    reasons = {}
    for fields in allocations:
        reason = fields.get("reason", "legacy")
        reasons[reason] = reasons.get(reason, 0) + 1
    accepted_gain = []
    comparable_accepted_gain = []
    by_reason = {}
    for fields in accepted:
        before = float(fields["prev_makespan"])
        after = float(fields["cur_makespan"])
        if before > 1.0e-9:
            gain = (before - after) / before
            accepted_gain.append(gain)
            if fields.get("reason") not in {"seed", "recovery_missed"}:
                comparable_accepted_gain.append(gain)
    for reason in reasons:
        reason_records = [
            fields for fields in allocations if fields.get("reason", "legacy") == reason
        ]
        by_reason[reason] = {
            "attempts": len(reason_records),
            "accepted": sum(fields.get("accepted") == "1" for fields in reason_records),
            "wall_ms": distribution(
                [float(fields["wall_ms"]) for fields in reason_records]
            ),
        }
    return {
        "attempts": len(allocations),
        "accepted": len(accepted),
        "acceptance_fraction": len(accepted) / len(allocations),
        "reasons": reasons,
        "wall_ms": distribution([float(fields["wall_ms"]) for fields in allocations]),
        "accepted_makespan_gain_fraction": distribution(accepted_gain),
        # Seed and missed-task recovery candidates contain work absent from
        # the old two routes, so their before/after makespans are not the same
        # optimization problem.  Keep the raw number above for auditability,
        # and use this subset to judge allocation quality.
        "comparable_accepted_makespan_gain_fraction": distribution(
            comparable_accepted_gain
        ),
        "by_reason": by_reason,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("npz", type=Path)
    parser.add_argument("planner_log", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    run_summary = json.loads(args.summary.read_text())
    archive = np.load(args.npz)
    text, records = parse_metrics(args.planner_log)
    result = {
        "duration_sim_s": run_summary.get("duration_sim_s"),
        "duration_wall_s": run_summary.get("duration_wall_s"),
        "rtf": run_summary.get("duration_sim_s", 0.0)
        / max(run_summary.get("duration_wall_s", 0.0), 1.0e-9),
        "mapped_surface_voxels": run_summary.get("mapped_surface_voxels"),
        "total_ground_truth_distance_m": run_summary.get(
            "total_ground_truth_distance_m"
        ),
        "emergency_stop_triggered": run_summary.get("emergency_stop_triggered"),
        "monitor_warnings": run_summary.get("monitor_warnings"),
        "motion": {},
        "slam": {},
        "planning": planner_summary(records),
        "viewpoint_information": viewpoint_information_summary(records),
        "allocation": allocation_summary(records),
        "astar_failure_count": text.count("No path to next viewpoint"),
        "released_unreachable_grid_count": text.count(
            "released_unreachable_grid"
        ),
        "quarantined_grid_count": text.count("quarantined_grid="),
        "reachability_deferred_count": text.count(
            "allocation_task_deferred"
        ),
        "ownership_duplicate_resolution_count": text.count(
            "ownership_duplicate_resolved"
        ),
        "stale_pair_transaction_rejection_count": text.count(
            "pair_transaction_rejected"
        ),
        "target_switch_suppressed_count": text.count(
            "target_switch_suppressed"
        ),
        "stagnant_viewpoint_release_count": text.count(
            "stagnant_viewpoint=1"
        ),
        "stagnant_hgrid_release_count": text.count("stagnant_hgrid=1"),
        "execution_blocked_replan_count": text.count(
            "execution_blocked_replan=1"
        ),
        "execution_blocked_local_retry_count": text.count(
            "execution_blocked_local_retry=1"
        ),
        "astar_timeout_retry_count": text.count("astar_timeout_retry"),
        "known_boundary_advance_count": text.count("advanced_known_boundary=1"),
        "known_path_budget_recovery_count": text.count(
            "recovered_known_path_after_budget=1"
        ),
        "one_point_hover_count": text.count("One-point exploration path"),
        "planner_abort_count": text.count("Aborted"),
    }
    minimum_clearance = float("inf")
    contact = False
    for stats in run_summary.get("inter_uav_clearance", {}).values():
        if stats.get("min_3d_m") is not None:
            minimum_clearance = min(minimum_clearance, stats["min_3d_m"])
        contact = contact or bool(stats.get("contact_detected"))
    result["minimum_inter_uav_3d_m"] = (
        minimum_clearance if math.isfinite(minimum_clearance) else None
    )
    result["contact_detected"] = contact

    for robot_id in archive["robot_ids"]:
        robot_id = int(robot_id)
        key = f"bot{robot_id}"
        result["motion"][key] = motion_summary(
            np.asarray(archive[f"t_{robot_id}"], dtype=np.float64),
            np.asarray(archive[f"truth_{robot_id}"], dtype=np.float64),
        )
        robot_summary = run_summary["robots"][key]
        result["slam"][key] = {
            "first_pose_se3_ate_rmse_m": robot_summary["first_pose_se3"][
                "ate_rmse_m"
            ],
            "first_pose_se3_ate_max_m": robot_summary["first_pose_se3"][
                "max_error_m"
            ],
            "root_trajectory_se3_ate_rmse_m": robot_summary[
                "root_trajectory_se3"
            ]["ate_rmse_m"],
            "root_trajectory_se3_ate_max_m": robot_summary[
                "root_trajectory_se3"
            ]["max_error_m"],
            "valid_fraction": robot_summary["slam_validity"]["valid_fraction"],
            "validity_transitions": robot_summary["slam_validity"]["transitions"],
        }

    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
