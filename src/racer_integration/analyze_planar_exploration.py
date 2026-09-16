#!/usr/bin/env python3
"""Evaluate a planar RACER exploration run against a known building layout.

The exploration metric is observable navigable free area: a grid cell counts
only after a ground-truth robot pose has a range-limited, wall-unoccluded ray
to it.  Gazebo truth is used for evaluation only, never by SLAM or control.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from racer_evaluation_math import apply_rigid_transform_se3


def load_layout(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def build_masks(layout, resolution: float, robot_radius: float, sensor_z: float):
    xmin, xmax, ymin, ymax = (float(value) for value in layout["bounds_xy_m"])
    x = np.arange(xmin + 0.5 * resolution, xmax, resolution)
    y = np.arange(ymin + 0.5 * resolution, ymax, resolution)
    grid_x, grid_y = np.meshgrid(x, y)
    occupied = np.zeros(grid_x.shape, dtype=bool)
    inflated = np.zeros(grid_x.shape, dtype=bool)
    obstacle_boxes = []

    for box in layout["boxes"]:
        cx, cy, cz = (float(value) for value in box["center"])
        sx, sy, sz = (float(value) for value in box["size"])
        if not (cz - 0.5 * sz <= sensor_z <= cz + 0.5 * sz):
            continue
        obstacle_boxes.append(box)
        occupied |= (
            (np.abs(grid_x - cx) <= 0.5 * sx)
            & (np.abs(grid_y - cy) <= 0.5 * sy)
        )
        inflated |= (
            (np.abs(grid_x - cx) <= 0.5 * sx + robot_radius)
            & (np.abs(grid_y - cy) <= 0.5 * sy + robot_radius)
        )

    return {
        "x": x,
        "y": y,
        "occupied": occupied,
        "navigable": ~inflated,
        "obstacle_boxes": obstacle_boxes,
        "bounds": (xmin, xmax, ymin, ymax),
        "resolution": resolution,
    }


def raycast_visible(position_xy, masks, lidar_range: float, ray_count: int):
    xmin, xmax, ymin, ymax = masks["bounds"]
    resolution = masks["resolution"]
    angles = np.linspace(0.0, 2.0 * math.pi, ray_count, endpoint=False)
    ranges = np.arange(
        0.0, lidar_range + 0.25 * resolution, 0.5 * resolution
    )
    sample_x = (
        float(position_xy[0])
        + np.cos(angles)[:, None] * ranges[None, :]
    )
    sample_y = (
        float(position_xy[1])
        + np.sin(angles)[:, None] * ranges[None, :]
    )
    ix = np.floor((sample_x - xmin) / resolution).astype(np.int32)
    iy = np.floor((sample_y - ymin) / resolution).astype(np.int32)
    valid = (
        (ix >= 0)
        & (ix < masks["occupied"].shape[1])
        & (iy >= 0)
        & (iy < masks["occupied"].shape[0])
    )
    clipped_x = np.clip(ix, 0, masks["occupied"].shape[1] - 1)
    clipped_y = np.clip(iy, 0, masks["occupied"].shape[0] - 1)
    blocked = ~valid | masks["occupied"][clipped_y, clipped_x]
    before_wall = np.cumsum(blocked, axis=1) == 0
    visible = np.zeros_like(masks["occupied"])
    usable = valid & before_wall
    visible[clipped_y[usable], clipped_x[usable]] = True
    return visible & masks["navigable"]


def select_motion_samples(times, positions, min_distance: float, max_dt: float):
    if len(times) == 0:
        return np.empty((0,), dtype=int)
    selected = [0]
    for index in range(1, len(times)):
        previous = selected[-1]
        if (
            times[index] - times[previous] >= max_dt
            or np.linalg.norm(positions[index, :2] - positions[previous, :2])
            >= min_distance
        ):
            selected.append(index)
    if selected[-1] != len(times) - 1:
        selected.append(len(times) - 1)
    return np.asarray(selected, dtype=int)


def first_time_at_fraction(times, fractions, threshold: float):
    indices = np.flatnonzero(fractions >= threshold)
    return float(times[indices[0]]) if len(indices) else None


def coverage_checkpoints(
    times,
    fractions,
    navigable_area_m2: float,
    duration_s: float,
    interval_s: float = 60.0,
):
    """Return cumulative coverage and incremental unknown-area reduction.

    Each interval rate measures only newly observable navigable area in that
    interval.  Values after an aborted run are deliberately not extrapolated.
    """
    if duration_s <= 0.0 or interval_s <= 0.0:
        return []
    checkpoints = list(
        np.arange(interval_s, duration_s + 1.0e-9, interval_s)
    )
    if not checkpoints or duration_s - checkpoints[-1] > 1.0e-6:
        checkpoints.append(duration_s)

    output = []
    previous_time = 0.0
    previous_area = 0.0
    for checkpoint in checkpoints:
        index = int(np.searchsorted(times, checkpoint, side="right")) - 1
        fraction = float(fractions[index]) if index >= 0 else 0.0
        area = fraction * navigable_area_m2
        elapsed = max(float(checkpoint) - previous_time, 1.0e-9)
        output.append(
            {
                "time_s": float(checkpoint),
                "observed_fraction": fraction,
                "observed_area_m2": area,
                "new_area_in_interval_m2": area - previous_area,
                "new_area_rate_m2_per_s": (area - previous_area) / elapsed,
            }
        )
        previous_time = float(checkpoint)
        previous_area = area
    return output


def wall_clearance_and_collisions(
    positions, obstacle_boxes, robot_radius: float, sensor_z: float
):
    collision_samples = 0
    minimum_clearance = float("inf")
    for point in positions:
        px, py = (float(value) for value in point[:2])
        for box in obstacle_boxes:
            cx, cy, cz = (float(value) for value in box["center"])
            sx, sy, sz = (float(value) for value in box["size"])
            if not (cz - 0.5 * sz <= sensor_z <= cz + 0.5 * sz):
                continue
            dx = max(abs(px - cx) - 0.5 * sx, 0.0)
            dy = max(abs(py - cy) - 0.5 * sy, 0.0)
            clearance = math.hypot(dx, dy) - robot_radius
            minimum_clearance = min(minimum_clearance, clearance)
            if clearance < 0.0:
                collision_samples += 1
                break
    return minimum_clearance, collision_samples


def longest_stationary_interval(times, positions, diameter: float = 1.0):
    """Longest interval whose trajectory stays inside a fixed diameter."""
    if len(times) == 0:
        return 0.0
    longest = 0.0
    start = 0
    for end in range(len(times)):
        while start < end:
            segment = positions[start : end + 1, :2]
            span = np.ptp(segment, axis=0)
            if math.hypot(float(span[0]), float(span[1])) <= diameter:
                break
            start += 1
        longest = max(longest, float(times[end] - times[start]))
    return longest


def analyze(run_path: Path, layout_path: Path, args):
    run = np.load(run_path, allow_pickle=False)
    layout = load_layout(layout_path)
    robot_ids = [int(value) for value in run["robot_ids"]]
    masks = build_masks(
        layout, args.grid_resolution, args.robot_radius, args.sensor_z
    )
    navigable_count = int(np.count_nonzero(masks["navigable"]))
    if navigable_count == 0:
        raise RuntimeError("layout contains no navigable cells")

    events = []
    robot_data = {}
    for robot_id in robot_ids:
        times = np.asarray(run[f"t_{robot_id}"], dtype=np.float64)
        truth = np.asarray(run[f"truth_{robot_id}"], dtype=np.float64)
        estimate = np.asarray(run[f"estimate_{robot_id}"], dtype=np.float64)
        count = min(len(times), len(truth), len(estimate))
        times, truth, estimate = times[:count], truth[:count], estimate[:count]
        chosen = select_motion_samples(
            times, truth, args.coverage_pose_step, args.coverage_max_dt
        )
        events.extend(
            (float(times[index]), robot_id, truth[index, :2])
            for index in chosen
        )
        robot_data[robot_id] = (times, truth, estimate)
    events.sort(key=lambda item: (item[0], item[1]))

    observed = np.zeros_like(masks["navigable"])
    observed_first_t = np.full(observed.shape, np.inf, dtype=np.float32)
    curve_t = []
    curve_fraction = []
    new_cells_by_robot = {robot_id: 0 for robot_id in robot_ids}
    for timestamp, robot_id, position_xy in events:
        visible = raycast_visible(
            position_xy, masks, args.lidar_range, args.ray_count
        )
        newly_observed = visible & ~observed
        new_cells_by_robot[robot_id] += int(np.count_nonzero(newly_observed))
        observed_first_t[newly_observed] = float(timestamp)
        observed |= visible
        curve_t.append(timestamp)
        curve_fraction.append(
            float(np.count_nonzero(observed)) / navigable_count
        )
    curve_t = np.asarray(curve_t, dtype=np.float64)
    curve_fraction = np.asarray(curve_fraction, dtype=np.float64)

    trajectory_rotation = np.asarray(
        run["trajectory_rotation"], dtype=np.float64
    )
    trajectory_translation = np.asarray(
        run["trajectory_translation"], dtype=np.float64
    )
    first_pose_rotation = np.asarray(
        run["first_pose_rotation"], dtype=np.float64
    )
    first_pose_translation = np.asarray(
        run["first_pose_translation"], dtype=np.float64
    )
    robots = {}
    total_distance = 0.0
    all_pair_times = []
    all_pair_positions = {}
    for robot_id, (times, truth, estimate) in robot_data.items():
        aligned = apply_rigid_transform_se3(
            estimate, trajectory_rotation, trajectory_translation
        )
        errors = np.linalg.norm(aligned - truth, axis=1)
        first_pose_aligned = apply_rigid_transform_se3(
            estimate, first_pose_rotation, first_pose_translation
        )
        first_pose_errors = np.linalg.norm(first_pose_aligned - truth, axis=1)
        distance = (
            float(np.linalg.norm(np.diff(truth[:, :2], axis=0), axis=1).sum())
            if len(truth) > 1
            else 0.0
        )
        duration = float(times[-1] - times[0]) if len(times) > 1 else 0.0
        wall_clearance, wall_collision_samples = wall_clearance_and_collisions(
            truth, masks["obstacle_boxes"], args.robot_radius, args.sensor_z
        )
        quarter = max(1, len(errors) // 4)
        robots[f"bot{robot_id}"] = {
            "distance_m": distance,
            "mean_speed_mps": distance / max(duration, 1.0e-9),
            "ate_root_trajectory_se3_rmse_m": (
                float(np.sqrt(np.mean(errors**2))) if len(errors) else None
            ),
            "ate_first_pose_se3_rmse_m": (
                float(np.sqrt(np.mean(first_pose_errors**2)))
                if len(first_pose_errors)
                else None
            ),
            "ate_first_quarter_rmse_m": (
                float(np.sqrt(np.mean(errors[:quarter] ** 2)))
                if len(errors)
                else None
            ),
            "ate_last_quarter_rmse_m": (
                float(np.sqrt(np.mean(errors[-quarter:] ** 2)))
                if len(errors)
                else None
            ),
            "ate_max_m": float(np.max(errors)) if len(errors) else None,
            "minimum_wall_clearance_m": (
                wall_clearance if math.isfinite(wall_clearance) else None
            ),
            "wall_collision_samples": int(wall_collision_samples),
            "longest_time_inside_1m_diameter_s": longest_stationary_interval(
                times, truth
            ),
            "new_observed_area_m2": (
                new_cells_by_robot[robot_id] * args.grid_resolution**2
            ),
        }
        total_distance += distance
        all_pair_times.extend(times.tolist())
        all_pair_positions[robot_id] = (times, truth)

    minimum_pair_distance = float("inf")
    pair_collision_samples = 0
    if len(robot_ids) >= 2:
        first_times, first_truth = all_pair_positions[robot_ids[0]]
        for second_id in robot_ids[1:]:
            second_times, second_truth = all_pair_positions[second_id]
            if len(first_times) and len(second_times):
                synced = np.column_stack(
                    [
                        np.interp(first_times, second_times, second_truth[:, axis])
                        for axis in range(3)
                    ]
                )
                distances = np.linalg.norm(first_truth[:, :2] - synced[:, :2], axis=1)
                minimum_pair_distance = min(
                    minimum_pair_distance, float(np.min(distances))
                )
                pair_collision_samples += int(
                    np.count_nonzero(distances < 2.0 * args.robot_radius)
                )

    final_fraction = float(curve_fraction[-1]) if len(curve_fraction) else 0.0
    duration = max(
        (
            float(values[0][-1])
            for values in robot_data.values()
            if len(values[0])
        ),
        default=0.0,
    )
    observed_area = final_fraction * navigable_count * args.grid_resolution**2
    navigable_area = navigable_count * args.grid_resolution**2
    result = {
        "run": str(run_path),
        "layout": str(layout_path),
        "robot_ids": robot_ids,
        "duration_sim_s": duration,
        "grid_resolution_m": args.grid_resolution,
        "lidar_range_m": args.lidar_range,
        "navigable_area_m2": navigable_area,
        "observed_navigable_area_m2": observed_area,
        "observed_fraction": final_fraction,
        "unobserved_navigable_area_m2": (
            (1.0 - final_fraction)
            * navigable_count
            * args.grid_resolution**2
        ),
        "time_to_80_percent_s": first_time_at_fraction(
            curve_t, curve_fraction, 0.80
        ),
        "time_to_90_percent_s": first_time_at_fraction(
            curve_t, curve_fraction, 0.90
        ),
        "time_to_95_percent_s": first_time_at_fraction(
            curve_t, curve_fraction, 0.95
        ),
        "exploration_efficiency_m2_per_s": observed_area / max(duration, 1.0e-9),
        "exploration_efficiency_m2_per_m": observed_area
        / max(total_distance, 1.0e-9),
        "complete_at_80_percent": final_fraction >= 0.80,
        "complete_at_95_percent": final_fraction >= 0.95,
        "coverage_checkpoints": coverage_checkpoints(
            curve_t, curve_fraction, navigable_area, duration
        ),
        "total_distance_m": total_distance,
        "minimum_inter_robot_distance_m": (
            minimum_pair_distance
            if math.isfinite(minimum_pair_distance)
            else None
        ),
        "inter_robot_collision_samples": pair_collision_samples,
        "robots": robots,
    }
    return result, {
        "observed": observed,
        "observed_first_t": observed_first_t,
        "navigable": masks["navigable"],
        "curve_t": curve_t,
        "curve_fraction": curve_fraction,
        "grid_x": masks["x"],
        "grid_y": masks["y"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--layout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--grid-resolution", type=float, default=0.25)
    parser.add_argument("--lidar-range", type=float, default=10.0)
    parser.add_argument("--robot-radius", type=float, default=0.40)
    parser.add_argument("--sensor-z", type=float, default=0.645)
    parser.add_argument("--ray-count", type=int, default=720)
    parser.add_argument("--coverage-pose-step", type=float, default=0.40)
    parser.add_argument("--coverage-max-dt", type=float, default=1.0)
    args = parser.parse_args()

    result, arrays = analyze(args.run, args.layout, args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
    np.savez_compressed(
        args.output.with_suffix(".coverage.npz"), **arrays
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
