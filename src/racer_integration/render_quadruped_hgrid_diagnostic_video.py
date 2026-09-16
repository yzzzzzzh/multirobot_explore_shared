#!/usr/bin/env python3
"""Render per-robot HGrid, target, frontier and coverage diagnostics.

The HGrid target/frontier fields come from RACER's ROS log.  Per-cell coverage
is reconstructed from Gazebo ground-truth poses and the known building layout;
it is evaluation data and was never available to the online planner.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FFMpegWriter  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Polygon, Rectangle  # noqa: E402

from analyze_planar_exploration import (  # noqa: E402
    build_masks,
    load_layout,
    raycast_visible,
    select_motion_samples,
)
from render_exploration_video import apply_transform  # noqa: E402


NEXT_VIEW_RE = re.compile(
    r"^(?P<stamp>\d+\.\d+).*RACER_METRIC next_view "
    r"drone=(?P<drone>\d+) grid=(?P<grid>-?\d+) "
    r"frontier=(?P<frontier>-?\d+) "
    r"start=\[(?P<start>[^]]+)\] target=\[(?P<target>[^]]+)\] "
    r"frontier_average=\[(?P<frontier_average>[^]]+)\] "
    r"distance_m=(?P<distance>[-+\d.eE]+) "
    r"hgrid_unknown=(?P<unknown>\d+) "
    r"hgrid_gain=(?P<hgrid_gain>\d+) view_gain=(?P<view_gain>\d+)"
)
RELEVANCE_RE = re.compile(
    r"^(?P<stamp>\d+\.\d+).*RACER_METRIC hgrid_relevance "
    r"drone=(?P<drone>\d+) level=(?P<level>\d+) "
    r"active=(?P<active>\d+) relevant=(?P<relevant>\d+) "
    r"visited=(?P<visited>\d+) "
    r"excluded_visited_no_frontier=(?P<excluded>\d+) "
    r"retained_unvisited_no_frontier=(?P<retained>\d+)"
)


@dataclass
class TargetRecord:
    stamp: float
    drone: int
    grid_id: int
    frontier_id: int
    start: np.ndarray
    target: np.ndarray
    frontier_average: np.ndarray
    distance_m: float
    hgrid_unknown: int
    hgrid_gain: int
    view_gain: int
    relative_t: float = 0.0
    start_world: np.ndarray | None = None
    target_world: np.ndarray | None = None
    frontier_world: np.ndarray | None = None


@dataclass
class RelevanceRecord:
    stamp: float
    drone: int
    level: int
    active: int
    relevant: int
    visited: int
    excluded: int
    retained: int
    relative_t: float = 0.0


def parse_vector(text: str) -> np.ndarray:
    return np.asarray([float(value.strip()) for value in text.split(",")])


def parse_racer_log(path: Path):
    targets = []
    relevance = []
    with path.open("r", encoding="utf-8", errors="ignore") as stream:
        for line in stream:
            match = NEXT_VIEW_RE.search(line)
            if match:
                targets.append(
                    TargetRecord(
                        stamp=float(match.group("stamp")),
                        drone=int(match.group("drone")),
                        grid_id=int(match.group("grid")),
                        frontier_id=int(match.group("frontier")),
                        start=parse_vector(match.group("start")),
                        target=parse_vector(match.group("target")),
                        frontier_average=parse_vector(
                            match.group("frontier_average")
                        ),
                        distance_m=float(match.group("distance")),
                        hgrid_unknown=int(match.group("unknown")),
                        hgrid_gain=int(match.group("hgrid_gain")),
                        view_gain=int(match.group("view_gain")),
                    )
                )
                continue
            match = RELEVANCE_RE.search(line)
            if match:
                relevance.append(
                    RelevanceRecord(
                        stamp=float(match.group("stamp")),
                        drone=int(match.group("drone")),
                        level=int(match.group("level")),
                        active=int(match.group("active")),
                        relevant=int(match.group("relevant")),
                        visited=int(match.group("visited")),
                        excluded=int(match.group("excluded")),
                        retained=int(match.group("retained")),
                    )
                )
    return targets, relevance


# Acceptance thresholds for the RACER-log/SLAM alignment.  The RACER frame is
# the adapter output frame (latched cross-robot transform), the recorder frame
# is first-pose SE(3) aligned to ground truth, so a residual of a few
# decimetres is expected whenever the latched transform carries an offset.
ALIGN_MAX_MEDIAN_M = 0.10
ALIGN_MAX_P90_M = 0.25


def fit_log_time_offset(targets, run_data, duration: float) -> float:
    """Align ROS simulation stamps to recorder-relative time using odometry."""
    if not targets:
        return 0.0
    by_drone = defaultdict(list)
    for record in targets:
        by_drone[record.drone].append(record)

    def objective(offset):
        squared = []
        used = 0
        for drone, records in by_drone.items():
            key_t = f"t_{drone}"
            key_estimate = f"estimate_{drone}"
            if key_t not in run_data or key_estimate not in run_data:
                continue
            times = run_data[key_t]
            estimate = run_data[key_estimate]
            stamps = np.asarray([item.stamp - offset for item in records])
            starts = np.asarray([item.start[:2] for item in records])
            valid = (stamps >= 0.0) & (stamps <= duration)
            if not np.any(valid):
                continue
            stamps = stamps[valid]
            starts = starts[valid]
            predicted = np.column_stack(
                [
                    np.interp(stamps, times, estimate[:, axis])
                    for axis in range(2)
                ]
            )
            squared.extend(np.sum((predicted - starts) ** 2, axis=1).tolist())
            used += len(stamps)
        if used < max(20, int(0.75 * len(targets))):
            return float("inf")
        return float(np.mean(squared))

    first_stamp = min(item.stamp for item in targets)
    coarse = np.linspace(first_stamp - 15.0, first_stamp + 2.0, 341)
    coarse_score = np.asarray([objective(value) for value in coarse])
    best = float(coarse[int(np.argmin(coarse_score))])
    fine = np.linspace(best - 0.08, best + 0.08, 321)
    fine_score = np.asarray([objective(value) for value in fine])
    return float(fine[int(np.argmin(fine_score))])


def time_alignment_diagnostics(targets, run_data, duration: float, offset: float):
    """Measure whether every logged planning start matches SLAM at that time."""
    residuals = []
    by_drone = defaultdict(list)
    for record in targets:
        key_t = f"t_{record.drone}"
        key_estimate = f"estimate_{record.drone}"
        if key_t not in run_data or key_estimate not in run_data:
            continue
        times = run_data[key_t]
        estimate = run_data[key_estimate]
        relative_t = record.stamp - offset
        if (
            relative_t < max(0.0, float(times[0]))
            or relative_t > min(duration, float(times[-1]))
        ):
            continue
        expected = np.asarray(
            [
                np.interp(relative_t, times, estimate[:, axis])
                for axis in range(2)
            ]
        )
        residual = float(np.linalg.norm(expected - record.start[:2]))
        residuals.append(residual)
        by_drone[record.drone].append(residual)

    if not residuals:
        raise RuntimeError("cannot align RACER log: no overlapping target records")

    def summarize(values):
        values = np.asarray(values, dtype=np.float64)
        return {
            "samples": int(values.size),
            "median_m": float(np.median(values)),
            "p90_m": float(np.percentile(values, 90.0)),
            "max_m": float(np.max(values)),
            "rmse_m": float(np.sqrt(np.mean(np.square(values)))),
        }

    diagnostics = summarize(residuals)
    diagnostics["by_drone"] = {
        str(drone): summarize(values)
        for drone, values in sorted(by_drone.items())
    }
    return diagnostics


def validate_time_alignment(diagnostics):
    """Reject a video instead of rendering visibly stale planning overlays."""
    if diagnostics["samples"] < 20:
        raise RuntimeError(
            "RACER log alignment rejected: fewer than 20 validation samples"
        )
    if diagnostics["median_m"] > ALIGN_MAX_MEDIAN_M or diagnostics["p90_m"] > ALIGN_MAX_P90_M:
        raise RuntimeError(
            "RACER log alignment rejected: planning-start/SLAM residual is "
            f"median={diagnostics['median_m']:.3f} m, "
            f"p90={diagnostics['p90_m']:.3f} m"
        )


def validate_rigid_transform(rotation, translation):
    rotation = np.asarray(rotation, dtype=np.float64)
    translation = np.asarray(translation, dtype=np.float64)
    if rotation.shape != (3, 3) or translation.shape != (3,):
        raise RuntimeError("invalid planner-to-Gazebo transform dimensions")
    orthogonality_error = float(
        np.linalg.norm(rotation.T @ rotation - np.eye(3), ord="fro")
    )
    determinant = float(np.linalg.det(rotation))
    if orthogonality_error > 1.0e-5 or abs(determinant - 1.0) > 1.0e-5:
        raise RuntimeError(
            "invalid planner-to-Gazebo rigid transform: "
            f"orthogonality_error={orthogonality_error:.3e}, "
            f"determinant={determinant:.6f}"
        )
    return {
        "orthogonality_error": orthogonality_error,
        "determinant": determinant,
    }


def transform_point(point, rotation, translation):
    if point is None or not np.all(np.isfinite(point)):
        return None
    return apply_transform(
        np.asarray(point, dtype=np.float64).reshape(1, 3),
        rotation,
        translation,
    )[0]


def hgrid_geometry(grid_id: int):
    """Return planner-frame bounds and level for RACER's two-level HGrid."""
    if 0 <= grid_id < 50:
        level = 1
        address = grid_id
        nx, ny = 10, 5
        size = 10.0
    elif 50 <= grid_id < 250:
        level = 2
        address = grid_id - 50
        nx, ny = 20, 10
        size = 5.0
    else:
        return None
    ix = address // ny
    iy = address % ny
    if ix < 0 or ix >= nx:
        return None
    xmin = -50.0 + ix * size
    ymin = -25.0 + iy * size
    return level, (xmin, xmin + size, ymin, ymin + size)


def transformed_polygon(bounds, rotation, translation, z=0.625):
    xmin, xmax, ymin, ymax = bounds
    points = np.asarray(
        [
            [xmin, ymin, z],
            [xmax, ymin, z],
            [xmax, ymax, z],
            [xmin, ymax, z],
        ],
        dtype=np.float64,
    )
    return apply_transform(points, rotation, translation)[:, :2]


def draw_building(axis, layout, sensor_z):
    for box in layout["boxes"]:
        cx, cy, cz = (float(value) for value in box["center"])
        sx, sy, sz = (float(value) for value in box["size"])
        if not (cz - 0.5 * sz <= sensor_z <= cz + 0.5 * sz):
            continue
        axis.add_patch(
            Rectangle(
                (cx - 0.5 * sx, cy - 0.5 * sy),
                sx,
                sy,
                facecolor="#3e4754",
                edgecolor="#d4dde9",
                linewidth=1.0,
                alpha=0.92,
                zorder=5,
            )
        )


def reconstruct_robot_visibility(run_data, robot_ids, masks, args):
    first_seen = {}
    for robot_id in robot_ids:
        times = run_data[f"t_{robot_id}"]
        truth = run_data[f"truth_{robot_id}"]
        chosen = select_motion_samples(
            times, truth, args.coverage_pose_step, args.coverage_max_dt
        )
        robot_first = np.full(masks["navigable"].shape, np.inf, np.float32)
        for index in chosen:
            visible = raycast_visible(
                truth[index, :2], masks, args.lidar_range, args.ray_count
            )
            newly_seen = visible & ~np.isfinite(robot_first)
            robot_first[newly_seen] = float(times[index])
        first_seen[robot_id] = robot_first
    return first_seen


def grid_cell_index(masks, rotation, translation):
    grid_x, grid_y = np.meshgrid(masks["x"], masks["y"])
    world = np.column_stack(
        (
            grid_x.ravel(),
            grid_y.ravel(),
            np.full(grid_x.size, 0.625),
        )
    )
    local = (world - translation) @ rotation
    ix = np.floor((local[:, 0] + 50.0) / 5.0).astype(np.int32)
    iy = np.floor((local[:, 1] + 25.0) / 5.0).astype(np.int32)
    valid = (ix >= 0) & (ix < 20) & (iy >= 0) & (iy < 10)
    ids = np.full(grid_x.size, -1, dtype=np.int32)
    ids[valid] = 50 + ix[valid] * 10 + iy[valid]
    return ids.reshape(grid_x.shape)


def per_hgrid_first_times(first_seen, navigable, cell_ids):
    output = {}
    for robot_id, robot_first in first_seen.items():
        output[robot_id] = {}
        for grid_id in range(50, 250):
            mask = navigable & (cell_ids == grid_id)
            values = np.sort(robot_first[mask & np.isfinite(robot_first)])
            output[robot_id][grid_id] = (
                values.astype(np.float32), int(np.count_nonzero(mask))
            )
    return output


def latest_record(records, times, current_t):
    index = bisect.bisect_right(times, current_t) - 1
    return records[index] if index >= 0 else None


def padded_bounds(points, minimum_span=25.0, padding=3.0):
    values = np.asarray(points, dtype=np.float64)
    finite = values[np.all(np.isfinite(values), axis=1)]
    if not len(finite):
        return (-25.0, 25.0, -15.0, 15.0)
    xmin, ymin = np.min(finite[:, :2], axis=0) - padding
    xmax, ymax = np.max(finite[:, :2], axis=0) + padding
    if xmax - xmin < minimum_span:
        middle = 0.5 * (xmin + xmax)
        xmin, xmax = middle - 0.5 * minimum_span, middle + 0.5 * minimum_span
    if ymax - ymin < minimum_span:
        middle = 0.5 * (ymin + ymax)
        ymin, ymax = middle - 0.5 * minimum_span, middle + 0.5 * minimum_span
    return (
        5.0 * math.floor(xmin / 5.0),
        5.0 * math.ceil(xmax / 5.0),
        5.0 * math.floor(ymin / 5.0),
        5.0 * math.ceil(ymax / 5.0),
    )


def coverage_fraction(first_times, total, current_t):
    if total <= 0:
        return 0.0
    return float(np.searchsorted(first_times, current_t, side="right")) / total


def render(args):
    with np.load(args.run, allow_pickle=False) as source:
        run_data = {key: np.asarray(source[key]) for key in source.files}
    layout = load_layout(args.layout)
    robot_ids = [int(value) for value in run_data["robot_ids"]]
    duration = max(float(run_data[f"t_{item}"][-1]) for item in robot_ids)
    rotation = np.asarray(run_data["first_pose_rotation"], dtype=np.float64)
    translation = np.asarray(
        run_data["first_pose_translation"], dtype=np.float64
    )
    transform_diagnostics = validate_rigid_transform(rotation, translation)

    targets, relevance = parse_racer_log(args.rosout)
    fitted_offset = fit_log_time_offset(targets, run_data, duration)
    offset = fitted_offset if args.log_time_offset is None else args.log_time_offset
    alignment_diagnostics = time_alignment_diagnostics(
        targets, run_data, duration, offset
    )
    validate_time_alignment(alignment_diagnostics)
    for record in targets:
        record.relative_t = record.stamp - offset
        record.start_world = transform_point(record.start, rotation, translation)
        record.target_world = transform_point(
            record.target, rotation, translation
        )
        record.frontier_world = transform_point(
            record.frontier_average, rotation, translation
        )
    for record in relevance:
        record.relative_t = record.stamp - offset

    target_by_robot = {
        robot_id: sorted(
            [
                item
                for item in targets
                if item.drone == robot_id
                and -2.0 <= item.relative_t <= duration + 2.0
            ],
            key=lambda item: item.relative_t,
        )
        for robot_id in robot_ids
    }
    target_times = {
        robot_id: [item.relative_t for item in target_by_robot[robot_id]]
        for robot_id in robot_ids
    }
    relevance_by_robot_level = {}
    for robot_id in robot_ids:
        for level in (1, 2):
            values = sorted(
                [
                    item
                    for item in relevance
                    if item.drone == robot_id and item.level == level
                ],
                key=lambda item: item.relative_t,
            )
            relevance_by_robot_level[(robot_id, level)] = (
                values,
                [item.relative_t for item in values],
            )

    masks = build_masks(
        layout, args.grid_resolution, args.robot_radius, args.sensor_z
    )
    first_seen = reconstruct_robot_visibility(
        run_data, robot_ids, masks, args
    )
    cell_ids = grid_cell_index(masks, rotation, translation)
    hgrid_visibility = per_hgrid_first_times(
        first_seen, masks["navigable"], cell_ids
    )

    target_grid_times = {robot_id: defaultdict(list) for robot_id in robot_ids}
    for robot_id, records in target_by_robot.items():
        for record in records:
            target_grid_times[robot_id][record.grid_id].append(record.relative_t)

    panel_bounds = {}
    for robot_id in robot_ids:
        points = [run_data[f"truth_{robot_id}"][:, :2]]
        target_points = [
            item.target_world[:2]
            for item in target_by_robot[robot_id]
            if item.target_world is not None
        ]
        if target_points:
            points.append(np.asarray(target_points))
        panel_bounds[robot_id] = padded_bounds(np.vstack(points))

    plt.rcParams["font.family"] = "Noto Sans CJK JP"
    plt.style.use("dark_background")
    figure, axes = plt.subplots(
        1,
        len(robot_ids),
        figsize=(19.2, 10.8),
        facecolor="#080d14",
        squeeze=False,
    )
    axes = axes[0]
    figure.subplots_adjust(
        left=0.035, right=0.965, bottom=0.105, top=0.82, wspace=0.12
    )
    cmap = plt.get_cmap("RdYlGn")
    norm = Normalize(0.0, 1.0)
    colors = plt.cm.tab10(np.arange(len(robot_ids)) % 10)
    artists = {}

    for axis, robot_id, robot_color in zip(axes, robot_ids, colors):
        bounds = panel_bounds[robot_id]
        axis.set_xlim(bounds[0], bounds[1])
        axis.set_ylim(bounds[2], bounds[3])
        axis.set_aspect("equal", adjustable="box")
        axis.set_facecolor("#111925")
        axis.grid(False)
        axis.set_xlabel("Gazebo世界坐标 x [m]")
        axis.set_ylabel("Gazebo世界坐标 y [m]")
        axis.set_title(
            f"机器狗 {robot_id}：HGrid、观测区域与当前目标",
            fontsize=15,
            fontweight="bold",
            pad=10,
        )

        grid_patches = {}
        grid_text = {}
        for grid_id in range(50, 250):
            geometry = hgrid_geometry(grid_id)
            polygon = transformed_polygon(geometry[1], rotation, translation)
            center = np.mean(polygon, axis=0)
            if not (
                bounds[0] - 5.0 <= center[0] <= bounds[1] + 5.0
                and bounds[2] - 5.0 <= center[1] <= bounds[3] + 5.0
            ):
                continue
            patch = Polygon(
                polygon,
                closed=True,
                facecolor=cmap(0.0),
                edgecolor="#65758c",
                linewidth=0.65,
                alpha=0.30,
                zorder=1,
            )
            axis.add_patch(patch)
            label = axis.text(
                center[0],
                center[1],
                "",
                ha="center",
                va="center",
                fontsize=6.3,
                color="#f3f7fb",
                alpha=0.88,
                zorder=3,
            )
            grid_patches[grid_id] = patch
            grid_text[grid_id] = label

        for coarse_id in range(50):
            geometry = hgrid_geometry(coarse_id)
            polygon = transformed_polygon(geometry[1], rotation, translation)
            center = np.mean(polygon, axis=0)
            if not (
                bounds[0] - 10.0 <= center[0] <= bounds[1] + 10.0
                and bounds[2] - 10.0 <= center[1] <= bounds[3] + 10.0
            ):
                continue
            axis.add_patch(
                Polygon(
                    polygon,
                    closed=True,
                    fill=False,
                    edgecolor="#d2dae6",
                    linewidth=1.35,
                    alpha=0.72,
                    zorder=4,
                )
            )

        draw_building(axis, layout, args.sensor_z)
        times = run_data[f"t_{robot_id}"]
        truth = run_data[f"truth_{robot_id}"]
        trajectory, = axis.plot(
            [], [], color=robot_color, linewidth=2.5, zorder=7,
            label="Gazebo真实轨迹"
        )
        robot_marker = axis.scatter(
            [], [], marker="D", s=150, color=robot_color,
            edgecolor="white", linewidth=1.4, zorder=13,
        )
        target_history = axis.scatter(
            [], [], marker="o", s=16, facecolor="none",
            edgecolor="#f6b44b", linewidth=0.65, alpha=0.38, zorder=8,
        )
        target_line, = axis.plot(
            [], [], color="#ffdc5e", linewidth=1.8,
            linestyle="--", alpha=0.95, zorder=10,
        )
        frontier_line, = axis.plot(
            [], [], color="#67e8f9", linewidth=1.2,
            linestyle=":", alpha=0.95, zorder=10,
        )
        viewpoint_marker = axis.scatter(
            [], [], marker="*", s=230, color="#ffcf4a",
            edgecolor="#271900", linewidth=1.0, zorder=14,
        )
        fallback_marker = axis.scatter(
            [], [], marker="s", s=120, color="#ff8c42",
            edgecolor="white", linewidth=1.0, zorder=14,
        )
        frontier_marker = axis.scatter(
            [], [], marker="X", s=120, color="#67e8f9",
            edgecolor="#063944", linewidth=1.0, zorder=14,
        )
        target_grid_patch = Polygon(
            np.zeros((4, 2)), closed=True, fill=False,
            edgecolor="#ff365e", linewidth=3.2, alpha=0.98, zorder=12,
        )
        axis.add_patch(target_grid_patch)
        stats = axis.text(
            0.012,
            0.985,
            "",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=9.0,
            linespacing=1.35,
            color="#f1f5fb",
            bbox={
                "boxstyle": "round,pad=0.45",
                "facecolor": "#07101d",
                "edgecolor": "#718099",
                "alpha": 0.88,
            },
            zorder=20,
        )
        axis.legend(
            handles=[
                Line2D([0], [0], color=robot_color, lw=2.5,
                       label="Gazebo真实轨迹"),
                Line2D([0], [0], marker="*", color="none",
                       markerfacecolor="#ffcf4a", markersize=12,
                       label="有frontier的当前观测目标"),
                Line2D([0], [0], marker="s", color="none",
                       markerfacecolor="#ff8c42", markersize=8,
                       label="无frontier时的引导目标"),
                Line2D([0], [0], marker="X", color="none",
                       markerfacecolor="#67e8f9", markersize=8,
                       label="frontier中心"),
            ],
            loc="lower left",
            fontsize=8.0,
            framealpha=0.78,
        )
        artists[robot_id] = {
            "axis": axis,
            "times": times,
            "truth": truth,
            "trajectory": trajectory,
            "robot": robot_marker,
            "target_history": target_history,
            "target_line": target_line,
            "frontier_line": frontier_line,
            "viewpoint": viewpoint_marker,
            "fallback": fallback_marker,
            "frontier": frontier_marker,
            "target_grid": target_grid_patch,
            "stats": stats,
            "grid_patches": grid_patches,
            "grid_text": grid_text,
        }

    figure.suptitle(
        "双机器狗RACER探索诊断：每台机器狗独立显示HGrid与规划目标",
        fontsize=19,
        fontweight="bold",
        y=0.975,
    )
    figure.text(
        0.5,
        0.925,
        "细格F=5×5 m，粗格边界=10×10 m；格内百分比=该机器狗截至当前从真实轨迹可见的可通行地面比例（仅用于离线评估）\n"
        f"日志与轨迹已自动配准：{alignment_diagnostics['samples']}个目标记录，"
        f"位置残差中位数={alignment_diagnostics['median_m']:.3f} m，"
        f"P90={alignment_diagnostics['p90_m']:.3f} m",
        ha="center",
        va="center",
        fontsize=10.0,
        color="#d6e0ec",
    )
    colorbar_axis = figure.add_axes([0.18, 0.045, 0.64, 0.022])
    scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    colorbar = figure.colorbar(scalar, cax=colorbar_axis, orientation="horizontal")
    colorbar.set_label(
        "单机真值观测比例：0%=该机器狗从未看见，100%=该HGrid内所有可通行地面都被该机器狗看见",
        fontsize=9.5,
    )

    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(
        fps=args.fps,
        codec="libx264",
        bitrate=6500,
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )
    frame_count = max(2, int(math.ceil(duration / args.speedup * args.fps)))
    pause_frames = int(round(args.fps))
    last_frame = None
    with writer.saving(figure, str(output), dpi=90):
        for frame_index in range(frame_count + pause_frames):
            current_t = (
                duration
                if frame_index >= frame_count
                else min(duration, frame_index * args.speedup / args.fps)
            )
            for robot_id in robot_ids:
                item = artists[robot_id]
                times = item["times"]
                truth = item["truth"]
                end = int(np.searchsorted(times, current_t, side="right"))
                truth_now = truth[:end]
                item["trajectory"].set_data(
                    truth_now[:, 0], truth_now[:, 1]
                )
                if len(truth_now):
                    robot_xy = truth_now[-1, :2]
                    item["robot"].set_offsets(robot_xy.reshape(1, 2))
                    path_m = float(
                        np.linalg.norm(
                            np.diff(truth_now[:, :2], axis=0), axis=1
                        ).sum()
                    )
                else:
                    robot_xy = np.empty((0,))
                    item["robot"].set_offsets(np.empty((0, 2)))
                    path_m = 0.0

                robot_first = first_seen[robot_id]
                observed_now = (
                    masks["navigable"]
                    & np.isfinite(robot_first)
                    & (robot_first <= current_t)
                )
                self_fraction = float(np.count_nonzero(observed_now)) / max(
                    int(np.count_nonzero(masks["navigable"])), 1
                )
                for grid_id, patch in item["grid_patches"].items():
                    first_values, total = hgrid_visibility[robot_id][grid_id]
                    fraction = coverage_fraction(
                        first_values, total, current_t
                    )
                    patch.set_facecolor(cmap(fraction))
                    selections = int(
                        np.searchsorted(
                            target_grid_times[robot_id].get(grid_id, []),
                            current_t,
                            side="right",
                        )
                    )
                    item["grid_text"][grid_id].set_text(
                        f"F{grid_id}\n{100.0 * fraction:.0f}%"
                        + (f" 选{selections}" if selections else "")
                    )

                records = target_by_robot[robot_id]
                records_t = target_times[robot_id]
                current_target = latest_record(records, records_t, current_t)
                history_end = bisect.bisect_right(records_t, current_t)
                history_points = [
                    record.target_world[:2]
                    for record in records[:history_end]
                    if record.target_world is not None
                ]
                item["target_history"].set_offsets(
                    np.asarray(history_points)
                    if history_points
                    else np.empty((0, 2))
                )

                coarse_records, coarse_times = relevance_by_robot_level[
                    (robot_id, 1)
                ]
                fine_records, fine_times = relevance_by_robot_level[
                    (robot_id, 2)
                ]
                coarse = latest_record(coarse_records, coarse_times, current_t)
                fine = latest_record(fine_records, fine_times, current_t)
                coarse_text = (
                    f"粗HGrid有效/激活={coarse.relevant}/{coarse.active}"
                    if coarse is not None
                    else "粗HGrid状态尚未记录"
                )
                fine_text = (
                    f"细HGrid有效/激活={fine.relevant}/{fine.active}，"
                    f"本机到过={fine.visited}"
                    if fine is not None
                    else "细HGrid状态尚未记录"
                )

                if current_target is None or current_target.target_world is None:
                    item["target_line"].set_data([], [])
                    item["frontier_line"].set_data([], [])
                    item["viewpoint"].set_offsets(np.empty((0, 2)))
                    item["fallback"].set_offsets(np.empty((0, 2)))
                    item["frontier"].set_offsets(np.empty((0, 2)))
                    item["target_grid"].set_xy(np.zeros((4, 2)))
                    target_text = "当前还没有RACER目标输出"
                else:
                    target_xy = current_target.target_world[:2]
                    if len(robot_xy):
                        item["target_line"].set_data(
                            [robot_xy[0], target_xy[0]],
                            [robot_xy[1], target_xy[1]],
                        )
                    if current_target.frontier_id >= 0:
                        item["viewpoint"].set_offsets(target_xy.reshape(1, 2))
                        item["fallback"].set_offsets(np.empty((0, 2)))
                        target_kind = "frontier观测位置"
                    else:
                        item["fallback"].set_offsets(target_xy.reshape(1, 2))
                        item["viewpoint"].set_offsets(np.empty((0, 2)))
                        target_kind = "无frontier的HGrid引导位置"
                    if current_target.frontier_world is not None:
                        frontier_xy = current_target.frontier_world[:2]
                        item["frontier"].set_offsets(
                            frontier_xy.reshape(1, 2)
                        )
                        item["frontier_line"].set_data(
                            [target_xy[0], frontier_xy[0]],
                            [target_xy[1], frontier_xy[1]],
                        )
                    else:
                        item["frontier"].set_offsets(np.empty((0, 2)))
                        item["frontier_line"].set_data([], [])
                    geometry = hgrid_geometry(current_target.grid_id)
                    if geometry is not None:
                        target_polygon = transformed_polygon(
                            geometry[1], rotation, translation
                        )
                        item["target_grid"].set_xy(target_polygon)
                        level_name = "粗10m" if geometry[0] == 1 else "细5m"
                    else:
                        item["target_grid"].set_xy(np.zeros((4, 2)))
                        level_name = "未知层级"
                    target_text = (
                        f"当前目标：HGrid {current_target.grid_id}（{level_name}）\n"
                        f"目标类型：{target_kind}；frontier编号="
                        f"{current_target.frontier_id}\n"
                        f"RACER当前记录：HGrid剩余unknown="
                        f"{current_target.hgrid_unknown}，HGrid预测收益="
                        f"{current_target.hgrid_gain}，viewpoint预测收益="
                        f"{current_target.view_gain}"
                    )

                item["stats"].set_text(
                    f"时间={current_t:6.1f}/{duration:6.1f} s；"
                    f"本机路径={path_m:6.1f} m；"
                    f"本机观测整图={100.0 * self_fraction:5.1f}%\n"
                    f"{coarse_text}；{fine_text}\n{target_text}\n"
                    "格内“选N”=截至当前该HGrid被输出为下一目标的次数；"
                    "重复重规划也会累计"
                )

            writer.grab_frame()
            if frame_index == frame_count - 1:
                figure.canvas.draw()
                last_frame = np.asarray(figure.canvas.buffer_rgba()).copy()

    if last_frame is not None:
        args.thumbnail.parent.mkdir(parents=True, exist_ok=True)
        plt.imsave(args.thumbnail, last_frame)
    plt.close(figure)
    metadata = {
        "output": str(output),
        "duration_sim_s": duration,
        "video_duration_s": (frame_count + pause_frames) / args.fps,
        "log_time_offset_s": offset,
        "auto_fitted_log_time_offset_s": fitted_offset,
        "manual_log_time_offset_used": args.log_time_offset is not None,
        "time_alignment": alignment_diagnostics,
        "planner_to_gazebo_transform": transform_diagnostics,
        "target_records": {
            str(robot_id): len(target_by_robot[robot_id])
            for robot_id in robot_ids
        },
        "coverage_definition": (
            "per-robot Gazebo-truth ray-visible navigable area; evaluation only"
        ),
    }
    metadata_path = (
        args.metadata
        if args.metadata is not None
        else output.with_suffix(".metadata.json")
    )
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


def main():
    global ALIGN_MAX_MEDIAN_M, ALIGN_MAX_P90_M
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--layout", type=Path, required=True)
    parser.add_argument("--rosout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--thumbnail", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--speedup", type=float, default=10.0)
    parser.add_argument("--grid-resolution", type=float, default=0.25)
    parser.add_argument("--lidar-range", type=float, default=10.0)
    parser.add_argument("--robot-radius", type=float, default=0.40)
    parser.add_argument("--sensor-z", type=float, default=0.645)
    parser.add_argument("--ray-count", type=int, default=720)
    parser.add_argument("--coverage-pose-step", type=float, default=0.40)
    parser.add_argument("--coverage-max-dt", type=float, default=1.0)
    parser.add_argument("--log-time-offset", type=float)
    parser.add_argument("--max-align-median", type=float, default=ALIGN_MAX_MEDIAN_M)
    parser.add_argument("--max-align-p90", type=float, default=ALIGN_MAX_P90_M)
    args = parser.parse_args()
    ALIGN_MAX_MEDIAN_M = float(args.max_align_median)
    ALIGN_MAX_P90_M = float(args.max_align_p90)
    render(args)


if __name__ == "__main__":
    main()
