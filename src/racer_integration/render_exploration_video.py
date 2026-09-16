#!/usr/bin/env python3
"""Analyse and render a RACER/Swarm-LIO2 Gazebo exploration recording."""

from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FFMpegWriter  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402


DRONE_XY_RADIUS = 0.356
DRONE_Z_HALF = 0.065


def parse_boxes(world_path: Path):
    root = ET.parse(world_path).getroot()
    boxes = []
    for link in root.findall(".//model[@name='racer_env']/link"):
        pose_text = link.findtext("pose", default="0 0 0 0 0 0")
        pose = np.asarray([float(value) for value in pose_text.split()[:3]])
        size_text = link.findtext("collision/geometry/box/size")
        if not size_text:
            continue
        size = np.asarray([float(value) for value in size_text.split()[:3]])
        boxes.append((link.attrib.get("name", ""), pose, size))
    return boxes


def generate_surface_reference(boxes, resolution: float) -> np.ndarray:
    keys = set()
    for _, center, size in boxes:
        lower = center - 0.5 * size
        upper = center + 0.5 * size
        axes = [
            np.arange(
                math.floor(lower[axis] / resolution),
                math.ceil(upper[axis] / resolution) + 1,
                dtype=np.int32,
            )
            for axis in range(3)
        ]
        for fixed_axis in range(3):
            other = [axis for axis in range(3) if axis != fixed_axis]
            first_grid, second_grid = np.meshgrid(
                axes[other[0]], axes[other[1]], indexing="ij"
            )
            for fixed_value in (axes[fixed_axis][0], axes[fixed_axis][-1]):
                face = np.zeros((first_grid.size, 3), dtype=np.int32)
                face[:, fixed_axis] = fixed_value
                face[:, other[0]] = first_grid.reshape(-1)
                face[:, other[1]] = second_grid.reshape(-1)
                keys.update(map(tuple, face.tolist()))
    if not keys:
        return np.empty((0, 3), dtype=np.float64)
    return np.asarray(list(keys), dtype=np.float64) * resolution


def collision_analysis(data, robot_ids, boxes):
    static = {}
    any_static_overlap = False
    for robot_id in robot_ids:
        truth = np.asarray(data[f"truth_{robot_id}"], dtype=np.float64)
        times = np.asarray(data[f"t_{robot_id}"], dtype=np.float64)
        overlap_samples = 0
        first_overlap = None
        last_overlap = None
        max_horizontal_overlap = 0.0
        closest_clearance = float("inf")
        closest_box = None
        for sample_index, position in enumerate(truth):
            for box_name, center, size in boxes:
                half = 0.5 * size
                horizontal_axis_gap = np.maximum(
                    np.abs(position[:2] - center[:2]) - half[:2], 0.0
                )
                horizontal_center_gap = float(np.linalg.norm(horizontal_axis_gap))
                vertical_center_gap = max(
                    float(abs(position[2] - center[2]) - half[2]), 0.0
                )
                horizontal_envelope_gap = max(
                    horizontal_center_gap - DRONE_XY_RADIUS, 0.0
                )
                vertical_envelope_gap = max(
                    vertical_center_gap - DRONE_Z_HALF, 0.0
                )
                clearance = math.hypot(
                    horizontal_envelope_gap, vertical_envelope_gap
                )
                if clearance < closest_clearance:
                    closest_clearance = clearance
                    closest_box = box_name
                overlap = (
                    horizontal_center_gap <= DRONE_XY_RADIUS
                    and vertical_center_gap <= DRONE_Z_HALF
                )
                if overlap:
                    overlap_samples += 1
                    if first_overlap is None:
                        first_overlap = float(times[sample_index])
                    last_overlap = float(times[sample_index])
                    max_horizontal_overlap = max(
                        max_horizontal_overlap,
                        DRONE_XY_RADIUS - horizontal_center_gap,
                    )
                    break
        overlapped = overlap_samples > 0
        any_static_overlap = any_static_overlap or overlapped
        static[f"bot{robot_id}"] = {
            "envelope_overlap_detected": overlapped,
            "overlap_samples": overlap_samples,
            "first_overlap_wall_s": first_overlap,
            "last_overlap_wall_s": last_overlap,
            "max_horizontal_overlap_m": max_horizontal_overlap,
            "closest_envelope_clearance_m": (
                None if not math.isfinite(closest_clearance) else closest_clearance
            ),
            "closest_box": closest_box,
        }
    return {
        "method": (
            "offline Gazebo-truth geometric broad phase: horizontal rotor disc "
            "radius 0.356 m and vertical body half-height 0.065 m against static "
            "world AABBs; this is not a Gazebo contact-sensor measurement"
        ),
        "any_static_envelope_overlap": any_static_overlap,
        "robots": static,
    }


def apply_transform(points, rotation, translation):
    return np.asarray(points, dtype=np.float64) @ rotation.T + translation


def update_analysis(summary_path, data, boxes, coverage_resolution):
    with summary_path.open("r", encoding="utf-8") as stream:
        summary = json.load(stream)
    robot_ids = [int(value) for value in summary["robot_ids"]]
    voxels = np.asarray(data["voxel_xyz"], dtype=np.float64)
    first_seen = np.asarray(data["voxel_first_t"], dtype=np.float64)

    reference = generate_surface_reference(boxes, coverage_resolution)
    matched_reference = np.full(len(voxels), -1, dtype=np.int64)
    if len(reference) and len(voxels):
        tree = cKDTree(reference)
        distances, indices = tree.query(voxels, k=1)
        valid = distances <= coverage_resolution
        matched_reference[valid] = indices[valid]
    observed_reference = np.unique(matched_reference[matched_reference >= 0])
    coverage = (
        100.0 * len(observed_reference) / len(reference) if len(reference) else 0.0
    )

    collision = collision_analysis(data, robot_ids, boxes)
    inter_uav_collision = any(
        entry.get("contact_detected", False)
        for entry in summary.get("inter_uav_clearance", {}).values()
    )
    summary["surface_coverage_proxy"] = {
        "description": (
            "fraction of 1 m static-world surface reference cells observed by "
            "the aligned, accumulated registered LiDAR scans"
        ),
        "resolution_m": coverage_resolution,
        "reference_cells": int(len(reference)),
        "observed_reference_cells": int(len(observed_reference)),
        "coverage_percent": coverage,
        "coverage_percent_per_wall_min": coverage
        * 60.0
        / max(float(summary["duration_wall_s"]), 1e-9),
    }
    summary["static_collision"] = collision
    summary.pop("any_collision_detected", None)
    summary["collision_assessment"] = {
        "physical_contact_sensor_available": False,
        "physical_collision_confirmed": None,
        "inter_uav_envelope_overlap_detected": bool(inter_uav_collision),
        "static_envelope_overlap_detected": bool(
            collision["any_static_envelope_overlap"]
        ),
        "any_geometric_envelope_overlap_detected": bool(
            inter_uav_collision or collision["any_static_envelope_overlap"]
        ),
        "interpretation": (
            "geometric envelope overlap is a conservative collision-risk proxy; "
            "it does not by itself prove a physics-engine contact"
        ),
    }
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
    return summary, reference, matched_reference, first_seen


def draw_world_xy(ax, boxes, bounds, z_range):
    count = 0
    for _, center, size in boxes:
        lower = center - 0.5 * size
        upper = center + 0.5 * size
        if upper[2] < z_range[0] or lower[2] > z_range[1]:
            continue
        if (
            upper[0] < bounds[0]
            or lower[0] > bounds[1]
            or upper[1] < bounds[2]
            or lower[1] > bounds[3]
        ):
            continue
        ax.add_patch(
            Rectangle(
                (lower[0], lower[1]),
                size[0],
                size[1],
                fill=False,
                edgecolor="#77808f",
                linewidth=0.45,
                alpha=0.24,
                zorder=0,
            )
        )
        count += 1
        if count >= 220:
            break


def draw_world_xz(ax, boxes, bounds, y_range):
    count = 0
    for _, center, size in boxes:
        lower = center - 0.5 * size
        upper = center + 0.5 * size
        if upper[1] < y_range[0] or lower[1] > y_range[1]:
            continue
        if (
            upper[0] < bounds[0]
            or lower[0] > bounds[1]
            or upper[2] < bounds[2]
            or lower[2] > bounds[3]
        ):
            continue
        ax.add_patch(
            Rectangle(
                (lower[0], lower[2]),
                size[0],
                size[2],
                fill=False,
                edgecolor="#77808f",
                linewidth=0.45,
                alpha=0.24,
                zorder=0,
            )
        )
        count += 1
        if count >= 220:
            break


def render_video(
    data,
    summary,
    boxes,
    matched_reference,
    first_seen,
    output_video,
    thumbnail,
    fps,
    speedup,
):
    robot_ids = [int(value) for value in summary["robot_ids"]]
    duration = float(summary["duration_wall_s"])
    rotation = np.asarray(data["first_pose_rotation"], dtype=np.float64)
    translation = np.asarray(data["first_pose_translation"], dtype=np.float64)
    voxels = np.asarray(data["voxel_xyz"], dtype=np.float64)
    voxel_robot = np.asarray(data["voxel_robot"], dtype=np.int32)

    trajectories = {}
    all_positions = []
    for robot_id in robot_ids:
        times = np.asarray(data[f"t_{robot_id}"], dtype=np.float64)
        truth = np.asarray(data[f"truth_{robot_id}"], dtype=np.float64)
        estimate = apply_transform(
            data[f"estimate_{robot_id}"], rotation, translation
        )
        trajectories[robot_id] = (times, estimate, truth)
        if len(truth):
            all_positions.append(truth)
    if len(voxels):
        all_positions.append(voxels)
    combined = (
        np.vstack(all_positions)
        if all_positions
        else np.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    )
    lower = np.percentile(combined, 0.5, axis=0) - np.asarray([2.0, 2.0, 2.0])
    upper = np.percentile(combined, 99.5, axis=0) + np.asarray([2.0, 2.0, 2.0])
    lower = np.maximum(lower, 0.0)
    upper = np.minimum(upper, 50.0)
    if np.any(upper - lower < 5.0):
        upper = np.maximum(upper, lower + 5.0)

    order = np.argsort(first_seen)
    voxels = voxels[order]
    first_seen = first_seen[order]
    voxel_robot = voxel_robot[order]
    matched_reference = matched_reference[order]

    # Coverage bookkeeping must use EVERY recorded voxel; the drawing stride
    # below only thins what is plotted.  (Computing it on the thinned set
    # under-reported the on-screen coverage by ~40% on a 940k-voxel run.)
    reference_first_seen = {}
    for timestamp, reference_index in zip(first_seen, matched_reference):
        if reference_index < 0:
            continue
        reference_first_seen[reference_index] = min(
            reference_first_seen.get(reference_index, float("inf")), float(timestamp)
        )

    stride = 1
    if len(voxels) > 45000:
        stride = int(math.ceil(len(voxels) / 45000))
        voxels = voxels[::stride]
        first_seen = first_seen[::stride]
        voxel_robot = voxel_robot[::stride]
        matched_reference = matched_reference[::stride]
    coverage_event_t = np.sort(
        np.asarray(list(reference_first_seen.values()), dtype=np.float64)
    )
    reference_count = int(
        summary["surface_coverage_proxy"]["reference_cells"]
    )
    coverage_event_y = (
        100.0 * np.arange(1, len(coverage_event_t) + 1) / max(reference_count, 1)
    )

    colors = plt.cm.tab10(np.linspace(0.0, 0.9, max(len(robot_ids), 1)))
    robot_color = {robot_id: colors[index] for index, robot_id in enumerate(robot_ids)}

    plt.style.use("dark_background")
    figure = plt.figure(figsize=(16, 9), facecolor="#10141d")
    grid = figure.add_gridspec(
        2, 2, width_ratios=[2.05, 1.0], height_ratios=[1.0, 1.0],
        left=0.055, right=0.975, top=0.90, bottom=0.08, wspace=0.19, hspace=0.24
    )
    ax_xy = figure.add_subplot(grid[:, 0])
    ax_xz = figure.add_subplot(grid[0, 1])
    ax_metric = figure.add_subplot(grid[1, 1])

    for axis in (ax_xy, ax_xz, ax_metric):
        axis.set_facecolor("#161c27")
        axis.grid(True, color="#384255", linewidth=0.45, alpha=0.45)

    ax_xy.set_title("Top view: accumulated registered LiDAR map and trajectories")
    ax_xy.set_xlabel("Gazebo x [m]")
    ax_xy.set_ylabel("Gazebo y [m]")
    ax_xy.set_aspect("equal", adjustable="box")
    ax_xy.set_xlim(lower[0], upper[0])
    ax_xy.set_ylim(lower[1], upper[1])
    draw_world_xy(ax_xy, boxes, (lower[0], upper[0], lower[1], upper[1]), (lower[2], upper[2]))

    ax_xz.set_title("Vertical view (proves 3D motion)")
    ax_xz.set_xlabel("Gazebo x [m]")
    ax_xz.set_ylabel("Gazebo z [m]")
    ax_xz.set_xlim(lower[0], upper[0])
    ax_xz.set_ylim(lower[2], upper[2])
    draw_world_xz(ax_xz, boxes, (lower[0], upper[0], lower[2], upper[2]), (lower[1], upper[1]))

    map_xy = ax_xy.scatter(
        [], [], s=2.0, c=[], cmap="turbo", vmin=lower[2], vmax=upper[2],
        alpha=0.78, linewidths=0, zorder=1, label="LiDAR surface voxels"
    )
    map_xz = ax_xz.scatter(
        [], [], s=1.4, c=[], cmap="turbo", vmin=lower[2], vmax=upper[2],
        alpha=0.72, linewidths=0, zorder=1
    )

    line_artists = {}
    marker_artists = {}
    for robot_id in robot_ids:
        color = robot_color[robot_id]
        estimated_xy, = ax_xy.plot(
            [], [], color=color, linewidth=2.0, label=f"bot{robot_id} SLAM", zorder=4
        )
        truth_xy, = ax_xy.plot(
            [], [], color=color, linewidth=1.0, linestyle="--", alpha=0.65,
            label=f"bot{robot_id} Gazebo", zorder=3
        )
        estimated_xz, = ax_xz.plot([], [], color=color, linewidth=1.8, zorder=4)
        truth_xz, = ax_xz.plot(
            [], [], color=color, linewidth=0.9, linestyle="--", alpha=0.6, zorder=3
        )
        current_xy, = ax_xy.plot(
            [], [], marker="o", markersize=8, color=color,
            markeredgecolor="white", markeredgewidth=0.8, zorder=6
        )
        current_xz, = ax_xz.plot(
            [], [], marker="o", markersize=6, color=color,
            markeredgecolor="white", markeredgewidth=0.7, zorder=6
        )
        line_artists[robot_id] = (
            estimated_xy, truth_xy, estimated_xz, truth_xz
        )
        marker_artists[robot_id] = (current_xy, current_xz)

    handles, labels = ax_xy.get_legend_handles_labels()
    ax_xy.legend(
        handles, labels, loc="upper right", fontsize=7.5, ncol=2,
        framealpha=0.65, facecolor="#10141d"
    )

    ax_metric.set_title("Exploration progress proxy")
    ax_metric.set_xlabel("recording wall time [s]")
    ax_metric.set_ylabel("static surface coverage [%]")
    ax_metric.set_xlim(0.0, max(duration, 1.0))
    max_coverage = max(
        float(summary["surface_coverage_proxy"]["coverage_percent"]) * 1.15,
        0.1,
    )
    ax_metric.set_ylim(0.0, max_coverage)
    if len(coverage_event_t):
        ax_metric.plot(
            coverage_event_t, coverage_event_y, color="#657083",
            linewidth=1.0, alpha=0.65
        )
    progress_line, = ax_metric.plot([], [], color="#31d7a0", linewidth=2.4)
    time_cursor = ax_metric.axvline(0.0, color="#ffcc66", linewidth=1.2, alpha=0.9)

    title = figure.suptitle(
        f"Gazebo LiDAR Exploration — {len(robot_ids)} UAV"
        f"{'s' if len(robot_ids) != 1 else ''} — RACER + Swarm-LIO2",
        fontsize=18, fontweight="bold", color="#f2f5fa"
    )
    stats_text = figure.text(
        0.055, 0.945, "", ha="left", va="top", fontsize=10.5, color="#d7deea"
    )
    note_text = figure.text(
        0.975, 0.945,
        f"video {speedup:g}x | solid: SLAM/common frame | dashed: Gazebo truth",
        ha="right", va="top", fontsize=9.5, color="#aeb8c8"
    )
    _ = (title, note_text)

    output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(
        fps=fps,
        codec="libx264",
        bitrate=4500,
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        metadata={"title": "RACER + Swarm-LIO2 Gazebo exploration"},
    )
    frame_count = max(2, int(math.ceil(duration / speedup * fps)))
    pause_frames = int(1.5 * fps)

    with writer.saving(figure, str(output_video), dpi=100):
        for frame_index in range(frame_count + pause_frames):
            if frame_index >= frame_count:
                current_t = duration
            else:
                current_t = min(duration, frame_index * speedup / fps)
            voxel_end = int(np.searchsorted(first_seen, current_t, side="right"))
            current_voxels = voxels[:voxel_end]
            if len(current_voxels):
                map_xy.set_offsets(current_voxels[:, :2])
                map_xy.set_array(current_voxels[:, 2])
                map_xz.set_offsets(current_voxels[:, [0, 2]])
                map_xz.set_array(current_voxels[:, 2])
            else:
                map_xy.set_offsets(np.empty((0, 2)))
                map_xy.set_array(np.empty((0,)))
                map_xz.set_offsets(np.empty((0, 2)))
                map_xz.set_array(np.empty((0,)))

            total_distance = 0.0
            for robot_id in robot_ids:
                times, estimate, truth = trajectories[robot_id]
                end = int(np.searchsorted(times, current_t, side="right"))
                estimate_now = estimate[:end]
                truth_now = truth[:end]
                artists = line_artists[robot_id]
                artists[0].set_data(estimate_now[:, 0], estimate_now[:, 1])
                artists[1].set_data(truth_now[:, 0], truth_now[:, 1])
                artists[2].set_data(estimate_now[:, 0], estimate_now[:, 2])
                artists[3].set_data(truth_now[:, 0], truth_now[:, 2])
                if len(estimate_now):
                    marker_artists[robot_id][0].set_data(
                        [estimate_now[-1, 0]], [estimate_now[-1, 1]]
                    )
                    marker_artists[robot_id][1].set_data(
                        [estimate_now[-1, 0]], [estimate_now[-1, 2]]
                    )
                if len(truth_now) > 1:
                    total_distance += float(
                        np.linalg.norm(np.diff(truth_now, axis=0), axis=1).sum()
                    )

            coverage_end = int(
                np.searchsorted(coverage_event_t, current_t, side="right")
            )
            if coverage_end:
                progress_line.set_data(
                    coverage_event_t[:coverage_end], coverage_event_y[:coverage_end]
                )
                current_coverage = float(coverage_event_y[coverage_end - 1])
            else:
                progress_line.set_data([], [])
                current_coverage = 0.0
            time_cursor.set_xdata([current_t, current_t])
            stats_text.set_text(
                f"t={current_t:5.1f}/{duration:5.1f} s   "
                f"mapped voxels={voxel_end * stride:,}   "
                f"surface coverage={current_coverage:.3f}%   "
                f"swarm distance={total_distance:.1f} m"
            )
            writer.grab_frame()

    figure.savefig(thumbnail, dpi=100, facecolor=figure.get_facecolor())
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--record-prefix", required=True)
    parser.add_argument("--world", required=True)
    parser.add_argument("--output-video", required=True)
    parser.add_argument("--thumbnail")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--speedup", type=float, default=2.0)
    parser.add_argument("--coverage-resolution", type=float, default=1.0)
    args = parser.parse_args()

    prefix = Path(args.record_prefix)
    data = np.load(str(prefix) + ".npz", allow_pickle=False)
    summary_path = Path(str(prefix) + ".summary.json")
    boxes = parse_boxes(Path(args.world))
    summary, _, matched_reference, first_seen = update_analysis(
        summary_path, data, boxes, args.coverage_resolution
    )
    output_video = Path(args.output_video)
    thumbnail = (
        Path(args.thumbnail)
        if args.thumbnail
        else output_video.with_suffix(".png")
    )
    render_video(
        data,
        summary,
        boxes,
        matched_reference,
        first_seen,
        output_video,
        thumbnail,
        args.fps,
        args.speedup,
    )
    print(
        json.dumps(
            {
                "video": str(output_video),
                "thumbnail": str(thumbnail),
                "summary": str(summary_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
