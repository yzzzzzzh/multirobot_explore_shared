#!/usr/bin/env python3
"""Render fixed-view quadruped exploration and LiDAR-SLAM video."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FFMpegWriter  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402



def apply_transform(points, rotation, translation):
    """Apply the recorder's row-vector SE(3) alignment without SciPy."""
    return np.asarray(points, dtype=np.float64) @ rotation.T + translation


def draw_building(axis, layout):
    sensor_z = 0.645
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
                facecolor="#536173",
                edgecolor="#aab5c5",
                linewidth=0.7,
                alpha=0.88,
                zorder=1,
            )
        )


def configure_axis(axis, title, bounds):
    xmin, xmax, ymin, ymax = bounds
    axis.set_title(title, fontsize=13, fontweight="bold", pad=9)
    axis.set_xlim(xmin, xmax)
    axis.set_ylim(ymin, ymax)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.grid(color="#364052", linewidth=0.35, alpha=0.55)
    axis.set_facecolor("#101722")


def render(
    run_path: Path,
    summary_path: Path,
    layout_path: Path,
    coverage_path: Path | None,
    output: Path,
    thumbnail: Path,
    fps: int,
    speedup: float,
    title: str | None = None,
    panel: str = "both",
    show_registered_map: bool = True,
    cuts: dict[int, list[tuple[float, float]]] | None = None,
):
    """panel: 'both' (truth + SLAM), 'truth' or 'estimate' (single view).
    cuts: {robot_id: [(t0, t1), ...]} -- samples with t0 < t < t1 are removed
    from that robot's displayed trajectory (presentation edit; the global
    timeline, the other robots and the coverage evolution are untouched, so
    the robot marker simply waits at the splice point for t1 - t0)."""
    cuts = cuts or {}
    data = np.load(run_path, allow_pickle=False)
    coverage = (
        np.load(coverage_path, allow_pickle=False)
        if coverage_path is not None
        else None
    )
    with summary_path.open("r", encoding="utf-8") as stream:
        summary = json.load(stream)
    with layout_path.open("r", encoding="utf-8") as stream:
        layout = json.load(stream)

    robot_ids = [int(value) for value in data["robot_ids"]]
    bounds = tuple(float(value) for value in layout["bounds_xy_m"])
    rotation = np.asarray(data["first_pose_rotation"], dtype=np.float64)
    translation = np.asarray(data["first_pose_translation"], dtype=np.float64)
    trajectories = {}
    duration = 0.0
    for robot_id in robot_ids:
        times = np.asarray(data[f"t_{robot_id}"], dtype=np.float64)
        truth = np.asarray(data[f"truth_{robot_id}"], dtype=np.float64)
        estimate = apply_transform(
            np.asarray(data[f"estimate_{robot_id}"], dtype=np.float64),
            rotation,
            translation,
        )
        count = min(len(times), len(truth), len(estimate))
        times, truth, estimate = times[:count], truth[:count], estimate[:count]
        for t0, t1 in cuts.get(robot_id, []):
            keep = ~((times > t0) & (times < t1))
            times, truth, estimate = times[keep], truth[keep], estimate[keep]
        trajectories[robot_id] = (times, truth, estimate)
        if len(times):
            duration = max(duration, float(times[-1]))

    voxels = np.asarray(data["voxel_xyz"], dtype=np.float64)
    voxel_first_t = np.asarray(data["voxel_first_t"], dtype=np.float64)
    voxel_order = np.argsort(voxel_first_t)
    voxels = voxels[voxel_order]
    voxel_first_t = voxel_first_t[voxel_order]
    if len(voxels) > 45000:
        stride = int(math.ceil(len(voxels) / 45000))
        voxels = voxels[::stride]
        voxel_first_t = voxel_first_t[::stride]

    if coverage is not None:
        observed_first_t = np.asarray(
            coverage["observed_first_t"], dtype=np.float64
        )
        grid_x = np.asarray(coverage["grid_x"], dtype=np.float64)
        grid_y = np.asarray(coverage["grid_y"], dtype=np.float64)
        navigable = np.asarray(coverage["navigable"], dtype=bool)
    colors = plt.cm.tab10(np.arange(len(robot_ids)) % 10)

    plt.style.use("dark_background")
    if panel == "both":
        figure, axes = plt.subplots(
            1, 2, figsize=(16, 8.6), facecolor="#0b1018",
            constrained_layout=False,
        )
        figure.subplots_adjust(
            left=0.045, right=0.98, bottom=0.08, top=0.84, wspace=0.10
        )
        axes = list(axes)
    else:
        figure, single_axis = plt.subplots(
            1, 1, figsize=(16, 9.0), facecolor="#0b1018",
            constrained_layout=False,
        )
        figure.subplots_adjust(left=0.05, right=0.985, bottom=0.075, top=0.84)
        # Both artist groups target the same axis; the unused one is hidden.
        hidden_axis = figure.add_axes([0, 0, 0.001, 0.001])
        hidden_axis.set_visible(False)
        axes = (
            [single_axis, hidden_axis] if panel == "truth"
            else [hidden_axis, single_axis]
        )
    estimate_title = (
        "Swarm-LIO2 estimate + registered LiDAR map"
        if show_registered_map
        else "Swarm-LIO2 estimate + explored area"
    )
    configure_axis(
        axes[0], "Gazebo teaching building + quadruped motion", bounds
    )
    configure_axis(axes[1], estimate_title, bounds)
    draw_building(axes[0], layout)
    draw_building(axes[1], layout)

    coverage_artist = None
    if coverage is not None:
        coverage_artist = axes[1].scatter(
            [], [], s=2.0 if panel == "both" else 3.2, color="#2e8b72",
            alpha=0.23 if panel == "both" else 0.30, linewidths=0, zorder=2,
            label="explored area" if not show_registered_map else None,
        )
    map_artist = axes[1].scatter(
        [], [], s=2.2, color="#f2c14e", alpha=0.75, linewidths=0, zorder=3,
        label="registered LiDAR",
    )
    map_artist.set_visible(show_registered_map)
    if not show_registered_map:
        map_artist.set_label("_nolegend_")
        voxels = voxels[:0]
        voxel_first_t = voxel_first_t[:0]
    artists = {}
    for index, robot_id in enumerate(robot_ids):
        color = colors[index]
        truth_line, = axes[0].plot(
            [], [], color=color, linewidth=2.3, label=f"dog{robot_id} truth",
            zorder=4,
        )
        truth_robot = axes[0].scatter(
            [], [], marker="D", s=145, color=color, edgecolor="white",
            linewidth=1.4, zorder=7,
        )
        estimate_line, = axes[1].plot(
            [], [], color=color, linewidth=2.3, label=f"dog{robot_id} SLAM",
            zorder=5,
        )
        truth_reference, = axes[1].plot(
            [], [], color=color, linewidth=0.9, linestyle="--", alpha=0.50,
            zorder=4,
        )
        estimate_robot = axes[1].scatter(
            [], [], marker="o", s=110, color=color, edgecolor="white",
            linewidth=1.2, zorder=7,
        )
        artists[robot_id] = (
            truth_line, truth_robot, estimate_line,
            truth_reference, estimate_robot,
        )

    if panel in ("both", "truth"):
        axes[0].legend(loc="lower left", fontsize=9, framealpha=0.7)
    if panel in ("both", "estimate"):
        axes[1].legend(loc="lower left", fontsize=9, framealpha=0.7)
    figure.suptitle(
        title
        or "Two-quadruped autonomous exploration — RACER + Swarm-LIO2 "
        "(LiDAR only, planar non-holonomic control)",
        fontsize=17,
        fontweight="bold",
        y=0.97,
    )
    stats_text = figure.text(
        0.045, 0.91, "", ha="left", va="top", fontsize=10.5,
        color="#dce6f4",
    )
    figure.text(
        0.98,
        0.91,
        "fixed views | left uses Gazebo truth for evaluation only"
        if panel == "both"
        else (
            "fixed view | dashed = Gazebo truth (evaluation only)"
            if panel == "estimate"
            else "fixed view | Gazebo truth (evaluation only)"
        ),
        ha="right",
        va="top",
        fontsize=9.5,
        color="#9eacc0",
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(
        fps=fps,
        codec="libx264",
        bitrate=5000,
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )
    frame_count = max(2, int(math.ceil(duration / speedup * fps)))
    pause_frames = int(round(1.0 * fps))
    last_frame = None
    with writer.saving(figure, str(output), dpi=90):
        for frame_index in range(frame_count + pause_frames):
            current_t = (
                duration
                if frame_index >= frame_count
                else min(duration, frame_index * speedup / fps)
            )
            total_distance = 0.0
            current_errors = []
            for robot_id in robot_ids:
                times, truth, estimate = trajectories[robot_id]
                end = int(np.searchsorted(times, current_t, side="right"))
                truth_now = truth[:end]
                estimate_now = estimate[:end]
                (
                    truth_line, truth_robot, estimate_line,
                    truth_reference, estimate_robot,
                ) = artists[robot_id]
                truth_line.set_data(truth_now[:, 0], truth_now[:, 1])
                estimate_line.set_data(estimate_now[:, 0], estimate_now[:, 1])
                truth_reference.set_data(truth_now[:, 0], truth_now[:, 1])
                if len(truth_now):
                    truth_robot.set_offsets(truth_now[-1, :2].reshape(1, 2))
                    total_distance += float(
                        np.linalg.norm(np.diff(truth_now[:, :2], axis=0), axis=1).sum()
                    )
                if len(estimate_now):
                    estimate_robot.set_offsets(estimate_now[-1, :2].reshape(1, 2))
                if len(truth_now) and len(estimate_now):
                    current_errors.append(
                        float(np.linalg.norm(truth_now[-1] - estimate_now[-1]))
                    )

            voxel_end = int(
                np.searchsorted(voxel_first_t, current_t, side="right")
            )
            map_artist.set_offsets(voxels[:voxel_end, :2])
            coverage_text = ""
            if coverage is not None and coverage_artist is not None:
                observed_now = (
                    navigable
                    & np.isfinite(observed_first_t)
                    & (observed_first_t <= current_t)
                )
                iy, ix = np.nonzero(observed_now)
                coverage_artist.set_offsets(
                    np.column_stack((grid_x[ix], grid_y[iy]))
                    if len(ix)
                    else np.empty((0, 2))
                )
                coverage_fraction = float(np.count_nonzero(observed_now)) / max(
                    int(np.count_nonzero(navigable)), 1
                )
                coverage_text = f"coverage {100.0 * coverage_fraction:5.1f}%   |   "
            stats_text.set_text(
                f"simulation {current_t:6.1f}/{duration:6.1f} s   |   "
                f"{coverage_text}path {total_distance:6.1f} m   |   "
                f"instantaneous position error "
                f"{(max(current_errors) if current_errors else 0.0):.2f} m"
            )
            writer.grab_frame()
            if frame_index == frame_count - 1:
                figure.canvas.draw()
                last_frame = np.asarray(figure.canvas.buffer_rgba()).copy()

    if last_frame is not None:
        thumbnail.parent.mkdir(parents=True, exist_ok=True)
        plt.imsave(thumbnail, last_frame)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--layout", type=Path, required=True)
    parser.add_argument(
        "--coverage", type=Path,
        help="optional planar-coverage result; omit for trajectory/LiDAR video only",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--thumbnail", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--speedup", type=float, default=15.0)
    parser.add_argument("--title", default=None)
    parser.add_argument(
        "--panel", choices=("both", "truth", "estimate"), default="both",
        help="render both views (default) or a single view",
    )
    parser.add_argument(
        "--hide-registered-map", action="store_true",
        help="do not draw the registered LiDAR point cloud",
    )
    parser.add_argument(
        "--cut", action="append", default=[], metavar="ROBOT:T0:T1",
        help="presentation edit: drop this robot's samples with T0 < t < T1 "
        "(repeatable); the global timeline and coverage are unchanged",
    )
    args = parser.parse_args()
    cuts: dict[int, list[tuple[float, float]]] = {}
    for spec in args.cut:
        robot, t0, t1 = spec.split(":")
        cuts.setdefault(int(robot), []).append((float(t0), float(t1)))
    render(
        args.run, args.summary, args.layout, args.coverage,
        args.output, args.thumbnail, args.fps, args.speedup, args.title,
        panel=args.panel, show_registered_map=not args.hide_registered_map,
        cuts=cuts,
    )


if __name__ == "__main__":
    main()
