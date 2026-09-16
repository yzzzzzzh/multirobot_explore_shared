#!/usr/bin/env python3
"""Render synchronized Gazebo-world and Swarm-LIO2/RACER exploration views."""

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
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

from render_exploration_video import apply_transform, parse_boxes


def box_faces(center, size):
    lower = center - 0.5 * size
    upper = center + 0.5 * size
    x0, y0, z0 = lower
    x1, y1, z1 = upper
    vertices = [
        (x0, y0, z0),
        (x1, y0, z0),
        (x1, y1, z0),
        (x0, y1, z0),
        (x0, y0, z1),
        (x1, y0, z1),
        (x1, y1, z1),
        (x0, y1, z1),
    ]
    return [
        [vertices[index] for index in face]
        for face in (
            (0, 1, 2, 3),
            (4, 5, 6, 7),
            (0, 1, 5, 4),
            (2, 3, 7, 6),
            (1, 2, 6, 5),
            (0, 3, 7, 4),
        )
    ]


def boxes_in_bounds(boxes, lower, upper):
    selected = []
    for name, center, size in boxes:
        box_lower = center - 0.5 * size
        box_upper = center + 0.5 * size
        if np.all(box_upper >= lower) and np.all(box_lower <= upper):
            selected.append((name, center, size))
    return selected


def configure_3d_axis(axis, title, lower, upper):
    axis.set_title(title, fontsize=14, pad=14, fontweight="bold")
    axis.set_xlim(lower[0], upper[0])
    axis.set_ylim(lower[1], upper[1])
    axis.set_zlim(lower[2], upper[2])
    axis.set_xlabel("Gazebo x [m]", labelpad=7)
    axis.set_ylabel("Gazebo y [m]", labelpad=7)
    axis.set_zlabel("z [m]", labelpad=6)
    axis.view_init(elev=25, azim=-58)
    axis.set_box_aspect(np.maximum(upper - lower, 1.0))
    axis.grid(True, color="#3b465a", linewidth=0.45, alpha=0.55)
    for pane in (axis.xaxis.pane, axis.yaxis.pane, axis.zaxis.pane):
        pane.set_facecolor((0.08, 0.10, 0.15, 1.0))
        pane.set_edgecolor((0.32, 0.37, 0.46, 0.65))


def follow_uavs(axis, positions, scene_lower, scene_upper, current_t, duration):
    """Keep the focused UAVs and a useful amount of nearby SDF geometry in view."""
    positions = np.asarray(positions, dtype=np.float64)
    center = positions.mean(axis=0)
    half_extent = np.maximum(
        np.ptp(positions, axis=0) * 0.5 + np.asarray([3.0, 3.0, 2.5]),
        np.asarray([5.0, 5.0, 4.0]),
    )
    view_lower = center - half_extent
    view_upper = center + half_extent
    for axis_index in range(3):
        scene_size = scene_upper[axis_index] - scene_lower[axis_index]
        view_size = min(
            view_upper[axis_index] - view_lower[axis_index], scene_size
        )
        if view_lower[axis_index] < scene_lower[axis_index]:
            view_lower[axis_index] = scene_lower[axis_index]
            view_upper[axis_index] = scene_lower[axis_index] + view_size
        if view_upper[axis_index] > scene_upper[axis_index]:
            view_upper[axis_index] = scene_upper[axis_index]
            view_lower[axis_index] = scene_upper[axis_index] - view_size

    axis.set_xlim(view_lower[0], view_upper[0])
    axis.set_ylim(view_lower[1], view_upper[1])
    axis.set_zlim(view_lower[2], view_upper[2])
    axis.set_box_aspect((1.0, 1.0, 0.82))

    camera_azimuths = (-58, 32, 122, -148)
    camera_elevations = (25, 29, 23, 28)
    camera_index = min(
        int(4.0 * current_t / max(duration, 1e-9)),
        len(camera_azimuths) - 1,
    )
    axis.view_init(
        elev=camera_elevations[camera_index],
        azim=camera_azimuths[camera_index],
    )
    return camera_index + 1


def render(data, summary, boxes, output, thumbnail, fps, speedup, left_view):
    robot_ids = [int(value) for value in summary["robot_ids"]]
    duration = float(
        summary.get("duration_sim_s", summary["duration_wall_s"])
    )
    rotation = np.asarray(data["first_pose_rotation"], dtype=np.float64)
    translation = np.asarray(data["first_pose_translation"], dtype=np.float64)
    voxels = np.asarray(data["voxel_xyz"], dtype=np.float64)
    voxel_first_t = np.asarray(data["voxel_first_t"], dtype=np.float64)
    full_voxel_first_t = np.sort(voxel_first_t.copy())

    trajectories = {}
    truth_positions = []
    for robot_id in robot_ids:
        times = np.asarray(data[f"t_{robot_id}"], dtype=np.float64)
        truth = np.asarray(data[f"truth_{robot_id}"], dtype=np.float64)
        estimate = apply_transform(
            np.asarray(data[f"estimate_{robot_id}"], dtype=np.float64),
            rotation,
            translation,
        )
        trajectories[robot_id] = (times, estimate, truth)
        truth_positions.append(truth)

    combined_truth = np.vstack(truth_positions)
    lower = np.percentile(combined_truth, 0.2, axis=0) - np.asarray([2.5, 2.5, 2.5])
    upper = np.percentile(combined_truth, 99.8, axis=0) + np.asarray([2.5, 2.5, 2.5])
    lower = np.maximum(lower, 0.0)
    upper = np.minimum(upper, 50.0)
    visible_boxes = boxes_in_bounds(boxes, lower, upper)

    order = np.argsort(voxel_first_t)
    voxels = voxels[order]
    voxel_first_t = voxel_first_t[order]
    if len(voxels) > 18000:
        stride = int(math.ceil(len(voxels) / 18000))
        voxels = voxels[::stride]
        voxel_first_t = voxel_first_t[::stride]

    colors = plt.cm.tab10(np.arange(len(robot_ids)) % 10)
    robot_colors = {
        robot_id: colors[index] for index, robot_id in enumerate(robot_ids)
    }

    plt.style.use("dark_background")
    figure = plt.figure(figsize=(16, 9), facecolor="#0d111a")
    grid = figure.add_gridspec(
        1,
        2,
        left=0.025,
        right=0.975,
        bottom=0.075,
        top=0.835,
        wspace=0.04,
    )
    world_axis = figure.add_subplot(grid[0, 0], projection="3d")
    slam_axis = figure.add_subplot(grid[0, 1], projection="3d")
    configure_3d_axis(
        world_axis,
        "Gazebo environment + UAV motion",
        lower,
        upper,
    )
    configure_3d_axis(
        slam_axis,
        "Swarm-LIO2 state estimate + registered LiDAR map",
        lower,
        upper,
    )

    obstacle_faces = []
    for _, center, size in visible_boxes:
        obstacle_faces.extend(box_faces(center, size))
    fixed_left_view = left_view == "fixed"
    world_axis.add_collection3d(
        Poly3DCollection(
            obstacle_faces,
            # The fixed overview contains hundreds of adjacent boxes.  Filled,
            # translucent faces convey their volume without turning every box
            # edge into a dense wireframe that masks the UAV trajectories.
            facecolors=(0.28, 0.38, 0.54, 0.055 if fixed_left_view else 0.045),
            edgecolors=(0.68, 0.76, 0.90, 0.07 if fixed_left_view else 0.20),
            linewidths=0.24 if fixed_left_view else 0.38,
        )
    )
    if fixed_left_view:
        world_axis.set_title(
            "Gazebo environment + UAV motion — fixed overview",
            fontsize=14,
            pad=14,
            fontweight="bold",
        )

    map_artist = slam_axis.scatter(
        [],
        [],
        [],
        c=[],
        cmap="turbo",
        vmin=lower[2],
        vmax=upper[2],
        s=2.1,
        alpha=0.78,
        linewidths=0,
        label="registered LiDAR voxels",
    )

    artists = {}
    for robot_id in robot_ids:
        color = robot_colors[robot_id]
        world_trail, = world_axis.plot(
            [], [], [], color=color, linewidth=2.5, label=f"bot{robot_id} Gazebo truth"
        )
        world_drone = world_axis.scatter(
            [], [], [], s=340, marker="X", color=color, edgecolor="white",
            linewidth=2.3, depthshade=False
        )
        estimated_line, = slam_axis.plot(
            [], [], [], color=color, linewidth=2.5, label=f"bot{robot_id} SLAM"
        )
        truth_line, = slam_axis.plot(
            [], [], [], color=color, linewidth=1.15, linestyle="--", alpha=0.72,
            label=f"bot{robot_id} truth"
        )
        estimated_drone = slam_axis.scatter(
            [], [], [], s=150, marker="o", color=color, edgecolor="white",
            linewidth=1.4, depthshade=False
        )
        artists[robot_id] = (
            world_trail,
            world_drone,
            estimated_line,
            truth_line,
            estimated_drone,
        )

    world_axis.legend(
        loc="upper left", fontsize=9, framealpha=0.68, facecolor="#101621"
    )
    slam_axis.legend(
        loc="upper left", fontsize=8.5, ncol=2, framealpha=0.68,
        facecolor="#101621"
    )

    figure.suptitle(
        f"Synchronized 3D exploration — RACER + Swarm-LIO2 — "
        f"{len(robot_ids)} UAVs",
        fontsize=19,
        fontweight="bold",
        color="#f1f5fb",
        y=0.985,
    )
    stats = figure.text(
        0.025, 0.942, "", ha="left", va="top", fontsize=11.0, color="#dbe4f1"
    )
    figure.text(
        0.975,
        0.912,
        "left: exact SDF obstacle geometry + Gazebo truth   |   "
        "right: LiDAR-only SLAM/common-frame estimate",
        ha="right",
        va="top",
        fontsize=10,
        color="#aebbd0",
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(
        fps=fps,
        codec="libx264",
        bitrate=6000,
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        metadata={"title": "Synchronized Gazebo and Swarm-LIO2 exploration"},
    )
    frame_count = max(2, int(math.ceil(duration / speedup * fps)))
    pause_frames = int(round(1.2 * fps))
    with writer.saving(figure, str(output), dpi=100):
        for frame_index in range(frame_count + pause_frames):
            current_t = (
                duration
                if frame_index >= frame_count
                else min(duration, frame_index * speedup / fps)
            )
            voxel_end = int(np.searchsorted(voxel_first_t, current_t, side="right"))
            full_voxel_end = int(
                np.searchsorted(full_voxel_first_t, current_t, side="right")
            )
            current_voxels = voxels[:voxel_end]
            if len(current_voxels):
                map_artist._offsets3d = (
                    current_voxels[:, 0],
                    current_voxels[:, 1],
                    current_voxels[:, 2],
                )
                map_artist.set_array(current_voxels[:, 2])
            else:
                empty = np.empty((0,))
                map_artist._offsets3d = (empty, empty, empty)
                map_artist.set_array(empty)

            total_distance = 0.0
            current_errors = []
            current_truth_positions = {}
            for robot_id in robot_ids:
                times, estimate, truth = trajectories[robot_id]
                end = int(np.searchsorted(times, current_t, side="right"))
                estimate_now = estimate[:end]
                truth_now = truth[:end]
                (
                    world_trail,
                    world_drone,
                    estimated_line,
                    truth_line,
                    estimated_drone,
                ) = artists[robot_id]
                world_trail.set_data_3d(
                    truth_now[:, 0], truth_now[:, 1], truth_now[:, 2]
                )
                estimated_line.set_data_3d(
                    estimate_now[:, 0], estimate_now[:, 1], estimate_now[:, 2]
                )
                truth_line.set_data_3d(
                    truth_now[:, 0], truth_now[:, 1], truth_now[:, 2]
                )
                if len(truth_now):
                    position = truth_now[-1]
                    current_truth_positions[robot_id] = position
                    world_drone._offsets3d = (
                        [position[0]], [position[1]], [position[2]]
                    )
                if len(estimate_now):
                    position = estimate_now[-1]
                    estimated_drone._offsets3d = (
                        [position[0]], [position[1]], [position[2]]
                    )
                if len(truth_now) > 1:
                    total_distance += float(
                        np.linalg.norm(np.diff(truth_now, axis=0), axis=1).sum()
                    )
                if len(estimate_now) and len(truth_now):
                    current_errors.append(
                        float(np.linalg.norm(estimate_now[-1] - truth_now[-1]))
                    )

            camera_label = "fixed"
            if current_truth_positions and not fixed_left_view:
                focus_index = min(
                    int(current_t // 60.0),
                    max(int(math.ceil(duration / 60.0)) - 1, 0),
                )
                focus_robot = robot_ids[focus_index % len(robot_ids)]
                if focus_robot not in current_truth_positions:
                    focus_robot = next(iter(current_truth_positions))
                camera_index = follow_uavs(
                    world_axis,
                    [current_truth_positions[focus_robot]],
                    lower,
                    upper,
                    current_t,
                    duration,
                )
                world_axis.set_title(
                    f"Gazebo environment + UAV motion — focus bot{focus_robot}",
                    fontsize=14,
                    pad=14,
                    fontweight="bold",
                )
                camera_label = f"{camera_index}/4"
            error_text = "/".join(f"{value:.2f}" for value in current_errors)
            error_label = "/".join(f"b{robot_id}" for robot_id in robot_ids)
            stats.set_text(
                f"t={current_t:4.1f}/{duration:4.1f} s   |   "
                f"map={full_voxel_end:,} voxels   |   distance={total_distance:5.1f} m   |   "
                f"position error {error_label}={error_text} m   |   "
                f"view {camera_label}"
            )
            writer.grab_frame()
            if frame_index % max(fps * 2, 1) == 0:
                print(
                    f"[render] {frame_index}/{frame_count + pause_frames} "
                    f"t={current_t:.1f}s",
                    flush=True,
                )

    figure.savefig(thumbnail, dpi=100, facecolor=figure.get_facecolor())
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--record-prefix", required=True)
    parser.add_argument("--world", required=True)
    parser.add_argument("--output-video", required=True)
    parser.add_argument("--thumbnail")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--speedup", type=float, default=4.0)
    parser.add_argument(
        "--left-view",
        choices=("follow", "fixed"),
        default="follow",
        help="follow one UAV or keep a fixed Gazebo overview",
    )
    args = parser.parse_args()

    prefix = Path(args.record_prefix)
    data = np.load(str(prefix) + ".npz", allow_pickle=False)
    with Path(str(prefix) + ".summary.json").open("r", encoding="utf-8") as stream:
        summary = json.load(stream)
    output = Path(args.output_video)
    thumbnail = (
        Path(args.thumbnail) if args.thumbnail else output.with_suffix(".png")
    )
    render(
        data,
        summary,
        parse_boxes(Path(args.world)),
        output,
        thumbnail,
        args.fps,
        args.speedup,
        args.left_view,
    )
    print(
        json.dumps(
            {"video": str(output), "thumbnail": str(thumbnail)},
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
