#!/usr/bin/env python3
"""Accumulate a PointCloud2 topic and write a binary PCD — no LIO restart needed.

Swarm-LIO2 only dumps its map from a SIGINT handler (and only when pcd_save_en
was true at launch), which is easy to miss and kills the node.  This subscribes
to the registered-cloud topic instead, voxel-downsamples on the fly and writes a
PCD you can open in CloudCompare / MeshLab / pcl_viewer.

    python3 save_cloud_pcd.py --topic /cloud_registered --seconds 60 --out /tmp/map.pcd
    python3 save_cloud_pcd.py --topic /cloud_registered --out /tmp/map.pcd   # until Ctrl-C
"""
import argparse
import struct
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2


def cloud_to_xyz(msg):
    """Extract x,y,z as an (N,3) float32 array straight from the raw buffer."""
    off = {f.name: f.offset for f in msg.fields}
    if not {"x", "y", "z"} <= set(off):
        return np.empty((0, 3), dtype=np.float32)
    n = msg.width * msg.height
    raw = np.frombuffer(msg.data, dtype=np.uint8, count=n * msg.point_step)
    raw = raw.reshape(n, msg.point_step)
    out = np.empty((n, 3), dtype=np.float32)
    for i, key in enumerate(("x", "y", "z")):
        out[:, i] = raw[:, off[key]:off[key] + 4].copy().view(np.float32).ravel()
    return out[np.isfinite(out).all(axis=1)]


class Saver(Node):
    def __init__(self, topic, voxel):
        super().__init__("save_cloud_pcd")
        self.voxel = voxel
        self.points = []          # list of (M,3) arrays, already voxel-keyed
        self.seen = set()         # voxel keys, keeps memory bounded
        self.frames = 0
        self.create_subscription(PointCloud2, topic, self.on_cloud, qos_profile_sensor_data)
        self.get_logger().info(f"subscribing {topic} (voxel {voxel} m)")

    def on_cloud(self, msg):
        pts = cloud_to_xyz(msg)
        if len(pts) == 0:
            return
        self.frames += 1
        keys = np.floor(pts / self.voxel).astype(np.int64)
        keep = []
        for p, k in zip(pts, map(tuple, keys)):
            if k not in self.seen:
                self.seen.add(k)
                keep.append(p)
        if keep:
            self.points.append(np.asarray(keep, dtype=np.float32))
        if self.frames % 20 == 0:
            self.get_logger().info(f"{self.frames} frames, {len(self.seen)} voxels kept")

    def write(self, path):
        if not self.points:
            self.get_logger().error("no points received — is the topic publishing?")
            return False
        cloud = np.concatenate(self.points, axis=0)
        with open(path, "wb") as fh:
            fh.write(("# .PCD v0.7 - Point Cloud Data file format\n"
                      "VERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n"
                      f"WIDTH {len(cloud)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n"
                      f"POINTS {len(cloud)}\nDATA binary\n").encode())
            fh.write(cloud.astype(np.float32).tobytes())
        self.get_logger().info(f"wrote {path}: {len(cloud)} points from {self.frames} frames")
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", default="/cloud_registered")
    ap.add_argument("--out", default="/tmp/map.pcd")
    ap.add_argument("--seconds", type=float, default=0.0, help="0 = until Ctrl-C")
    ap.add_argument("--voxel", type=float, default=0.05)
    a = ap.parse_args()
    rclpy.init()
    node = Saver(a.topic, a.voxel)
    t0 = time.time()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            if a.seconds > 0 and time.time() - t0 >= a.seconds:
                break
    except KeyboardInterrupt:
        pass
    ok = node.write(a.out)
    node.destroy_node()
    try:
        rclpy.shutdown()
    except Exception:  # noqa: BLE001
        pass
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
