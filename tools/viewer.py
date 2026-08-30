#!/usr/bin/env python3
"""Scenario viewer , the 5-minute path's first stop.

Shows a Track A scenario's lidar scan floating in its own local frame next
to the prior map (the "kidnapped robot" view: nothing ties the scan to the
map), plus the scenario's camera image in a second window. Pass --show-gt
with a KITTI 3x4 pose line to snap the scan into the map frame instead.

Usage:
    python viewer.py <scenario_dir> --map <prior_map.pcd> [--show-gt "<12 floats>"]
"""

import argparse
import os
import sys

import numpy as np
import open3d as o3d
from PIL import Image


def parse_kitti_line(line: str) -> np.ndarray:
    """A KITTI pose line is either `<scenario_id> <12 floats>` (gt files) or
    just `<12 floats>` (what a user pastes from one) , accept both."""
    values = line.strip().split()
    if len(values) == 13:
        values = values[1:]
    if len(values) != 12:
        raise ValueError(f"expected 12 pose values (optionally prefixed by a scenario_id), got {len(values)}")
    T = np.eye(4)
    T[:3, :] = np.array([float(v) for v in values]).reshape(3, 4)
    return T


def load_scenario(scenario_dir: str):
    lidar_path = os.path.join(scenario_dir, "lidar.pcd")
    camera_path = os.path.join(scenario_dir, "camera.png")
    scan = o3d.io.read_point_cloud(lidar_path)
    image = np.array(Image.open(camera_path)) if os.path.exists(camera_path) else None
    return scan, image


def apply_gt_transform(scan: o3d.geometry.PointCloud, T: np.ndarray) -> o3d.geometry.PointCloud:
    transformed = o3d.geometry.PointCloud(scan)
    transformed.transform(T)
    return transformed


def build_scene(scan: o3d.geometry.PointCloud, map_pcd: o3d.geometry.PointCloud, snapped: bool):
    """Returns the list of geometries to visualize, colored so the scan and
    map are visually distinguishable regardless of --show-gt."""
    map_colored = o3d.geometry.PointCloud(map_pcd)
    map_colored.paint_uniform_color([0.6, 0.6, 0.6])

    scan_colored = o3d.geometry.PointCloud(scan)
    scan_colored.paint_uniform_color([0.9, 0.2, 0.2])

    if not snapped:
        # keep the scan visibly disconnected from the map when not snapped ,
        # nudge it off to one side rather than overlapping the map's own
        # (0, 0)-relative origin, which would otherwise coincidentally overlap
        scan_pts = np.asarray(scan_colored.points)
        map_pts = np.asarray(map_colored.points)
        if scan_pts.size and map_pts.size:
            offset = np.array([map_pts[:, 0].max() - map_pts[:, 0].min() + 5.0, 0.0, 0.0])
            scan_colored.points = o3d.utility.Vector3dVector(scan_pts + offset)

    return [map_colored, scan_colored]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario_dir")
    parser.add_argument("--map", required=True)
    parser.add_argument("--show-gt", help="a KITTI 3x4 pose line (12 floats, optionally prefixed by scenario_id)")
    parser.add_argument("--dry-run", action="store_true",
                         help="load and transform data, print shapes, skip opening any window")
    args = parser.parse_args()

    scan, image = load_scenario(args.scenario_dir)
    map_pcd = o3d.io.read_point_cloud(args.map)

    snapped = False
    if args.show_gt:
        T = parse_kitti_line(args.show_gt)
        scan = apply_gt_transform(scan, T)
        snapped = True

    print(f"scan points: {len(scan.points)}")
    print(f"map points: {len(map_pcd.points)}")
    print(f"camera image: {'none' if image is None else image.shape}")
    print(f"snapped to GT: {snapped}")

    if args.dry_run:
        print("dry run: skipping window display")
        return

    geometries = build_scene(scan, map_pcd, snapped)
    if image is not None:
        try:
            import matplotlib.pyplot as plt
            plt.figure("camera.png")
            plt.imshow(image)
            plt.axis("off")
            plt.show(block=False)
        except Exception as e:
            print(f"warning: could not open camera image window ({e})", file=sys.stderr)

    o3d.visualization.draw_geometries(geometries, window_name="Eternal GLoc Challenge scenario viewer")


if __name__ == "__main__":
    main()
