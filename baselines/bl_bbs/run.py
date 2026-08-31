#!/usr/bin/env python3
"""B1 baseline: multi-slice 2D correlative matching (an exhaustive FFT-accelerated SE(2) search
standing in for branch-and-bound) plus point-to-plane ICP refinement.

Usage: run.py --scenarios <dir_root> --map <prior_map.pcd> --out <submission.txt>
"""

import argparse
import os
import sys
import time

import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.bev import match_scan_to_map
from common.icp import refine_pose, crop_map_near
from common.submission_writer import SubmissionWriter, write_submission_meta, pose_matrix_from_xy_yaw

RESOLUTION_M = 0.25
YAW_STEP_DEG = 3.0
ICP_CROP_RADIUS_M = 5.0
SENSOR_HEIGHT_M = 1.0  # fixed rig height (see docs/SENSORS.md); ICP needs
                       # this as initial Z to find correspondences


def build_slice_bands(z_min: float, z_max: float):
    """5 bands rescaled to the map's z-extent; top two (ceiling layer) weighted 2x per spec."""
    height = z_max - z_min
    fracs = [(0.0, 1 / 12), (1 / 12, 3.5 / 12), (3.5 / 12, 6.5 / 12), (6.5 / 12, 9.5 / 12), (9.5 / 12, 1.0)]
    bands = [(z_min + lo * height, z_min + hi * height) for lo, hi in fracs]
    weights = [1.0, 1.0, 1.0, 2.0, 2.0]
    return bands, weights


def run_scenario(scenario_dir: str, map_points: np.ndarray, length: float, width: float,
                  slice_bands, slice_weights, query_half_extent_m: float):
    lidar_path = os.path.join(scenario_dir, "lidar.pcd")
    scan = np.asarray(o3d.io.read_point_cloud(lidar_path).points)

    x, y, yaw, score, per_slice = match_scan_to_map(
        scan, map_points, length, width, slice_bands, slice_weights,
        resolution=RESOLUTION_M, yaw_step_deg=YAW_STEP_DEG, query_half_extent_m=query_half_extent_m,
    )

    local_map = crop_map_near(map_points, x, y, ICP_CROP_RADIUS_M)
    if local_map.shape[0] > 100:
        x, y, yaw, fitness = refine_pose(scan, local_map, x, y, yaw, init_z=SENSOR_HEIGHT_M)
    else:
        fitness = 0.0

    return x, y, yaw, score, fitness, per_slice


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", required=True)
    parser.add_argument("--map", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    map_pcd = o3d.io.read_point_cloud(args.map)
    map_points = np.asarray(map_pcd.points)
    x_min, x_max = map_points[:, 0].min(), map_points[:, 0].max()
    y_min, y_max = map_points[:, 1].min(), map_points[:, 1].max()
    z_min, z_max = map_points[:, 2].min(), map_points[:, 2].max()
    length, width = x_max - x_min, y_max - y_min
    # match_scan_to_map assumes map origin (0, 0); shift points so the map's min corner sits there
    map_points = map_points - np.array([x_min, y_min, 0.0])

    slice_bands, slice_weights = build_slice_bands(z_min, z_max)
    # cap the 70 m lidar range so small maps still leave room for the correlation search (see match_scan_to_map)
    query_half_extent_m = min(75.0, min(length, width) / 2.5)

    writer = SubmissionWriter(args.out)
    scenario_ids = sorted(d for d in os.listdir(args.scenarios)
                           if os.path.isdir(os.path.join(args.scenarios, d)))

    t_start = time.time()
    for scenario_id in scenario_ids:
        scenario_dir = os.path.join(args.scenarios, scenario_id)
        t0 = time.time()
        x, y, yaw, score, fitness, per_slice = run_scenario(
            scenario_dir, map_points, length, width, slice_bands, slice_weights, query_half_extent_m)
        elapsed = time.time() - t0

        # answers are in the ORIGINAL map frame (undo the shift above)
        T = pose_matrix_from_xy_yaw(x + x_min, y + y_min, yaw)
        writer.add(scenario_id, T, weight=1.0, k=0)
        print(f"{scenario_id}: pose=({x + x_min:.2f}, {y + y_min:.2f}, yaw={np.degrees(yaw):.1f}deg) "
              f"score={score:.1f} icp_fitness={fitness:.2f} slices={[f'{s:.1f}' for s in per_slice]} "
              f"({elapsed:.1f}s)")

    writer.write()
    write_submission_meta(
        args.out + ".meta.json", method_name="bl_bbs", runtime_sec_total=time.time() - t_start,
        params={"resolution_m": RESOLUTION_M, "yaw_step_deg": YAW_STEP_DEG,
                "icp_crop_radius_m": ICP_CROP_RADIUS_M, "slice_weights": slice_weights},
    )
    print(f"wrote {args.out} ({len(scenario_ids)} scenarios)")


if __name__ == "__main__":
    main()
