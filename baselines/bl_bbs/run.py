#!/usr/bin/env python3
"""B1 baseline: multi-slice 2D correlative matching (an exhaustive FFT-accelerated SE(2) search
standing in for branch-and-bound) plus point-to-plane ICP refinement.

Usage: run.py --scenarios <dir_root> --map <prior_map.pcd> --out <submission.txt>
"""

import argparse
import multiprocessing
import os
import sys
import time

import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.bev import build_slice_bands, match_scan_to_map
from common.icp import refine_pose, crop_map_near
from common.submission_writer import SubmissionWriter, write_submission_meta, pose_matrix_from_xy_yaw

RESOLUTION_M = 0.25
YAW_STEP_DEG = 3.0
ICP_CROP_RADIUS_M = 5.0
SENSOR_HEIGHT_M = 1.0  # fixed rig height (see docs/SENSORS.md); ICP needs
                       # this as initial Z to find correspondences


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


# Scenarios are independent and numpy's FFT is single-threaded, so a
# sequential run leaves every core but one idle.
#
# Workers start via "spawn" and load the map themselves rather than
# inheriting it through fork. Inheriting is tempting , copy-on-write would
# share the 9 M-point map for free , but the parent has already used Open3D
# by then, and forking a process with live thread pools gives the child locks
# whose owning threads do not exist in it. Confirmed here: every fork worker
# completed its scenario, then hung forever at pool teardown. Re-reading the
# map costs a few seconds per worker and cannot deadlock. Same shape as
# generate_dataset._init_worker in the datagen package.
_CTX = {}


def _init_worker(map_path, x_min, y_min, length, width, slice_bands, slice_weights,
                  query_half_extent_m, scenarios):
    map_points = np.asarray(o3d.io.read_point_cloud(map_path).points)
    _CTX.update(scenarios=scenarios, map_points=map_points - np.array([x_min, y_min, 0.0]),
                length=length, width=width, slice_bands=slice_bands,
                slice_weights=slice_weights, query_half_extent_m=query_half_extent_m)


def _run_one(scenario_id: str):
    t0 = time.time()
    result = run_scenario(
        os.path.join(_CTX["scenarios"], scenario_id), _CTX["map_points"],
        _CTX["length"], _CTX["width"], _CTX["slice_bands"], _CTX["slice_weights"],
        _CTX["query_half_extent_m"])
    return (scenario_id, *result, time.time() - t0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", required=True)
    parser.add_argument("--map", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=0,
                         help="parallel scenario workers (0 = all cores, 1 = sequential)")
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

    workers = args.workers if args.workers > 0 else (os.cpu_count() or 1)
    workers = max(1, min(workers, len(scenario_ids)))
    init_args = (args.map, x_min, y_min, length, width, slice_bands, slice_weights,
                 query_half_extent_m, args.scenarios)

    t_start = time.time()
    if workers == 1:
        _init_worker(*init_args)
        results = [_run_one(sid) for sid in scenario_ids]
    else:
        # map() keeps results in scenario order, so the submission is
        # identical to the sequential one regardless of worker count
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(workers, initializer=_init_worker, initargs=init_args) as pool:
            results = pool.map(_run_one, scenario_ids)

    cpu_sec_total = sum(r[-1] for r in results)
    for scenario_id, x, y, yaw, score, fitness, per_slice, elapsed in results:
        # answers are in the ORIGINAL map frame (undo the shift above)
        T = pose_matrix_from_xy_yaw(x + x_min, y + y_min, yaw, z=SENSOR_HEIGHT_M)
        writer.add(scenario_id, T, weight=1.0, k=0)
        print(f"{scenario_id}: pose=({x + x_min:.2f}, {y + y_min:.2f}, yaw={np.degrees(yaw):.1f}deg) "
              f"score={score:.1f} icp_fitness={fitness:.2f} slices={[f'{s:.1f}' for s in per_slice]} "
              f"({elapsed:.1f}s)")

    writer.write()
    write_submission_meta(
        args.out + ".meta.json", method_name="bl_bbs", runtime_sec_total=time.time() - t_start,
        params={"resolution_m": RESOLUTION_M, "yaw_step_deg": YAW_STEP_DEG,
                "icp_crop_radius_m": ICP_CROP_RADIUS_M, "slice_weights": slice_weights,
                 "workers": workers, "cpu_sec_total": cpu_sec_total},
        n_scenarios=len(scenario_ids),
    )
    print(f"wrote {args.out} ({len(scenario_ids)} scenarios)")


if __name__ == "__main__":
    main()
