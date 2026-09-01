#!/usr/bin/env python3
"""B2 baseline: rotation-invariant polar-histogram retrieval against virtual scans sampled
every ~2m from the prior map, plus circular-correlation yaw estimation and ICP refinement.
The fast/scalable contrast to B1's exhaustive search: retrieval is O(database size) per
query instead of O(yaws x map cells), at the cost of being more alias-prone at rack level.
Uses the full vertical extent in one descriptor, not per-band weights like B1, the simpler,
faster tradeoff the design calls for.

Usage: run.py --scenarios <dir_root> --map <prior_map.pcd> --out <submission.txt>
"""

import argparse
import os
import sys
import time

import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from descriptor import build_polar_histogram, ring_key, estimate_yaw_shift
from common.icp import refine_pose, crop_map_near
from common.submission_writer import SubmissionWriter, write_submission_meta, pose_matrix_from_xy_yaw

DB_SPACING_M = 2.0
MAX_RADIUS_M = 30.0
N_ANGLE_BINS = 72
N_RADIUS_BINS = 20
TOP_K = 25
ICP_CROP_RADIUS_M = 5.0
MIN_DB_POINTS = 50  # skip candidate positions with too little nearby map data
SENSOR_HEIGHT_M = 1.0  # fixed rig height per design (pose_strata.py, track_b_paths.py)


def build_database(map_points_xy: np.ndarray, length: float, width: float):
    """Candidate positions on a DB_SPACING_M grid, keeping only ones with
    enough nearby map structure to be a meaningful virtual scan.
    """
    xs = np.arange(DB_SPACING_M / 2, length, DB_SPACING_M)
    ys = np.arange(DB_SPACING_M / 2, width, DB_SPACING_M)

    positions, keys = [], []
    for x in xs:
        for y in ys:
            dist2 = (map_points_xy[:, 0] - x) ** 2 + (map_points_xy[:, 1] - y) ** 2
            local = map_points_xy[dist2 <= MAX_RADIUS_M ** 2]
            if local.shape[0] < MIN_DB_POINTS:
                continue
            ph = build_polar_histogram(local, center=(x, y), n_angle_bins=N_ANGLE_BINS,
                                        n_radius_bins=N_RADIUS_BINS, max_radius=MAX_RADIUS_M)
            positions.append((x, y))
            keys.append(ring_key(ph))
    return positions, np.array(keys)


def run_scenario(scan: np.ndarray, map_points: np.ndarray, map_points_xy: np.ndarray,
                  db_positions, db_keys):
    """scan: full 3D sensor-local points (used whole for ICP; only xy for the descriptor,
    since the descriptor deliberately ignores height, see module docstring).

    Selects the final candidate by running ICP on every top-K retrieval and keeping the
    one with the best *ICP fitness* (real geometric agreement), not the coarse descriptor's
    yaw-correlation score. An earlier version picked by yaw-correlation score alone: on real
    captured scenarios that reliably chose the wrong candidate even when the true one was
    in the top-K, since a near-tied score isn't a reliable ranking signal in a repetitive
    warehouse.
    """
    scan_xy = scan[:, :2]
    query_ph = build_polar_histogram(scan_xy, center=(0.0, 0.0), n_angle_bins=N_ANGLE_BINS,
                                      n_radius_bins=N_RADIUS_BINS, max_radius=MAX_RADIUS_M)
    q_key = ring_key(query_ph)

    dists = np.linalg.norm(db_keys - q_key[None, :], axis=1)
    top_k_idx = np.argsort(dists)[:TOP_K]

    best = None  # (fitness, x, y, yaw)
    for idx in top_k_idx:
        x, y = db_positions[idx]
        dist2 = (map_points_xy[:, 0] - x) ** 2 + (map_points_xy[:, 1] - y) ** 2
        local = map_points_xy[dist2 <= MAX_RADIUS_M ** 2]
        cand_ph = build_polar_histogram(local, center=(x, y), n_angle_bins=N_ANGLE_BINS,
                                         n_radius_bins=N_RADIUS_BINS, max_radius=MAX_RADIUS_M)
        yaw, _score = estimate_yaw_shift(query_ph, cand_ph)

        local_map = crop_map_near(map_points, x, y, ICP_CROP_RADIUS_M)
        if local_map.shape[0] < 100:
            continue
        rx, ry, ryaw, fitness = refine_pose(scan, local_map, x, y, yaw, init_z=SENSOR_HEIGHT_M)
        if best is None or fitness > best[0]:
            best = (fitness, rx, ry, ryaw)

    if best is None:
        return 0.0, 0.0, 0.0, 0.0
    fitness, x, y, yaw = best
    return x, y, yaw, fitness


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
    length, width = x_max - x_min, y_max - y_min
    map_points = map_points - np.array([x_min, y_min, 0.0])
    map_points_xy = map_points[:, :2]

    print(f"building retrieval database (spacing={DB_SPACING_M}m over {length:.1f}x{width:.1f}m map)...")
    t0 = time.time()
    db_positions, db_keys = build_database(map_points_xy, length, width)
    print(f"database: {len(db_positions)} candidates ({time.time() - t0:.1f}s)")

    writer = SubmissionWriter(args.out)
    scenario_ids = sorted(d for d in os.listdir(args.scenarios)
                           if os.path.isdir(os.path.join(args.scenarios, d)))

    t_start = time.time()
    for scenario_id in scenario_ids:
        lidar_path = os.path.join(args.scenarios, scenario_id, "lidar.pcd")
        scan = np.asarray(o3d.io.read_point_cloud(lidar_path).points)
        t0 = time.time()
        x, y, yaw, fitness = run_scenario(scan, map_points, map_points_xy, db_positions, db_keys)
        elapsed = time.time() - t0

        T = pose_matrix_from_xy_yaw(x + x_min, y + y_min, yaw, z=SENSOR_HEIGHT_M)
        writer.add(scenario_id, T, weight=1.0, k=0)
        print(f"{scenario_id}: pose=({x + x_min:.2f}, {y + y_min:.2f}, yaw={np.degrees(yaw):.1f}deg) "
              f"icp_fitness={fitness:.2f} ({elapsed:.2f}s)")

    writer.write()
    write_submission_meta(
        args.out + ".meta.json", method_name="bl_retrieval_gicp", runtime_sec_total=time.time() - t_start,
        params={"db_spacing_m": DB_SPACING_M, "max_radius_m": MAX_RADIUS_M, "top_k": TOP_K,
                "n_angle_bins": N_ANGLE_BINS, "n_radius_bins": N_RADIUS_BINS},
        n_scenarios=len(scenario_ids),
    )
    print(f"wrote {args.out} ({len(scenario_ids)} scenarios)")


if __name__ == "__main__":
    main()
