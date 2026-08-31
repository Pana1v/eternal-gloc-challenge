#!/usr/bin/env python3
"""B4 baseline: naive evolutionary search over SE(2) poses. Scatter random pose guesses,
score by scan coverage, keep the best, mutate into the next generation, repeat.

Unlike B1's exhaustive FFT correlation, this only pays for poses it samples, so it scales
past B1's grid limit, but it's stochastic and can converge into a rack-level alias; the
surviving population then doubles as the challenge's 3 pose hypotheses.

Usage: run.py --scenarios <dir_root> --map <prior_map.pcd> --out <submission.txt>
"""

import argparse
import os
import sys
import time

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.icp import refine_pose, crop_map_near
from common.submission_writer import SubmissionWriter, write_submission_meta, pose_matrix_from_xy_yaw

POPULATION = 300
GENERATIONS = 40
ELITE_K = 30
IMMIGRANTS = 30          # fresh random blood each generation, guards against early convergence
SIGMA_XY_M = 2.0
SIGMA_YAW_DEG = 30.0
SIGMA_DECAY = 0.92       # anneals a coarse scatter into a fine local jitter
INLIER_DIST_M = 0.5
SCAN_SAMPLE = 500        # scan points per fitness evaluation
N_HYPOTHESES = 3
HYPOTHESIS_MIN_SEP_M = 1.0   # below this two hypotheses are the same answer twice
ICP_CROP_RADIUS_M = 5.0
SENSOR_HEIGHT_M = 1.0    # fixed rig height (docs/SENSORS.md), as in bl_bbs/bl_retrieval_gicp
DEFAULT_SEED = 0


def random_population(rng, n: int, bounds) -> np.ndarray:
    """(n, 3) array of (x, y, yaw) sampled uniformly over the map footprint."""
    (x_lo, x_hi), (y_lo, y_hi) = bounds
    return np.stack([
        rng.uniform(x_lo, x_hi, n),
        rng.uniform(y_lo, y_hi, n),
        rng.uniform(-np.pi, np.pi, n),
    ], axis=1)


def evaluate(poses: np.ndarray, scan: np.ndarray, tree: cKDTree) -> np.ndarray:
    """Fitness = fraction of scan points within INLIER_DIST_M of a map point at the candidate
    pose. Batches all candidates into one KD-tree query; a per-candidate Python loop wouldn't
    scale to a 300-member population over 40 generations."""
    x, y, yaw = poses[:, 0:1], poses[:, 1:2], poses[:, 2:3]
    c, s = np.cos(yaw), np.sin(yaw)

    sx, sy, sz = scan[:, 0], scan[:, 1], scan[:, 2]
    wx = c * sx - s * sy + x
    wy = s * sx + c * sy + y
    wz = np.broadcast_to(sz + SENSOR_HEIGHT_M, wx.shape)

    points = np.stack([wx, wy, wz], axis=-1).reshape(-1, 3)
    dist, _ = tree.query(points, k=1, workers=-1)
    return (dist.reshape(len(poses), -1) <= INLIER_DIST_M).mean(axis=1)


def mutate(elites: np.ndarray, rng, n: int, sigma_xy: float, sigma_yaw: float) -> np.ndarray:
    """n children, each a randomly chosen elite plus Gaussian jitter."""
    parents = elites[rng.integers(0, len(elites), n)]
    children = parents.copy()
    children[:, 0] += rng.normal(0.0, sigma_xy, n)
    children[:, 1] += rng.normal(0.0, sigma_xy, n)
    children[:, 2] += rng.normal(0.0, np.radians(sigma_yaw), n)
    children[:, 2] = np.arctan2(np.sin(children[:, 2]), np.cos(children[:, 2]))
    return children


def distinct_top(poses: np.ndarray, fitness: np.ndarray, n: int, min_sep: float):
    """The n best poses at least min_sep apart in xy; after the sigma anneal converges
    the elite pool, a plain top-n would submit the same answer three times and waste the hedge."""
    picked = []
    for idx in np.argsort(fitness)[::-1]:
        if any(np.hypot(*(poses[idx][:2] - poses[p][:2])) < min_sep for p in picked):
            continue
        picked.append(idx)
        if len(picked) == n:
            break
    return poses[picked], fitness[picked]


def evolve(scan: np.ndarray, tree: cKDTree, bounds, rng,
            population: int = POPULATION, generations: int = GENERATIONS):
    """Runs the search and returns (poses, fitness) for the final population."""
    poses = random_population(rng, population, bounds)
    sigma_xy, sigma_yaw = SIGMA_XY_M, SIGMA_YAW_DEG

    for _ in range(generations):
        fitness = evaluate(poses, scan, tree)
        elites = poses[np.argsort(fitness)[::-1][:ELITE_K]]

        n_children = population - ELITE_K - IMMIGRANTS
        poses = np.concatenate([
            elites,
            mutate(elites, rng, n_children, sigma_xy, sigma_yaw),
            random_population(rng, IMMIGRANTS, bounds),
        ], axis=0)

        sigma_xy *= SIGMA_DECAY
        sigma_yaw *= SIGMA_DECAY

    return poses, evaluate(poses, scan, tree)


def run_scenario(scenario_dir: str, map_points: np.ndarray, tree: cKDTree, bounds, seed: int):
    scan = np.asarray(o3d.io.read_point_cloud(os.path.join(scenario_dir, "lidar.pcd")).points)

    rng = np.random.default_rng(seed)
    sample = scan if len(scan) <= SCAN_SAMPLE else scan[rng.choice(len(scan), SCAN_SAMPLE, replace=False)]

    poses, fitness = evolve(sample, tree, bounds, rng)
    top_poses, top_fitness = distinct_top(poses, fitness, N_HYPOTHESES, HYPOTHESIS_MIN_SEP_M)

    refined = []
    for (x, y, yaw), fit in zip(top_poses, top_fitness):
        local_map = crop_map_near(map_points, x, y, ICP_CROP_RADIUS_M)
        if local_map.shape[0] > 100:
            x, y, yaw, _ = refine_pose(scan, local_map, x, y, yaw, init_z=SENSOR_HEIGHT_M)
        refined.append((x, y, yaw, float(fit)))

    return refined


def hypothesis_weights(fitness) -> np.ndarray:
    """Fitness shares, normalized to sum to 1. Unassigned weight is scored as
    a full-loss miss, so there is never a reason to hold mass back."""
    fitness = np.asarray(fitness, dtype=np.float64)
    if fitness.sum() <= 0:
        return np.full(len(fitness), 1.0 / len(fitness))
    return fitness / fitness.sum()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", required=True)
    parser.add_argument("--map", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                         help="the search is stochastic; fix this to make a run reproducible")
    args = parser.parse_args()

    map_points = np.asarray(o3d.io.read_point_cloud(args.map).points)
    tree = cKDTree(map_points)
    bounds = ((map_points[:, 0].min(), map_points[:, 0].max()),
               (map_points[:, 1].min(), map_points[:, 1].max()))
    print(f"map: {len(map_points)} points, footprint "
           f"{bounds[0][1] - bounds[0][0]:.1f}x{bounds[1][1] - bounds[1][0]:.1f}m")

    scenario_ids = sorted(d for d in os.listdir(args.scenarios)
                           if os.path.isdir(os.path.join(args.scenarios, d)))

    writer = SubmissionWriter(args.out)
    started = time.time()
    for scenario_id in scenario_ids:
        t0 = time.time()
        refined = run_scenario(os.path.join(args.scenarios, scenario_id), map_points, tree,
                                bounds, args.seed)
        weights = hypothesis_weights([f for _, _, _, f in refined])

        for k, ((x, y, yaw, fit), w) in enumerate(zip(refined, weights)):
            writer.add(scenario_id, pose_matrix_from_xy_yaw(x, y, yaw, z=SENSOR_HEIGHT_M),
                        weight=float(w), k=k)

        best_x, best_y, best_yaw, best_fit = refined[0]
        print(f"{scenario_id}: pose=({best_x:.2f}, {best_y:.2f}, yaw={np.degrees(best_yaw):.1f}deg) "
               f"fitness={best_fit:.3f} hypotheses={len(refined)} "
               f"weights={[f'{w:.2f}' for w in weights]} ({time.time() - t0:.1f}s)")

    writer.write()
    write_submission_meta(args.out + ".meta.json", "bl_ga", time.time() - started, {
        "population": POPULATION, "generations": GENERATIONS, "elite_k": ELITE_K,
        "immigrants": IMMIGRANTS, "sigma_xy_m": SIGMA_XY_M, "sigma_yaw_deg": SIGMA_YAW_DEG,
        "sigma_decay": SIGMA_DECAY, "inlier_dist_m": INLIER_DIST_M,
        "scan_sample": SCAN_SAMPLE, "seed": args.seed,
    })
    print(f"wrote {args.out} ({len(scenario_ids)} scenarios)")


if __name__ == "__main__":
    main()
