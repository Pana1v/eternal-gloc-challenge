#!/usr/bin/env python3
"""B3 baseline: vision tie-breaker. Re-ranks another baseline's (e.g. B1's)
top-K SE(2) hypotheses per scenario using the scenario's camera image,
cross-checked against the prior map's geometry (no textures exist, so this
is a structural edge-agreement check, not appearance matching). Cannot
localize alone by design; it only re-orders/re-weights a hypothesis set it
did not generate.

Usage: run.py --scenarios <dir_root> --map <prior_map.pcd>
              --hypotheses <submission.txt with top-K per scenario>
              --out <reranked_submission.txt>
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import open3d as o3d
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.submission_writer import SubmissionWriter  # noqa: E402

from projection import (
    project_points, hypothesis_pose_matrix, remove_occluded, K_from_camera_info, T_base_camera_from_calib,
)
from edge_score import image_edge_map, projected_edge_map, score_edge_agreement

MAP_CROP_RADIUS_M = 30.0
WEIGHT_DECIMALS = 4          # SubmissionWriter formats weights as %.4f


def parse_hypotheses(path: str):
    """Groups lines by scenario_id -> list of (k, w, T, extra_fields)."""
    groups = defaultdict(list)
    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            scenario_id, k, w = parts[0], int(parts[1]), float(parts[2])
            rest = parts[3:]
            steps_used = None
            if len(rest) == 13:
                steps_used = int(rest[0])
                rest = rest[1:]
            values = [float(v) for v in rest]
            T = np.eye(4)
            T[:3, :] = np.array(values).reshape(3, 4)
            groups[scenario_id].append({"k": k, "w": w, "T": T, "steps_used": steps_used})
    return groups


def yaw_from_matrix(T: np.ndarray) -> float:
    return float(np.arctan2(T[1, 0], T[0, 0]))


def score_hypothesis(map_points: np.ndarray, T_hyp: np.ndarray, T_base_camera: np.ndarray,
                      K: np.ndarray, width: int, height: int, image_edges: np.ndarray) -> float:
    x, y = T_hyp[0, 3], T_hyp[1, 3]
    yaw = yaw_from_matrix(T_hyp)
    T_map_base = hypothesis_pose_matrix(x, y, yaw, z=T_hyp[2, 3])

    dist2 = (map_points[:, 0] - x) ** 2 + (map_points[:, 1] - y) ** 2
    local_map = map_points[dist2 <= MAP_CROP_RADIUS_M ** 2]
    if local_map.shape[0] == 0:
        return 0.0

    u, v, z = project_points(local_map, T_map_base, T_base_camera, K, width, height)
    if len(u) == 0:
        return 0.0
    u, v, z = remove_occluded(u, v, z, width, height)

    proj_edges = projected_edge_map(u, v, width, height)
    return score_edge_agreement(proj_edges, image_edges)


def rerank_scenario(scenario_dir: str, map_points: np.ndarray, hypotheses: list):
    with open(os.path.join(scenario_dir, "camera_info.json")) as f:
        camera_info = json.load(f)
    with open(os.path.join(scenario_dir, "calib.json")) as f:
        calib = json.load(f)
    image = np.array(Image.open(os.path.join(scenario_dir, "camera.png")))

    K = K_from_camera_info(camera_info)
    T_base_camera = T_base_camera_from_calib(calib)
    width, height = camera_info["width"], camera_info["height"]
    image_edges = image_edge_map(image)

    scored = []
    for hyp in hypotheses:
        score = score_hypothesis(map_points, hyp["T"], T_base_camera, K, width, height, image_edges)
        scored.append((score, hyp))

    scored.sort(key=lambda t: t[0], reverse=True)
    scores = np.array([s for s, _ in scored])
    if scores.sum() > 0:
        weights = scores / scores.sum()
    else:
        weights = np.full(len(scores), 1.0 / len(scores))

    # the writer rounds each weight to WEIGHT_DECIMALS, and rounding a set that
    # sums to exactly 1 can total 1.0001, which the scorer rejects outright as
    # a malformed submission: settle the rounding here and hand the excess back
    weights = np.round(weights, WEIGHT_DECIMALS)
    excess = weights.sum() - 1.0
    if excess > 0:
        weights[np.argmax(weights)] -= excess
    return scored, weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", required=True)
    parser.add_argument("--map", required=True)
    parser.add_argument("--hypotheses", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    map_points = np.asarray(o3d.io.read_point_cloud(args.map).points)
    groups = parse_hypotheses(args.hypotheses)

    writer = SubmissionWriter(args.out)
    for scenario_id, hypotheses in sorted(groups.items()):
        scenario_dir = os.path.join(args.scenarios, scenario_id)
        scored, weights = rerank_scenario(scenario_dir, map_points, hypotheses)

        for new_k, ((score, hyp), w) in enumerate(zip(scored, weights)):
            writer.add(scenario_id, hyp["T"], weight=float(w), k=new_k, steps_used=hyp["steps_used"])
        print(f"{scenario_id}: {len(hypotheses)} hypotheses, edge-scores="
              f"{[f'{s:.3f}' for s, _ in scored]}, weights={[f'{w:.3f}' for w in weights]}")

    writer.write()
    print(f"wrote {args.out} ({len(groups)} scenarios)")


if __name__ == "__main__":
    main()
