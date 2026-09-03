#!/usr/bin/env python3
"""Renders one top-down figure per scenario: the prior map, the ground-truth
pose, and each submitted pose, beside the scenario's camera image.

Written as PNGs next to the report so the whole thing browses offline from
the filesystem , no server, no network.

    python render_scenarios.py --scenarios <dir> --map <prior_map.pcd> \
        --gt <gt/A.txt> --out <results/figures> \
        --submission bl_bbs=<sub.txt> --submission bl_ga=<sub.txt>

Poses come from eval/io_formats, the same parser the scorer uses, so a
figure cannot disagree with the score beside it.
"""

import argparse
import math
import multiprocessing
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from io_formats import load_gt, load_submission
from report import load_tiers   # full csv rows, not just the tier label

# The band drawn top-down. It starts above the floor on purpose: the floor is
# a continuous surface covering every cell, so including it renders the whole
# map as solid occupied and hides the racks entirely.
RACK_BAND_M = (0.6, 6.0)
MAP_RES_M = 0.25          # map raster cell; a raster of every point reads far
                          # better than a scatter of a subsample of them
SCAN_SAMPLE = 15_000
ARROW_M = 4.0             # heading arrow length
# distinguishable without relying on hue alone being readable at a glance
METHOD_COLORS = ["#D55E00", "#0072B2", "#009E73", "#CC79A7", "#E69F00"]


def pose_xy_yaw(T):
    return float(T[0, 3]), float(T[1, 3]), math.atan2(float(T[1, 0]), float(T[0, 0]))


def draw_pose(ax, T, color, label, marker="o"):
    x, y, yaw = pose_xy_yaw(T)
    ax.plot([x], [y], marker=marker, color=color, markersize=9, markeredgecolor="black",
            markeredgewidth=0.6, linestyle="none", label=label, zorder=6)
    ax.arrow(x, y, ARROW_M * math.cos(yaw), ARROW_M * math.sin(yaw), color=color,
             width=0.35, head_width=1.6, length_includes_head=True, zorder=6)
    return x, y


def render_one(scenario_id, gt_T, submissions, map_raster, scenarios_dir,
                tier_row, out_path):
    fig = plt.figure(figsize=(15, 7.5))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.0], wspace=0.12)
    ax = fig.add_subplot(grid[0, 0])

    raster, extent = map_raster
    ax.imshow(raster.T, origin="lower", extent=extent, cmap="Greys", vmin=0, vmax=1.35,
              interpolation="nearest", zorder=1)

    scan_path = os.path.join(scenarios_dir, scenario_id, "lidar.pcd")
    if not os.path.exists(scan_path):
        scan_path = os.path.join(scenarios_dir, scenario_id, "steps", "000", "lidar.pcd")
    if os.path.exists(scan_path):
        scan = np.asarray(o3d.io.read_point_cloud(scan_path).points)
        if scan.shape[0] > SCAN_SAMPLE:
            scan = scan[np.random.default_rng(0).choice(scan.shape[0], SCAN_SAMPLE, replace=False)]
        # placed by ground truth: shows what the sensor actually saw, in map frame
        world = (gt_T[:3, :3] @ scan.T).T + gt_T[:3, 3]
        ax.scatter(world[:, 0], world[:, 1], s=0.5, c="#1F77B4", alpha=0.55, linewidths=0,
                    zorder=3, label="scan at ground truth")

    gx, gy = draw_pose(ax, gt_T, "#00A000", "ground truth", marker="*")
    for i, (method, T) in enumerate(sorted(submissions.items())):
        color = METHOD_COLORS[i % len(METHOD_COLORS)]
        ex, ey = draw_pose(ax, T, color, method)
        err = math.hypot(ex - gx, ey - gy)
        if err > 1.0:   # only draw the error line when there is a visible error
            ax.plot([gx, ex], [gy, ey], color=color, linestyle=":", linewidth=1.2, zorder=5)

    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    context = ""
    if tier_row:
        context = (f"   tier {tier_row.get('tier', '?')}, "
                   f"{tier_row.get('ambiguity_lidar_low', '?')} rack-level aliases")
    ax.set_title(f"{scenario_id}{context}", fontsize=11)

    cam = fig.add_subplot(grid[0, 1])
    cam_path = os.path.join(scenarios_dir, scenario_id, "camera.png")
    if not os.path.exists(cam_path):
        cam_path = os.path.join(scenarios_dir, scenario_id, "steps", "000", "camera.png")
    if os.path.exists(cam_path):
        cam.imshow(np.array(Image.open(cam_path)))
        cam.set_title("camera", fontsize=10)
    else:
        cam.text(0.5, 0.5, "no camera image", ha="center", va="center")
    cam.axis("off")

    fig.savefig(out_path, dpi=85, bbox_inches="tight")
    plt.close(fig)


# Scenarios render independently and matplotlib figure work is CPU-bound, so
# a sequential loop leaves every core but one idle (profiled: rendering is
# ~90% of this script's wall time on a 40-scenario fixture, ~0.18s/figure).
#
# Unlike bl_bbs's worker init, which reloads the map from disk per worker
# because the map itself is the payload, here main() has already reduced the
# 9M-point map to a small 2D raster (~1 MB) before any worker exists: that
# raster is what render_one actually draws, so it is cheaper to pickle it
# once into every worker's initargs than to re-read and re-rasterize the
# full point cloud in each one. Same hoist-out-of-the-loop idiom, applied to
# whichever payload is actually small.
#
# spawn, not fork: matplotlib is not thread/process-safe across a fork that
# inherits live state, and bl_bbs already hit a hang forking after Open3D
# had run in the parent (see baselines/bl_bbs/run.py). The map read here
# happens before the pool too.
_CTX = {}


def _init_worker(map_raster, scenarios_dir, gt, per_scenario, tiers, out_dir):
    matplotlib.use("Agg")   # belt and suspenders: spawn re-imports this module,
                             # which already sets Agg, but a worker must never
                             # fall back to an interactive backend
    _CTX.update(map_raster=map_raster, scenarios_dir=scenarios_dir, gt=gt,
                per_scenario=per_scenario, tiers=tiers, out_dir=out_dir)


def _render_one(scenario_id):
    out_path = os.path.join(_CTX["out_dir"], f"{scenario_id}.png")
    render_one(scenario_id, _CTX["gt"][scenario_id], _CTX["per_scenario"].get(scenario_id, {}),
               _CTX["map_raster"], _CTX["scenarios_dir"], _CTX["tiers"].get(scenario_id), out_path)
    return scenario_id


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", required=True)
    ap.add_argument("--map", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--out", required=True, help="directory for the PNGs")
    ap.add_argument("--track", default="A", choices=["A", "B"])
    ap.add_argument("--tiers")
    ap.add_argument("--submission", action="append", default=[],
                     metavar="NAME=PATH", help="repeatable: a method name and its submission file")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1,
                     help="parallel render workers (1 = serial, no pool)")
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    gt = load_gt(args.gt)
    tiers = load_tiers(args.tiers)

    # {scenario_id: {method: T}} , the primary hypothesis only, which is what
    # the headline score is driven by
    per_scenario = {}
    for spec in args.submission:
        name, _, path = spec.partition("=")
        if not os.path.exists(path):
            print(f"render: skipping {name}, no {path}")
            continue
        for sid, hyps in load_submission(path, args.track).items():
            primary = min(hyps, key=lambda h: h.k)
            per_scenario.setdefault(sid, {})[name] = primary.T

    points = np.asarray(o3d.io.read_point_cloud(args.map).points)
    z = points[:, 2]
    low = points[(z >= RACK_BAND_M[0]) & (z < RACK_BAND_M[1])]
    x0, x1 = float(points[:, 0].min()), float(points[:, 0].max())
    y0, y1 = float(points[:, 1].min()), float(points[:, 1].max())
    nx = max(1, int(np.ceil((x1 - x0) / MAP_RES_M)))
    ny = max(1, int(np.ceil((y1 - y0) / MAP_RES_M)))
    counts, _, _ = np.histogram2d(low[:, 0], low[:, 1], bins=(nx, ny),
                                   range=((x0, x1), (y0, y1)))
    map_raster = (np.minimum(counts, 1.0), (x0, x1, y0, y1))   # occupancy, not density

    scenario_ids = sorted(gt)
    jobs = max(1, min(args.jobs, len(scenario_ids)))
    init_args = (map_raster, args.scenarios, gt, per_scenario, tiers, args.out)

    if jobs == 1:
        _init_worker(*init_args)
        written = [_render_one(sid) for sid in scenario_ids]
    else:
        # jobs only changes how the same set of independent files gets
        # produced, never their content or the order scenarios are listed
        # in below, so the report cannot vary with worker count.
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(jobs, initializer=_init_worker, initargs=init_args) as pool:
            written = pool.map(_render_one, scenario_ids)

    print(f"wrote {len(written)} scenario figures to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
