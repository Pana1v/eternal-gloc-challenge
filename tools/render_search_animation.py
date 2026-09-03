#!/usr/bin/env python3
"""Animates how bl_bbs and bl_ga search, side by side, on synthetic warehouse geometry.

The results tables in docs/BASELINES.md say which baseline wins. They do not show what
either one does. This writes one GIF that does, on geometry generated here rather than on
the dataset, so it reproduces from a clean checkout with no map and no scenarios.

Both panels run the actual searches, not a scripted mock-up:

  left   bl_ga    the real population loop, using bl_ga's own constants and genetic
                  operators, scored by the same inlier fraction against a KD-tree
  right  bl_bbs   the real FFT correlative match from baselines/common/bev.py, with
                  band edges from bl_bbs.build_slice_bands

Both search SE(2) with z pinned at the rig height, which is what the baselines do. bl_ga's
"3D" is its scoring metric, not its search space, so neither panel scatters poses through a
volume: every pose in this figure sits on the floor plane.

Usage:
    python tools/render_search_animation.py --out docs/images/search_ga_vs_slices.gif
    python tools/render_search_animation.py --verify        # numbers only, no render

Mirrors the search semantics of baselines/bl_bbs/run.py and baselines/bl_ga/run.py as of
99dcfdc. Constants and operators are imported from those files, so a parameter change
follows through, but a change to what bl_ga's fitness *means* (its z treatment, say) would
need the caption text here and the matching points in docs/BASELINES.md revisited.
--verify prints the fitness definition in use, so that drift is detectable.

If bl_ga ever gains a switch over its fitness, ask it here for the un-demeaned,
height-agnostic behaviour by name rather than taking whichever default it ships: the
figure's third contrast is a statement about that specific scoring rule, and a default is
free to move under it.
"""

import argparse
import importlib.util
import os
import time

import matplotlib
matplotlib.use("Agg")   # no display in CI or the container; must precede pyplot
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from PIL import Image
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.spatial import cKDTree

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name: str, relpath: str):
    """Imports a repo file by path. The baselines are scripts, not a package, and this tool
    must not be able to drift from their constants."""
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO_ROOT, relpath))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bl_bbs = _load("bl_bbs_run", "baselines/bl_bbs/run.py")
bl_ga = _load("bl_ga_run", "baselines/bl_ga/run.py")
bev = _load("common_bev", "baselines/common/bev.py")
# Okabe-Ito, already spelled out twice in the repo (eval/report.py, eval/render_scenarios.py)
# with no shared module. Borrowing the report's copy rather than adding a third.
VIEWER_COLORS = _load("eval_report", "eval/report.py").VIEWER_COLORS

GA_COLOR, BBS_COLOR = VIEWER_COLORS[0], VIEWER_COLORS[1]
TRUTH_COLOR = "#000000"

# --- synthetic warehouse -----------------------------------------------------------------
# Footprint and z-extent of the released dev map, near enough that the pose densities below
# are the real ones. HEIGHT_M is exactly 12.0 because build_slice_bands works in fractions
# of the map's z-extent: the band table in the docs is a 0-12 m table.
LENGTH_M, WIDTH_M, HEIGHT_M = 160.0, 93.0, 12.0
# Sample step for every surface and beam. It has to be finer than bl_bbs's BEV cell, not
# merely finer than bl_ga's 0.5 m inlier radius: sampled at 0.5 m the floor rasterizes to a
# lattice with every second cell empty, and against a demeaned map grid the true pose then
# scores negative. The released map is 9 M points, dense enough that this never arises.
SAMPLE_STEP_M = 0.2
RACK_PITCH_M = 5.0        # row-to-row; the alias period bl_ga has to resolve
RACK_ROWS_Y = np.arange(10.0, 83.0, RACK_PITCH_M)
# Rack blocks split by cross-aisles. Uninterrupted rows spanning the whole hall make the
# map very nearly translation-invariant, and then even an exhaustive search has genuine
# ties; real warehouses break their rows for traffic, and the breaks carry position.
RACK_BLOCKS_X = ((14.0, 60.0), (66.0, 112.0), (118.0, 150.0))
WALL_TOP_M = 11.7
BAY_PITCH_M = 2.5         # upright spacing, so every bay in a block looks like every other
# One full-length beam saturates its row's plan-view footprint, so bands 2 and 3 rasterize
# to bit-identical grids: BEV occupancy is a union over the band, and a second beam adds no
# cell the first did not. Real racking behaves the same way, so this is not worth "fixing".
BEAM_LEVELS_M = (1.5, 3.0, 4.5, 6.0)
RACK_TOP_M = 6.5
COLUMN_TOP_M = 10.5
COLUMN_X = np.arange(10.0, 160.0, 20.0)
COLUMN_Y = np.array([6.0, 26.5, 47.0, 67.5, 88.0])
TRUSS_Z_M = 11.5
TRUSS_PITCH_M = 8.0
ROOF_Z_M = 11.7           # the deck, not the top: rasterize_slice masks z < z_hi, so points
RIDGE_Z_M = 12.0          # at exactly z_max fall outside every band. The ridge pins the
                          # extent; the deck has to sit below it to be seen at all.
# Asymmetric roof plant, written as literals rather than sampled: this is the only geometry
# that distinguishes one bay from another in the top two bands, and it has to be identical
# on every run. (x, y, half-length, half-width) in metres.
HVAC_UNITS = ((28.0, 20.0, 3.0, 2.0), (55.0, 71.0, 4.0, 2.5), (96.0, 33.0, 2.5, 3.5),
              (121.0, 62.0, 3.5, 2.0), (140.0, 15.0, 2.0, 2.0))
HVAC_Z_M = (10.6, 11.4)
SKYLIGHTS = ((40.0, 46.5, 6.0, 1.5), (88.0, 46.5, 6.0, 1.5), (132.0, 46.5, 6.0, 1.5))
# Above-rack landmarks: a mezzanine deck and two silos, the only things besides the columns
# that live in band 4. Deliberately few and unevenly placed, which is what makes that band
# worth 2x: it is the one place where bays stop looking alike.
MEZZANINE = (12.0, 66.0, 148.0, 88.0, 8.0)        # x0, y0, x1, y1, z
SILOS = ((64.0, 8.0, 3.2), (108.0, 8.0, 3.2))     # x, y, radius
SILO_TOP_M = 9.0

# --- sensor, from docs/SENSORS.md --------------------------------------------------------
LIDAR_BEAMS = 32
LIDAR_AZIMUTHS = 1800
LIDAR_EL_DEG = (-15.0, 45.0)
LIDAR_RANGE_M = (0.5, 70.0)
LIDAR_RANGE_NOISE_M = 0.02
SCAN_SEED = 0

TRUE_POSE = (72.5, 42.5, np.radians(12.0))   # in an aisle, mid-map, off-axis heading


def _line(p0, p1, step: float) -> np.ndarray:
    """Points along a segment, spaced no coarser than step."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    n = max(2, int(np.ceil(np.linalg.norm(p1 - p0) / step)) + 1)
    return p0 + np.linspace(0.0, 1.0, n)[:, None] * (p1 - p0)


def _plane(x0, x1, y0, y1, z, step: float) -> np.ndarray:
    xs = np.arange(x0, x1 + 1e-9, step)
    ys = np.arange(y0, y1 + 1e-9, step)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    return np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, z)], axis=1)


def _vertical(p0, p1, z0: float, z1: float, step: float) -> np.ndarray:
    """A vertical wall panel between two ground points."""
    base = _line((*p0, 0.0), (*p1, 0.0), step)[:, :2]
    zs = np.arange(z0, z1 + 1e-9, step)
    return np.concatenate([np.column_stack([base, np.full(len(base), z)]) for z in zs], axis=0)


def _inside_any(points: np.ndarray, boxes) -> np.ndarray:
    hit = np.zeros(len(points), dtype=bool)
    for cx, cy, hx, hy in boxes:
        hit |= ((np.abs(points[:, 0] - cx) <= hx) & (np.abs(points[:, 1] - cy) <= hy))
    return hit


def build_warehouse():
    """The scored point cloud plus the segments the figure draws.

    Two representations on purpose. bl_ga's fitness is a 0.5 m nearest-neighbour test, so a
    cloud of a few thousand points would score the true pose as a near-miss and would hide
    the floor-and-roof domination that is the whole point of the third contrast. The figure
    meanwhile has to stay legible and small, so it draws structure as line segments and
    never scatters the cloud.
    """
    clouds, racks, structure = [], [], []

    clouds.append(_plane(0.0, LENGTH_M, 0.0, WIDTH_M, 0.0, SAMPLE_STEP_M))

    corners = [(0.0, 0.0), (LENGTH_M, 0.0), (LENGTH_M, WIDTH_M), (0.0, WIDTH_M)]
    for a, b in zip(corners, corners[1:] + corners[:1]):
        clouds.append(_vertical(a, b, 0.0, WALL_TOP_M, SAMPLE_STEP_M))
        structure.append([(*a, 0.0), (*b, 0.0)])
        structure.append([(*a, 0.0), (*a, WALL_TOP_M)])
        structure.append([(*a, WALL_TOP_M), (*b, WALL_TOP_M)])

    for y in RACK_ROWS_Y:
        for x0, x1 in RACK_BLOCKS_X:
            for z in BEAM_LEVELS_M:
                clouds.append(_line((x0, y, z), (x1, y, z), SAMPLE_STEP_M))
                racks.append([(x0, y, z), (x1, y, z)])
            for x in np.arange(x0, x1 + 1e-9, BAY_PITCH_M):
                clouds.append(_line((x, y, 0.0), (x, y, RACK_TOP_M), SAMPLE_STEP_M))
            racks.append([(x0, y, 0.0), (x0, y, RACK_TOP_M)])
            racks.append([(x1, y, 0.0), (x1, y, RACK_TOP_M)])

    for x in COLUMN_X:
        for y in COLUMN_Y:
            clouds.append(_line((x, y, 0.0), (x, y, COLUMN_TOP_M), SAMPLE_STEP_M))
            structure.append([(x, y, 0.0), (x, y, COLUMN_TOP_M)])

    for x in np.arange(4.0, LENGTH_M, TRUSS_PITCH_M):
        clouds.append(_line((x, 0.0, TRUSS_Z_M), (x, WIDTH_M, TRUSS_Z_M), SAMPLE_STEP_M))
        structure.append([(x, 0.0, TRUSS_Z_M), (x, WIDTH_M, TRUSS_Z_M)])

    mx0, my0, mx1, my1, mz = MEZZANINE
    clouds.append(_plane(mx0, mx1, my0, my1, mz, SAMPLE_STEP_M))
    for corner in ((mx0, my0), (mx0, my1), (mx1, my0), (mx1, my1)):
        structure.append([(corner[0], corner[1], 0.0), (corner[0], corner[1], mz)])
    structure.append([(mx0, my0, mz), (mx1, my0, mz)])
    structure.append([(mx0, my1, mz), (mx1, my1, mz)])

    for sx, sy, radius in SILOS:
        theta = np.linspace(0.0, 2 * np.pi, int(2 * np.pi * radius / SAMPLE_STEP_M))
        ring = np.stack([sx + radius * np.cos(theta), sy + radius * np.sin(theta)], axis=1)
        for z in np.arange(0.0, SILO_TOP_M, SAMPLE_STEP_M):
            clouds.append(np.column_stack([ring, np.full(len(ring), z)]))
        structure.append([(sx, sy, 0.0), (sx, sy, SILO_TOP_M)])

    roof = _plane(0.0, LENGTH_M, 0.0, WIDTH_M, ROOF_Z_M, SAMPLE_STEP_M)
    clouds.append(roof[~_inside_any(roof, SKYLIGHTS)])
    for x in np.arange(0.0, LENGTH_M + 1e-9, TRUSS_PITCH_M):
        clouds.append(_line((x, 0.0, RIDGE_Z_M), (x, WIDTH_M, RIDGE_Z_M), SAMPLE_STEP_M))

    for cx, cy, hx, hy in HVAC_UNITS:
        for z in HVAC_Z_M:
            clouds.append(_plane(cx - hx, cx + hx, cy - hy, cy + hy, z, SAMPLE_STEP_M))
        structure.append([(cx, cy, HVAC_Z_M[0]), (cx, cy, HVAC_Z_M[1])])

    points = np.concatenate(clouds, axis=0)
    return points, {"racks": racks, "structure": structure}


def simulate_scan(map_points: np.ndarray, x: float, y: float, yaw: float) -> np.ndarray:
    """The scan a spec-conformant lidar at (x, y, rig height) would return, in sensor-local
    coordinates with z measured from the sensor, which is the frame bl_ga.evaluate expects.

    Occlusion is a z-buffer over the sensor's own beam grid: bin every map point by azimuth
    and elevation and keep the nearest one per beam. Without it the sensor sees the whole
    floor and every rack row through every other row, which makes the map far less
    ambiguous than it is and quietly deletes the aliasing the figure is about.
    """
    rel = map_points - np.array([x, y, bl_ga.SENSOR_HEIGHT_M])
    r = np.linalg.norm(rel, axis=1)
    el = np.degrees(np.arcsin(np.clip(rel[:, 2] / np.maximum(r, 1e-9), -1.0, 1.0)))
    keep = ((r >= LIDAR_RANGE_M[0]) & (r <= LIDAR_RANGE_M[1])
            & (el >= LIDAR_EL_DEG[0]) & (el <= LIDAR_EL_DEG[1]))
    rel, r, el = rel[keep], r[keep], el[keep]

    az = np.arctan2(rel[:, 1], rel[:, 0])
    i_az = np.minimum((((az + np.pi) / (2 * np.pi)) * LIDAR_AZIMUTHS).astype(np.int64),
                      LIDAR_AZIMUTHS - 1)
    el_span = LIDAR_EL_DEG[1] - LIDAR_EL_DEG[0]
    i_el = np.minimum((((el - LIDAR_EL_DEG[0]) / el_span) * LIDAR_BEAMS).astype(np.int64),
                      LIDAR_BEAMS - 1)

    beam = i_az * LIDAR_BEAMS + i_el
    order = np.lexsort((r, beam))          # nearest return first within each beam
    _, first = np.unique(beam[order], return_index=True)
    keep_idx = order[first]

    # Range noise per docs/SENSORS.md, so the scan is not a verbatim subset of the map and a
    # fitness of exactly 1.0 at the true pose cannot be an artefact of construction.
    rng = np.random.default_rng(SCAN_SEED)
    scale = 1.0 + rng.normal(0.0, LIDAR_RANGE_NOISE_M, len(keep_idx)) / r[keep_idx]
    hits = rel[keep_idx] * scale[:, None]

    c, s = np.cos(-yaw), np.sin(-yaw)      # world -> sensor frame
    return np.stack([c * hits[:, 0] - s * hits[:, 1],
                     s * hits[:, 0] + c * hits[:, 1],
                     hits[:, 2]], axis=1)


# --- the two searches --------------------------------------------------------------------

def run_ga(scan: np.ndarray, tree: cKDTree, bounds, seed: int):
    """bl_ga's population loop, unrolled one generation at a time so each can be drawn.

    bl_ga.evolve returns only the final population, so the loop is restated here; every
    constant and every genetic operator is imported, and the fitness is bl_ga.evaluate
    itself, so the trajectory is the baseline's and not an imitation of it.
    """
    rng = np.random.default_rng(seed)
    sample = (scan if len(scan) <= bl_ga.SCAN_SAMPLE
              else scan[rng.choice(len(scan), bl_ga.SCAN_SAMPLE, replace=False)])

    poses = bl_ga.random_population(rng, bl_ga.POPULATION, bounds)
    sigma_xy, sigma_yaw = bl_ga.SIGMA_XY_M, bl_ga.SIGMA_YAW_DEG
    history = []

    for _ in range(bl_ga.GENERATIONS):
        fitness = bl_ga.evaluate(poses, sample, tree)
        history.append({"poses": poses, "fitness": fitness,
                        "sigma_xy": sigma_xy, "sigma_yaw": sigma_yaw})

        elites = poses[np.argsort(fitness)[::-1][:bl_ga.ELITE_K]]
        n_children = bl_ga.POPULATION - bl_ga.ELITE_K - bl_ga.IMMIGRANTS
        poses = np.concatenate([
            elites,
            bl_ga.mutate(elites, rng, n_children, sigma_xy, sigma_yaw),
            bl_ga.random_population(rng, bl_ga.IMMIGRANTS, bounds),
        ], axis=0)
        sigma_xy *= bl_ga.SIGMA_DECAY
        sigma_yaw *= bl_ga.SIGMA_DECAY

    fitness = bl_ga.evaluate(poses, sample, tree)
    history.append({"poses": poses, "fitness": fitness,
                    "sigma_xy": sigma_xy, "sigma_yaw": sigma_yaw})

    top_poses, top_fitness = bl_ga.distinct_top(poses, fitness, bl_ga.N_HYPOTHESES,
                                                bl_ga.HYPOTHESIS_MIN_SEP_M)
    kept, _ = bl_ga.confident_subset(top_poses, top_fitness, bl_ga.HYPOTHESIS_KEEP_RATIO)
    truth = np.array([list(TRUE_POSE)])
    return {"history": history, "hypotheses": top_poses, "fitness": top_fitness,
            "n_submitted": len(kept), "sample": sample,
            "truth_fitness": float(bl_ga.evaluate(truth, sample, tree)[0])}


def run_bbs(scan: np.ndarray, map_points: np.ndarray):
    """bl_bbs's exhaustive correlative match, keeping the per-band score surfaces.

    match_scan_to_map returns only the winning placement's per-band scalars, so the winning
    heading is taken from it and the band surfaces are then recomputed at that heading with
    the same bev primitives. The search over headings is the full one; only the surfaces the
    inset draws are single-heading.
    """
    z_min, z_max = float(map_points[:, 2].min()), float(map_points[:, 2].max())
    bands, weights = bl_bbs.build_slice_bands(z_min, z_max)
    half = min(75.0, min(LENGTH_M, WIDTH_M) / 2.5)

    x, y, yaw, score, _ = bev.match_scan_to_map(
        scan, map_points, LENGTH_M, WIDTH_M, bands, weights,
        resolution=bl_bbs.RESOLUTION_M, yaw_step_deg=bl_bbs.YAW_STEP_DEG,
        query_half_extent_m=half)

    map_grids = [bev.rasterize_slice(map_points, 0.0, 0.0, LENGTH_M, WIDTH_M,
                                     bl_bbs.RESOLUTION_M, lo, hi).grid for lo, hi in bands]
    nx, ny = map_grids[0].shape
    ref_px = int(round(half / bl_bbs.RESOLUTION_M))
    qn = max(1, int(np.ceil(2 * half / bl_bbs.RESOLUTION_M)))
    rotated = bev.rotate_points_2d(scan, yaw)

    surfaces, cumulative = [], np.zeros((nx, ny), dtype=np.float32)
    for (lo, hi), grid, w in zip(bands, map_grids, weights):
        query = bev.rasterize_slice(rotated, -half, -half, 2 * half, 2 * half,
                                    bl_bbs.RESOLUTION_M, lo, hi).grid
        map_f, map_shape = bev.precompute_map_fft(grid, qn, qn)
        full, (qnx, qny) = bev.correlate_translation_full_precomputed(query, map_f, map_shape)
        cumulative = cumulative + w * bev.reference_point_scores(full, qnx, qny, ref_px,
                                                                 ref_px, nx, ny)
        surfaces.append(cumulative.copy())

    n_yaws = len(np.arange(0.0, 2 * np.pi, np.radians(bl_bbs.YAW_STEP_DEG)))
    return {"pose": (x, y, yaw), "score": score, "bands": bands, "weights": weights,
            "surfaces": surfaces, "grid_shape": (nx, ny), "n_yaws": n_yaws,
            "placements": nx * ny * n_yaws, "coverage": scan_coverage(scan, bands),
            "occupancy": [float(g.mean()) for g in map_grids]}


def scan_coverage(scan: np.ndarray, bands):
    """Where on the floor plan each band's scan evidence actually comes from.

    Sliced on the scan's own z against the band edges, which is what match_scan_to_map
    does, so the footprint drawn is the one that band's correlation consumed. Answers a
    question the height axis cannot: a band can be tall and still be told almost nothing,
    because the sensor's vertical limits and the racking decide what reaches it.
    """
    tx, ty, tyaw = TRUE_POSE
    c, s = np.cos(tyaw), np.sin(tyaw)
    world = np.stack([c * scan[:, 0] - s * scan[:, 1] + tx,
                      s * scan[:, 0] + c * scan[:, 1] + ty], axis=1)
    out = []
    for z_lo, z_hi in bands:
        xy = world[(scan[:, 2] >= z_lo) & (scan[:, 2] < z_hi)]
        touched = len(set(map(tuple, np.floor(xy).astype(np.int64)))) if len(xy) else 0
        out.append({"xy": xy, "returns": len(xy), "cells": touched,
                    "pct": 100.0 * touched / (LENGTH_M * WIDTH_M)})
    return out


# --- rendering ---------------------------------------------------------------------------
# Measured sec/scenario from the results table in docs/BASELINES.md. The animation runs on
# one clock scaled to these, so bl_bbs finishes and holds while bl_ga is still working.
BBS_SEC, GA_SEC = 2.65, 10.27
FRAMES, FPS = 60, 10
HOLD_FRAMES = 10         # pause on the final state, which is also the loop point
DPI = 90                  # the GIF is committed, so pixels are a size decision
Z_EXAGGERATION = 4.0      # a 12 m ceiling over a 160 m hall is otherwise an invisible sliver
VIEW = (32.0, -62.0)      # fixed: a rotating camera changes every pixel of every frame,
                          # which is the difference between a 2 MB GIF and a 6 MB one
# The "still in contention" overlay on bl_bbs's top-down panel. docs/BASELINES.md uses the
# same 0.9-of-the-best convention for its alias counter, so this is measured the way the
# tiers are.
CONTENTION = 0.9
# Contenders farther than this from the winning cell are a different aliased placement, not
# the same near-tie as the winner; drawn dimmed so the aliasing is visible without competing
# with the answer. Half the rack pitch: the winner's own cluster sits within a metre of it,
# the aliased clusters sit at whole rack-pitch multiples or more.
CONTENTION_NEAR_M = RACK_PITCH_M / 2
# Same idea on the bl_ga side: elites this far from the population's current best are a
# distinct, weaker mode rather than the cluster converging on the answer, and get drawn as
# plain population dots instead of the elite highlight. Larger than the converged cluster's
# own spread (a metre or so late in the run), far smaller than the gap to another mode
# (tens of metres, since bl_ga's raw fitness is nearly indifferent to a lot of the hall).
ELITE_CLUSTER_RADIUS_M = 5.0
# Above this plan-view occupancy a band is effectively a continuous surface, and demeaning
# cancels it. Not a synthetic artefact: eval/map_svg.py:16-19 says the released map's floor
# and ceiling "are continuous surfaces covering every cell", from bounds read off that map's
# own height histogram.
SOLID_OCCUPANCY = 0.9
# Racking grey, structure pale red, as docs/BASELINES.md:109-111 describes the report's
# scenario map. Pale here rather than saturated: the shell is context, not the subject.
SHELL_COLOR = "#d2d2d2"
BAND_DONE_COLOR = "#a9cfe8"   # bands already folded in, kept behind the one sweeping now
SHELL_STRUCTURE_COLOR = "#e6d2d2"
# The sensor's own limit, so neither method's colour: it constrains both of them equally.
RANGE_COLOR = "#7f7f7f"


def _clip(segments, z_lo: float, z_hi: float):
    """The parts of each segment inside a height band, so a band lights up the geometry that
    actually lives in it rather than a floating slab."""
    out = []
    for (x0, y0, z0), (x1, y1, z1) in segments:
        if max(z0, z1) < z_lo or min(z0, z1) >= z_hi:
            continue
        span = z1 - z0
        if abs(span) < 1e-9:
            out.append([(x0, y0, z0), (x1, y1, z1)])
            continue
        lo_t, hi_t = sorted(((z_lo - z0) / span, (z_hi - z0) / span))
        t0, t1 = max(0.0, lo_t), min(1.0, hi_t)
        if t1 <= t0:
            continue
        out.append([(x0 + t0 * (x1 - x0), y0 + t0 * (y1 - y0), z0 + t0 * span),
                    (x0 + t1 * (x1 - x0), y0 + t1 * (y1 - y0), z0 + t1 * span)])
    return out


def _setup(ax, top: bool = False):
    """Shared axes setup for both viewpoints. `top` swaps in a near-orthographic camera
    looking straight down the z-axis, so plan-view distances read true instead of foreshortened
    by perspective; the perspective camera stays fixed per VIEW for the height view."""
    ax.set_xlim(0, LENGTH_M)
    ax.set_ylim(0, WIDTH_M)
    ax.set_zlim(0, HEIGHT_M)
    ax.set_box_aspect((LENGTH_M, WIDTH_M, HEIGHT_M * Z_EXAGGERATION))
    if top:
        ax.set_proj_type("ortho")
        ax.view_init(90, -90)
    else:
        ax.view_init(*VIEW)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_alpha(0.0)
        axis.line.set_visible(False)   # a bare axis line with no ticks reads as an artefact
    ax.grid(False)


def _shell(ax, render, done=(), active=()):
    """The warehouse, drawn once per frame. Bands already scored stay in a pale blue and the
    band being scored now is drawn over them in full colour, so the eye follows one sweep
    upward instead of watching the whole hall turn blue and stay there."""
    ax.add_collection3d(Line3DCollection(render["racks"], colors=SHELL_COLOR, linewidths=0.4,
                                          antialiased=False))
    ax.add_collection3d(Line3DCollection(render["structure"], colors=SHELL_STRUCTURE_COLOR,
                                          linewidths=0.4, antialiased=False))
    if len(done):
        ax.add_collection3d(Line3DCollection(done, colors=BAND_DONE_COLOR, linewidths=0.6,
                                              antialiased=False))
    if len(active):
        ax.add_collection3d(Line3DCollection(active, colors=BBS_COLOR, linewidths=0.9,
                                              antialiased=False))


def _pose(ax, x, y, yaw, color, size, marker="*", filled=True):
    z = bl_ga.SENSOR_HEIGHT_M
    ax.plot([x], [y], [z], marker=marker, color=color, markersize=size,
            markerfacecolor=color if filled else "none",
            markeredgecolor="black" if filled else color,
            markeredgewidth=0.4 if filled else 1.1, linestyle="none", zorder=12)
    ax.plot([x, x + 10.0 * np.cos(yaw)], [y, y + 10.0 * np.sin(yaw)], [z, z],
            color=color, linewidth=1.0, zorder=12)


def _range_ring(x: float, y: float):
    """The sensor's maximum range as segments, clipped to the hall.

    Drawn from the true pose because that is where the scan was taken: nothing outside this
    circle is in the scan at all, so the hall's far end is evidence neither search can use.
    Clipped rather than drawn whole, since mplot3d does not clip to the axes limits and the
    ring is wider than the hall in y.
    """
    theta = np.linspace(0.0, 2 * np.pi, 361)
    rx = x + LIDAR_RANGE_M[1] * np.cos(theta)
    ry = y + LIDAR_RANGE_M[1] * np.sin(theta)
    inside = (rx >= 0.0) & (rx <= LENGTH_M) & (ry >= 0.0) & (ry <= WIDTH_M)
    z = bl_ga.SENSOR_HEIGHT_M
    return [[(rx[i], ry[i], z), (rx[i + 1], ry[i + 1], z)]
            for i in range(len(theta) - 1) if inside[i] and inside[i + 1]]


def render_gif(out_path: str, ga, bbs, render, frames: int = FRAMES, fps: int = FPS,
                hold: int = HOLD_FRAMES):
    """One GIF, one clock, four axes: one method per column, each shown top-down above and in
    perspective below. Every panel is rebuilt per frame rather than tracked as artists:
    mplot3d has no usable blitting, and at 60 frames the redraw is not the cost. Text is the
    title, the two method names and the shared clock; everything else is geometry, motion
    and colour, so each column is named once and read by colour after that."""
    # mplot3d draws into a square viewport inside its axes, so a wide, short cell renders a
    # small plan view with dead space either side and raising `zoom` clips rather than fills.
    # Same fix the two-panel version used: give each axes far more height than its row needs
    # and let the empty part of the square overflow into the neighbouring row, which is
    # transparent. The panels overlap; their ink does not.
    fig = plt.figure(figsize=(10.0, 8.0))
    ax_ga_top = fig.add_axes((0.005, 0.444, 0.49, 0.6125), projection="3d")
    ax_bbs_top = fig.add_axes((0.505, 0.444, 0.49, 0.6125), projection="3d")
    ax_ga_persp = fig.add_axes((0.005, -0.010, 0.49, 0.600), projection="3d")
    ax_bbs_persp = fig.add_axes((0.505, -0.010, 0.49, 0.600), projection="3d")

    fig.text(0.5, 0.997, "Global Localization", ha="center", va="top", fontsize=16,
             color="#222222", fontweight="bold")
    fig.text(0.25, 0.955, "Genetic Evolution", ha="center", va="top", fontsize=11,
             color=GA_COLOR)
    fig.text(0.75, 0.955, "Fast Fourier Transform", ha="center", va="top", fontsize=11,
             color=BBS_COLOR)
    # One clock for both columns, scaled to the measured sec/scenario, so the frame where
    # bl_bbs stops and bl_ga keeps going is legible as a time rather than inferred.
    clock = fig.text(0.5, 0.008, "", ha="center", va="bottom", fontsize=13, color="#444444")

    history = ga["history"]
    tx, ty, tyaw = TRUE_POSE
    bands, weights = bbs["bands"], bbs["weights"]
    surfaces = bbs["surfaces"]

    per_band = [_clip(render["racks"], lo, hi) + _clip(render["structure"], lo, hi)
                for lo, hi in bands]
    below_band = [sum(per_band[:k], []) for k in range(len(bands))]
    range_ring = _range_ring(tx, ty)

    def draw(frame):
        verdict = frame >= frames
        t = min(frame, frames - 1) / (frames - 1)
        elapsed = t * GA_SEC
        gen = min(int(round(t * (len(history) - 1))), len(history) - 1)
        band = max(1, min(int(np.ceil(min(1.0, elapsed / BBS_SEC) * len(bands))), len(bands)))
        if elapsed >= BBS_SEC:
            clock.set_text(f"{elapsed:.2f} s   (bl_bbs done, +{elapsed - BBS_SEC:.2f} s ago)")
        else:
            clock.set_text(f"{elapsed:.2f} s")

        for ax in (ax_ga_top, ax_ga_persp, ax_bbs_top, ax_bbs_persp):
            ax.clear()

        # --- bl_ga: the population, drawn top-down and in perspective ------------------
        state = history[gen]
        poses, fitness = state["poses"], state["fitness"]
        elites = np.argsort(fitness)[::-1][:bl_ga.ELITE_K]
        floor = np.full(len(poses), bl_ga.SENSOR_HEIGHT_M)
        best = poses[fitness.argmax()]
        # Elites near-tied with a distant pose are real (bl_ga's raw fitness barely separates
        # some wrong poses from the true one), but highlighting all of them makes a transient
        # secondary mode look like a rendering glitch next to the one the run submits. Only
        # the elites clustered with the current best get the elite treatment; a stray one a
        # rack row or more away just renders as an ordinary population dot.
        near_best = elites[np.hypot(poses[elites, 0] - best[0],
                                    poses[elites, 1] - best[1]) <= ELITE_CLUSTER_RADIUS_M]

        for ax, top in ((ax_ga_top, True), (ax_ga_persp, False)):
            _setup(ax, top=top)
            _shell(ax, render)
            ax.add_collection3d(Line3DCollection(range_ring, colors=RANGE_COLOR,
                                                 linewidths=0.8, linestyles=(0, (4, 3))))
            ax.scatter(poses[:, 0], poses[:, 1], floor, s=5.0, c=GA_COLOR,
                       depthshade=False, linewidths=0)
            ax.scatter(poses[near_best, 0], poses[near_best, 1], floor[near_best], s=16.0,
                       c=GA_COLOR, depthshade=False, edgecolors="black", linewidths=0.35)
            # an open ring on the population's own best, so convergence is visible without
            # reading a fitness number off the frame
            _pose(ax, best[0], best[1], best[2], GA_COLOR, 10, marker="o", filled=False)
            _pose(ax, tx, ty, tyaw, TRUTH_COLOR, 10)
            if verdict:
                for hx, hy, hyaw in ga["hypotheses"]:
                    _pose(ax, hx, hy, hyaw, GA_COLOR, 11, marker="o", filled=False)

        # --- bl_bbs: the band sweep, drawn top-down and in perspective ------------------
        surface = surfaces[band - 1]
        peak = float(surface.max())
        winner = np.unravel_index(surface.argmax(), surface.shape) if peak > 0.0 else None
        contenders = np.argwhere(surface >= CONTENTION * peak) if peak > 0.0 else None

        # The current band's actual scan returns, reprojected to world coordinates and
        # flattened to the floor plane: not a candidate placement like the contention dots
        # below, but the raw evidence bl_bbs's correlation for this band is scored against.
        # Plan view only, and drawn first so the search-state markers layer on top of it.
        cover = bbs["coverage"][band - 1]
        cover_xy = cover["xy"]

        for ax, top in ((ax_bbs_top, True), (ax_bbs_persp, False)):
            _setup(ax, top=top)
            _shell(ax, render, done=below_band[band - 1], active=per_band[band - 1])
            ax.add_collection3d(Line3DCollection(range_ring, colors=RANGE_COLOR,
                                                 linewidths=0.8, linestyles=(0, (4, 3))))
            if top and len(cover_xy):
                ax.scatter(cover_xy[:, 0], cover_xy[:, 1],
                          np.full(len(cover_xy), bl_ga.SENSOR_HEIGHT_M), s=1.2, c=BBS_COLOR,
                          depthshade=False, alpha=0.25, linewidths=0)
            if winner is not None:
                # an open ring, so it reads as agreement when it lands on the truth's star
                _pose(ax, winner[0] * bl_bbs.RESOLUTION_M, winner[1] * bl_bbs.RESOLUTION_M,
                      bbs["pose"][2], BBS_COLOR, 9, marker="o", filled=False)
            _pose(ax, tx, ty, tyaw, TRUTH_COLOR, 10)

        # placements still within CONTENTION of the best, in plan view only: this is what
        # collapses from "most of the band" to "one cell" as the sweep narrows, and it is
        # exactly where the aliasing bays line up. A fully occupied band demeans to zero
        # score everywhere, so band 1 draws nothing here rather than a stand-in for that.
        if contenders is not None:
            cx = contenders[:, 0] * bl_bbs.RESOLUTION_M
            cy = contenders[:, 1] * bl_bbs.RESOLUTION_M
            near = np.hypot(cx - winner[0] * bl_bbs.RESOLUTION_M,
                            cy - winner[1] * bl_bbs.RESOLUTION_M) <= CONTENTION_NEAR_M
            z = np.full(len(cx), bl_ga.SENSOR_HEIGHT_M)
            if np.any(~near):
                # aliased elsewhere in the hall: dimmed and unedged, so it reads as background
                # texture rather than competing with the answer for attention
                ax_bbs_top.scatter(cx[~near], cy[~near], z[~near], s=4.0, c=BBS_COLOR,
                                   depthshade=False, alpha=0.35, linewidths=0)
            # the winner's own near-ties: black-edged rather than flat BBS_COLOR, since the
            # active band's own geometry is drawn in that same blue and a flat-blue dot on a
            # flat-blue beam line is invisible
            ax_bbs_top.scatter(cx[near], cy[near], z[near], s=7.0, c=BBS_COLOR,
                               depthshade=False, edgecolors="black", linewidths=0.3)

        if verdict:
            return

    animation = FuncAnimation(fig, draw, frames=frames + hold, interval=1000 // fps)
    animation.save(out_path, writer=PillowWriter(fps=fps), dpi=DPI)
    plt.close(fig)


# --- verification ------------------------------------------------------------------------
# The band table as docs/BASELINES.md describes it, for a 0-12 m map. build_slice_bands
# works in fractions of the map's z-extent, so this only reproduces when the synthetic
# warehouse is exactly 12 m tall. Compared numerically rather than by eye.
DOC_BANDS = ((0.0, 1.0), (1.0, 3.5), (3.5, 6.5), (6.5, 9.5), (9.5, 12.0))
DOC_WEIGHTS = (1.0, 1.0, 1.0, 2.0, 2.0)


def verify(points, ga, bbs) -> bool:
    ok = True
    z_min, z_max = float(points[:, 2].min()), float(points[:, 2].max())
    bands, weights = bl_bbs.build_slice_bands(z_min, z_max)

    print(f"\n1. band edges, build_slice_bands({z_min}, {z_max}) vs the documented table")
    for k, ((lo, hi), w, (dlo, dhi), dw) in enumerate(zip(bands, weights, DOC_BANDS,
                                                          DOC_WEIGHTS), 1):
        match = (abs(lo - dlo) < 1e-9 and abs(hi - dhi) < 1e-9 and w == dw)
        ok &= match
        n = int(((points[:, 2] >= lo) & (points[:, 2] < hi)).sum())
        print(f"   band {k}  computed {lo:6.3f}-{hi:6.3f} w={w:.0f}   "
              f"documented {dlo:6.3f}-{dhi:6.3f} w={dw:.0f}   "
              f"{'match' if match else 'DIFFERS'}   {n:7d} map points")

    print("\n2. pose coverage")
    n_evals = (bl_ga.GENERATIONS + 1) * bl_ga.POPULATION
    print(f"   bl_bbs  {bbs['placements']:,} placements "
          f"({bbs['grid_shape'][0]}x{bbs['grid_shape'][1]} cells at "
          f"{bl_bbs.RESOLUTION_M} m x {bbs['n_yaws']} yaws at {bl_bbs.YAW_STEP_DEG} deg)")
    print(f"   bl_ga   {n_evals:,} evaluations "
          f"({bl_ga.GENERATIONS} generations x {bl_ga.POPULATION} + {bl_ga.POPULATION})"
          f"   ratio {bbs['placements'] / n_evals:,.0f}x")

    print("\n3. initial scatter against the alias period, to scale")
    area = LENGTH_M * WIDTH_M
    grid_equiv = np.sqrt(area / bl_ga.POPULATION)
    first = ga["history"][0]["poses"][:, :2]
    nn = cKDTree(first).query(first, k=2)[0][:, 1]
    print(f"   {bl_ga.POPULATION} poses over {LENGTH_M:.0f}x{WIDTH_M:.0f} m "
          f"= {area / bl_ga.POPULATION:.1f} m2/pose")
    print(f"   grid-equivalent spacing sqrt(area/n) = {grid_equiv:.2f} m   "
          f"vs rack pitch {RACK_PITCH_M:.1f} m   "
          f"{'coarser (cannot resolve the alias period)' if grid_equiv > RACK_PITCH_M else 'finer'}")
    print(f"   mean nearest-neighbour distance of the actual draw = {nn.mean():.2f} m; a "
          f"uniform draw clumps, so this is well under the grid-equivalent figure")
    ok &= grid_equiv > RACK_PITCH_M

    print("\n4. what each search returned")
    tx, ty, tyaw = TRUE_POSE
    bx, by, byaw = bbs["pose"]
    print(f"   truth              ({tx:7.2f}, {ty:6.2f}, {np.degrees(tyaw):6.2f} deg)")
    print(f"   bl_bbs             ({bx:7.2f}, {by:6.2f}, {np.degrees(byaw):6.2f} deg)   "
          f"err {np.hypot(bx - tx, by - ty):5.2f} m, "
          f"{abs(np.degrees(byaw - tyaw)):.2f} deg   "
          f"(one BEV cell is {bl_bbs.RESOLUTION_M} m; bl_bbs then runs ICP, not animated)")
    final = ga["history"][-1]
    gx, gy, gyaw = final["poses"][final["fitness"].argmax()]
    print(f"   bl_ga              ({gx:7.2f}, {gy:6.2f}, {np.degrees(gyaw):6.2f} deg)   "
          f"err {np.hypot(gx - tx, gy - ty):5.2f} m, "
          f"{abs(np.degrees(gyaw - tyaw)):.2f} deg")
    print(f"   bl_ga hypotheses   {len(ga['hypotheses'])} distinct modes, of which "
          f"confident_subset keeps {ga['n_submitted']} at "
          f"{bl_ga.HYPOTHESIS_KEEP_RATIO} of the best fitness")

    print("\n5. the evidence signal each search uses")
    band1 = bev.rasterize_slice(points, 0.0, 0.0, LENGTH_M, WIDTH_M, bl_bbs.RESOLUTION_M,
                                *bands[0]).grid
    band5 = bev.rasterize_slice(points, 0.0, 0.0, LENGTH_M, WIDTH_M, bl_bbs.RESOLUTION_M,
                                *bands[4]).grid
    print(f"   bl_bbs  map grid demeaned before correlation (common/bev.py:71). Band 1 "
          f"occupancy {band1.mean():.4f}, band 5 {band5.mean():.4f}:")
    print("           a near-solid surface demeans to nothing, which is what stops trivial "
          "floor-matches-floor from")
    print(f"           swamping the sparse structure. Band 1's score surface here is "
          f"identically zero (max {float(bbs['surfaces'][0].max()):.1f}).")

    # Scored on the run's own scan sample and its own opening population, so these are the
    # numbers the animation drew and not an independent re-draw of them.
    opening = ga["history"][0]["fitness"]
    print(f"   bl_ga   raw inlier fraction, not demeaned: fraction of the "
          f"{len(ga['sample'])} sampled scan points within {bl_ga.INLIER_DIST_M} m of a "
          f"map point.")
    print(f"           At the truth {ga['truth_fitness']:.3f}. Across the "
          f"{bl_ga.POPULATION} uniformly random opening poses: mean {opening.mean():.3f}, "
          f"best {opening.max():.3f}.")
    print(f"           So a pose picked at random already 'explains' "
          f"{opening.mean() * 100:.0f}% of the scan. That floor is exactly what demeaning "
          f"removes.")

    print("\n6. per-band map occupancy and scan evidence, no longer drawn on the animation")
    band_notes = ("floor slab and rack feet", "rack beams at 1.5 and 3.0 m",
                  "rack beams at 4.5 and 6.0 m", "above the racking: columns, mezzanine, silos",
                  "roof deck, trusses, plant")
    sparse_doubled = [k + 1 for k, o in enumerate(bbs["occupancy"])
                      if o <= SOLID_OCCUPANCY and weights[k] > 1.0]
    for k, ((lo, hi), w, occ, cover, note) in enumerate(zip(bands, weights, bbs["occupancy"],
                                                             bbs["coverage"], band_notes), 1):
        state = "near-solid, demeans to almost nothing" if occ > SOLID_OCCUPANCY \
            else "sparse, survives demeaning"
        print(f"   band {k}  z {lo:5.2f}-{hi:5.2f}  weight {w:.0f}x  {note}")
        print(f"           map occupancy {occ:.3f}: {state}   |   scan evidence covers "
              f"{cover['pct']:.1f}% of the floor ({cover['returns']} returns)")
    print(f"   of the two doubled bands only band {sparse_doubled[0]} is sparse enough to "
          f"survive demeaning, which is why the doubled weight earns its keep there and not "
          f"on the roof")

    print(f"\n{'all checks passed' if ok else 'CHECKS FAILED'}")
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "docs", "images",
                                                   "search_ga_vs_slices.gif"))
    ap.add_argument("--verify", action="store_true", help="print the numbers, render nothing")
    ap.add_argument("--frames", type=int, default=FRAMES,
                     help="override the frame count; for size smoke tests only, since the "
                          "shared clock is sampled at this rate")
    args = ap.parse_args(argv)

    t0 = time.time()
    points, render = build_warehouse()
    scan = simulate_scan(points, *TRUE_POSE)
    print(f"warehouse {len(points):,} points, scan {len(scan):,} returns "
          f"({time.time() - t0:.1f}s)")

    t0 = time.time()
    ga = run_ga(scan, cKDTree(points), ((0.0, LENGTH_M), (0.0, WIDTH_M)), bl_ga.DEFAULT_SEED)
    print(f"bl_ga  {bl_ga.GENERATIONS} generations ({time.time() - t0:.1f}s)")

    t0 = time.time()
    bbs = run_bbs(scan, points)
    print(f"bl_bbs {bbs['placements']:,} placements ({time.time() - t0:.1f}s)")

    if args.verify:
        return 0 if verify(points, ga, bbs) else 1

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    t0 = time.time()
    render_gif(args.out, ga, bbs, render, frames=args.frames)
    size = os.path.getsize(args.out)
    # Pillow folds the identical hold frames into one long-duration frame, so the count
    # stored in the file is lower than the number rendered.
    with Image.open(args.out) as gif:
        stored = gif.n_frames
    print(f"wrote {args.out}  {size / 1e6:.2f} MB, {args.frames + HOLD_FRAMES} rendered, "
          f"{stored} stored ({size / stored / 1024:.0f} KB/stored frame, "
          f"{time.time() - t0:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
