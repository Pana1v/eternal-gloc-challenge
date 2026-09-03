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
from matplotlib.collections import LineCollection
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
HOLD_FRAMES = 10         # a verdict card at the end, which is also the loop point
DPI = 90                  # the GIF is committed, so pixels are a size decision
Z_EXAGGERATION = 4.0      # a 12 m ceiling over a 160 m hall is otherwise an invisible sliver
VIEW = (32.0, -62.0)      # fixed: a rotating camera changes every pixel of every frame,
                          # which is the difference between a 2 MB GIF and a 6 MB one
# "Still in contention" bar. docs/BASELINES.md uses the same 0.9-of-the-best convention for
# its alias counter, so the shrinking set in the inset is measured the way the tiers are.
CONTENTION = 0.9
# Above this plan-view occupancy a band is effectively a continuous surface, and demeaning
# cancels it. Not a synthetic artefact: eval/map_svg.py:16-19 says the released map's floor
# and ceiling "are continuous surfaces covering every cell", from bounds read off that map's
# own height histogram.
SOLID_OCCUPANCY = 0.9
# A 2 m error is six pixels across a 160 m hall, so the left panel gets a plan-view zoom.
# Without it the figure asserts that bl_ga missed and then shows a marker on the truth.
ZOOM_HALF_M = 10.0
# Racking grey, structure pale red, as docs/BASELINES.md:109-111 describes the report's
# scenario map. Pale here rather than saturated: the shell is context, not the subject.
SHELL_COLOR = "#d2d2d2"
BAND_DONE_COLOR = "#a9cfe8"   # bands already folded in, kept behind the one sweeping now
SHELL_STRUCTURE_COLOR = "#e6d2d2"


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


def _setup(ax, z_labels=True):
    ax.set_xlim(0, LENGTH_M)
    ax.set_ylim(0, WIDTH_M)
    ax.set_zlim(0, HEIGHT_M)
    ax.set_box_aspect((LENGTH_M, WIDTH_M, HEIGHT_M * Z_EXAGGERATION))
    ax.view_init(*VIEW)
    ax.set_xticks([0, 80, 160])
    ax.set_yticks([0, 93])
    ax.set_zticks([0, 6, 12])
    if not z_labels:
        ax.set_zticklabels([])
    ax.tick_params(labelsize=6, pad=-3)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_alpha(0.0)
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


def render_gif(out_path: str, ga, bbs, render, frames: int = FRAMES, fps: int = FPS,
                hold: int = HOLD_FRAMES):
    """One GIF, one clock. Both panels are rebuilt per frame rather than tracked as artists:
    mplot3d has no usable blitting, and at 60 frames the redraw is not the cost."""
    # The 3D boxes project wide and flat, so they are given more axes height than the
    # figure has and allowed to overflow: that crops the dead band under the content
    # instead of shipping it as committed bytes.
    fig = plt.figure(figsize=(12.0, 5.5))
    ax_ga = fig.add_axes((0.005, -0.045, 0.49, 0.92), projection="3d")
    ax_bbs = fig.add_axes((0.500, -0.045, 0.49, 0.92), projection="3d")
    ax_zoom = fig.add_axes((0.340, 0.555, 0.150, 0.230))
    ax_inset = fig.add_axes((0.838, 0.555, 0.150, 0.230))
    ax_cover = fig.add_axes((0.838, 0.145, 0.150, 0.230))

    history = ga["history"]
    tx, ty, tyaw = TRUE_POSE
    bands, weights = bbs["bands"], bbs["weights"]
    surfaces = bbs["surfaces"]
    cells = bbs["grid_shape"][0] * bbs["grid_shape"][1]
    band_notes = ("floor slab and rack feet", "rack beams at 1.5 and 3.0 m",
                  "rack beams at 4.5 and 6.0 m", "above the racking: columns, mezzanine, silos",
                  "roof deck, trusses, plant")
    # plan-view racking inside the zoom window, built once
    zoom_box = (tx - ZOOM_HALF_M, tx + ZOOM_HALF_M, ty - ZOOM_HALF_M, ty + ZOOM_HALF_M)
    zoom_rows, zoom_feet = [], []
    for y in RACK_ROWS_Y:
        if not zoom_box[2] <= y <= zoom_box[3]:
            continue
        for bx0, bx1 in RACK_BLOCKS_X:
            x0, x1 = max(bx0, zoom_box[0]), min(bx1, zoom_box[1])
            if x1 <= x0:
                continue
            zoom_rows.append([(x0, y), (x1, y)])
            zoom_feet += [(x, y) for x in np.arange(bx0, bx1 + 1e-9, BAY_PITCH_M)
                          if x0 <= x <= x1]
    zoom_feet = np.array(zoom_feet).reshape(-1, 2)

    plan_rows = [[(a[0], a[1]), (b[0], b[1])] for a, b in render["racks"]
                 if abs(a[2] - b[2]) < 1e-9]

    per_band = [_clip(render["racks"], lo, hi) + _clip(render["structure"], lo, hi)
                for lo, hi in bands]
    below_band = [sum(per_band[:k], []) for k in range(len(bands))]

    truth_fit = ga["truth_fitness"]
    # The two blind zones are set by the vertical field of view and the mount height, not by
    # the 70 m range: occlusion and elevation limits are what cap coverage here.
    floor_blind = bl_ga.SENSOR_HEIGHT_M / np.tan(np.radians(-LIDAR_EL_DEG[0]))
    roof_blind = (ROOF_Z_M - bl_ga.SENSOR_HEIGHT_M) / np.tan(np.radians(LIDAR_EL_DEG[1]))
    hall_note = (f"{LENGTH_M:.0f} x {WIDTH_M:.0f} x {HEIGHT_M:.0f} m hall, "
                 f"{LENGTH_M * WIDTH_M:,.0f} m2 of floor,\n{len(RACK_ROWS_Y)} rack rows at "
                 f"{RACK_PITCH_M:.1f} m pitch")
    scatter_note = (f"{hall_note}, sampled by {bl_ga.POPULATION} poses at a "
                    f"{np.sqrt(LENGTH_M * WIDTH_M / bl_ga.POPULATION):.1f} m "
                    f"grid-equivalent spacing: coarser than the pitch")
    fig.text(0.012, 0.980, "bl_ga   sampled search over (x, y, yaw)", fontsize=9.5,
             va="top", color=GA_COLOR)
    fig.text(0.507, 0.980, "bl_bbs   exhaustive search, five height bands", fontsize=9.5,
             va="top", color=BBS_COLOR)
    fig.text(0.012, 0.945, scatter_note, fontsize=6.8, va="top", color="#555555",
             linespacing=1.45)
    fig.text(0.507, 0.945, f"{hall_note}, every placement scored at "
             f"{bl_bbs.RESOLUTION_M} m and {bl_bbs.YAW_STEP_DEG:.0f} deg", fontsize=6.8,
             va="top", color="#555555", linespacing=1.45)
    info_ga = fig.text(0.012, 0.872, "", fontsize=7.6, va="top", color="#222222",
                       linespacing=1.5)
    info_bbs = fig.text(0.507, 0.872, "", fontsize=7.6, va="top", color="#222222",
                        linespacing=1.5)
    footer = fig.text(0.5, 0.005, "", ha="center", va="bottom", fontsize=6.8,
                      color="#333333", linespacing=1.5)

    def draw(frame):
        verdict = frame >= frames
        t = min(frame, frames - 1) / (frames - 1)
        elapsed = t * GA_SEC
        gen = min(int(round(t * (len(history) - 1))), len(history) - 1)
        band = max(1, min(int(np.ceil(min(1.0, elapsed / BBS_SEC) * len(bands))), len(bands)))
        done = elapsed >= BBS_SEC

        ax_ga.clear()
        ax_bbs.clear()
        ax_zoom.clear()
        ax_inset.clear()
        ax_cover.clear()

        # --- left: the population -----------------------------------------------------
        state = history[gen]
        poses, fitness = state["poses"], state["fitness"]
        elites = np.argsort(fitness)[::-1][:bl_ga.ELITE_K]
        floor = np.full(len(poses), bl_ga.SENSOR_HEIGHT_M)
        _setup(ax_ga)
        _shell(ax_ga, render)
        ax_ga.scatter(poses[:, 0], poses[:, 1], floor, s=5.0, c=GA_COLOR,
                      depthshade=False, linewidths=0)
        ax_ga.scatter(poses[elites, 0], poses[elites, 1], floor[elites], s=16.0, c=GA_COLOR,
                      depthshade=False, edgecolors="black", linewidths=0.35)
        _pose(ax_ga, tx, ty, tyaw, TRUTH_COLOR, 10)
        best = poses[fitness.argmax()]
        ga_err = float(np.hypot(best[0] - tx, best[1] - ty))

        if verdict:
            for hx, hy, hyaw in ga["hypotheses"]:
                _pose(ax_ga, hx, hy, hyaw, GA_COLOR, 11, marker="o", filled=False)
            info_ga.set_text(
                f"done: {(bl_ga.GENERATIONS + 1) * bl_ga.POPULATION:,} poses evaluated, "
                f"best inlier fraction {fitness.max():.3f} against {truth_fit:.3f} "
                f"at the truth\n"
                f"{len(ga['hypotheses'])} modes survive the {bl_ga.HYPOTHESIS_MIN_SEP_M:.0f} m "
                f"separation filter, {ga['n_submitted']} of them clears the confidence "
                f"filter\n"
                f"submitted pose {ga_err:.2f} m from the truth: the right aisle, the wrong "
                f"place along it")
        else:
            info_ga.set_text(
                f"generation {gen} of {bl_ga.GENERATIONS}   "
                f"{(gen + 1) * bl_ga.POPULATION:,} poses evaluated\n"
                f"jitter {state['sigma_xy']:.3f} m / {state['sigma_yaw']:.2f} deg   "
                f"best inlier fraction {fitness.max():.3f}\n"
                f"best pose {ga_err:.2f} m from the truth")

        # --- left inset: the same scene at a scale where 2 m is visible ----------------
        ax_zoom.add_collection(LineCollection(zoom_rows, colors="#c4c4c4", linewidths=1.6))
        ax_zoom.scatter(zoom_feet[:, 0], zoom_feet[:, 1], s=5.0, c="#8c8c8c", linewidths=0,
                        marker="|")
        inside = ((np.abs(poses[:, 0] - tx) <= ZOOM_HALF_M)
                  & (np.abs(poses[:, 1] - ty) <= ZOOM_HALF_M))
        ax_zoom.scatter(poses[inside, 0], poses[inside, 1], s=6.0, c=GA_COLOR, linewidths=0)
        if abs(best[0] - tx) <= ZOOM_HALF_M and abs(best[1] - ty) <= ZOOM_HALF_M:
            # the dotted line back to the truth, as in eval/render_scenarios.py
            ax_zoom.plot([best[0], tx], [best[1], ty], color=GA_COLOR, linestyle=":",
                         linewidth=0.9)
            ax_zoom.plot([best[0]], [best[1]], marker="o", markersize=6,
                         markerfacecolor="none", markeredgecolor=GA_COLOR,
                         markeredgewidth=1.1)
        ax_zoom.plot([tx], [ty], marker="*", color=TRUTH_COLOR, markersize=7,
                     markeredgewidth=0.0)
        ax_zoom.set_xlim(zoom_box[0], zoom_box[1])
        ax_zoom.set_ylim(zoom_box[2], zoom_box[3])
        ax_zoom.set_aspect("equal")
        ax_zoom.set_xticks([])
        ax_zoom.set_yticks([])
        ax_zoom.set_title(f"zoom, {2 * ZOOM_HALF_M:.0f} m across: best pose vs truth",
                          fontsize=5.9, pad=2.0)
        for spine in ax_zoom.spines.values():
            spine.set_linewidth(0.4)

        # --- right: the bands ---------------------------------------------------------
        lo, hi = bands[band - 1]
        _setup(ax_bbs, z_labels=False)
        _shell(ax_bbs, render, done=below_band[band - 1], active=per_band[band - 1])
        surface = surfaces[band - 1]
        peak = float(surface.max())
        if peak > 0.0:
            # an open ring, so it reads as agreement when it lands on the truth's star
            i, j = np.unravel_index(surface.argmax(), surface.shape)
            _pose(ax_bbs, i * bl_bbs.RESOLUTION_M, j * bl_bbs.RESOLUTION_M,
                  bbs["pose"][2], BBS_COLOR, 9, marker="o", filled=False)
        _pose(ax_bbs, tx, ty, tyaw, TRUTH_COLOR, 10)
        bbs_err = float(np.hypot(bbs["pose"][0] - tx, bbs["pose"][1] - ty))
        if verdict:
            sparse = [k + 1 for k, o in enumerate(bbs["occupancy"])
                      if o <= SOLID_OCCUPANCY and weights[k] > 1.0]
            info_bbs.set_text(
                f"done in {BBS_SEC:.2f} s, all {len(bands)} bands folded in, all "
                f"{bbs['placements']:,} placements scored\n"
                f"the global maximum over every placement, not a sampled peak\n"
                f"of the two doubled bands only band {sparse[0]} is sparse enough to survive "
                f"demeaning\n"
                f"winning pose {bbs_err:.2f} m from the truth, one BEV cell, pre-ICP")
        else:
            occ = bbs["occupancy"][band - 1]
            info_bbs.set_text(
                f"band {band} of {len(bands)}   z {lo:.2f} to {hi:.2f} m   "
                f"weight {weights[band - 1]:.0f}x\n"
                f"{band_notes[band - 1]}\n"
                f"map occupancy {occ:.3f}: "
                + ("near-solid, demeans to almost nothing" if occ > SOLID_OCCUPANCY
                   else "sparse, survives demeaning") + "\n"
                f"all {bbs['placements']:,} placements scored, "
                f"{bbs['placements'] // 12300:,}x bl_ga's 12,300")

        # --- inset: how many placements are still tied --------------------------------
        if peak <= 0.0:
            in_play = cells
            ax_inset.text(0.5, 0.5, "all of them:\na fully occupied band\ndemeans to "
                          "exactly zero", ha="center", va="center", fontsize=6.0,
                          color="#222222", transform=ax_inset.transAxes)
        else:
            contenders = np.argwhere(surface >= CONTENTION * peak)
            in_play = len(contenders)
            ax_inset.scatter(contenders[:, 0] * bl_bbs.RESOLUTION_M,
                             contenders[:, 1] * bl_bbs.RESOLUTION_M,
                             s=2.2, c=BBS_COLOR, linewidths=0)
        ax_inset.plot([tx], [ty], marker="*", color=TRUTH_COLOR, markersize=5.5,
                      markeredgewidth=0.0)
        ax_inset.set_xlim(0, LENGTH_M)
        ax_inset.set_ylim(0, WIDTH_M)
        ax_inset.set_aspect("equal")
        ax_inset.set_xticks([])
        ax_inset.set_yticks([])
        ax_inset.set_title(f"within {100 - CONTENTION * 100:.0f}% of the best: {in_play:,}",
                           fontsize=5.9, pad=2.0)
        for spine in ax_inset.spines.values():
            spine.set_linewidth(0.4)

        # --- right lower inset: where this band's evidence came from ------------------
        cover = bbs["coverage"][band - 1]
        ax_cover.add_collection(LineCollection(plan_rows, colors="#dcdcdc", linewidths=0.5))
        if cover["returns"]:
            ax_cover.scatter(cover["xy"][:, 0], cover["xy"][:, 1], s=0.5, c=BBS_COLOR,
                             linewidths=0)
        ax_cover.plot([tx], [ty], marker="*", color=TRUTH_COLOR, markersize=5.5,
                      markeredgewidth=0.0)
        ax_cover.set_xlim(0, LENGTH_M)
        ax_cover.set_ylim(0, WIDTH_M)
        ax_cover.set_aspect("equal")
        ax_cover.set_xticks([])
        ax_cover.set_yticks([])
        ax_cover.set_title(f"band {band} evidence: {cover['pct']:.1f}% of the floor",
                           fontsize=5.9, pad=2.0)
        for spine in ax_cover.spines.values():
            spine.set_linewidth(0.4)

        if verdict:
            footer.set_text(
                f"on the released dev split bl_bbs scores 97.74 with SR@fine 0.975 and "
                f"bl_ga scores 22.36 with SR@fine 0.025 (docs/BASELINES.md), and the repo "
                f"credits the exhaustive search for the difference\n"
                f"this run is synthetic geometry, one scenario, and reproduces the same "
                f"two outcomes: an exact match against a confident near miss")
            return
        footer.set_text(
            f"{elapsed:.2f} s of {GA_SEC:.2f} s on one clock scaled to the measured "
            f"sec/scenario in docs/BASELINES.md; bl_bbs finishes at {BBS_SEC:.2f} s despite "
            f"scoring {bbs['placements'] // 12300:,}x more placements\n"
            f"both searches are SE(2) with z pinned at {bl_ga.SENSOR_HEIGHT_M:.1f} m   |   "
            f"z drawn {Z_EXAGGERATION:.0f}x   |   black star: the truth   |   "
            f"the {LIDAR_EL_DEG[0]:.0f} to +{LIDAR_EL_DEG[1]:.0f} deg vertical field of view "
            f"blinds the sensor to the floor within {floor_blind:.1f} m and to the roof "
            f"within {roof_blind:.1f} m, so no band sees the whole hall")

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
    print(f"           a near-solid surface demeans to nothing, which is what stops trivial "
          f"floor-matches-floor from")
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
