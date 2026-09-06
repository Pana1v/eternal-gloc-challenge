#!/usr/bin/env python3
"""Animates how bl_bbs and bl_ga search, side by side, on two released dev scenarios.

The results tables in docs/BASELINES.md say which baseline wins. They do not show what
either one does, and a single score hides that bl_ga's outcome is a coin toss: it finds
000030 on three seeds of five and 000010 on none. So the GIF runs both methods twice, once
on each, at bl_ga's shipped default seed.

Both panels run the actual searches, not a scripted mock-up:

  left   bl_ga    the real population loop, using bl_ga's own constants and genetic
                  operators, scored by the same inlier fraction against a KD-tree
  right  bl_bbs   the real FFT correlative match from baselines/common/bev.py, with
                  band edges from bl_bbs.build_slice_bands

The inset under bl_ga's column splits its fitness into the five height bands it averages
over. That is where the failure becomes legible rather than merely visible: at a pose 142 m
from the truth it still explains most of every band, because floor and roof deck match
almost anywhere and the bands that would disagree are the sparse ones.

Both search SE(2) with z pinned at the rig height, which is what the baselines do. bl_ga's
"3D" is its scoring metric, not its search space, so neither panel scatters poses through a
volume: every pose in this figure sits on the floor plane.

Needs the dev split: scenarios/dev/ is in the repository and the map is one
./tools/fetch_map.sh away. --verify does not, and deliberately so: its expected numbers
were derived against the synthetic warehouse build_warehouse still generates, so it keeps
checking against that and stays runnable with no data at all.

Usage:
    python tools/render_search_animation.py --out docs/images/search_ga_vs_slices.gif
    python tools/render_search_animation.py --verify        # numbers only, no render

Constants and operators are imported from baselines/bl_bbs/run.py and baselines/bl_ga/run.py,
so a parameter change follows through, but a change to what bl_ga's fitness *means* would
need the caption text here and the matching points in docs/BASELINES.md revisited.

Which fitness ran is never left to a default, because bl_ga has switches now (--banding,
--crossover) and this figure makes claims that depend on the answer. run_ga takes the rule
by name: --verify asks for "none", the un-demeaned raw fraction its printed numbers
describe, and the rendered scenarios ask for "per-band", because their captions call an
outcome a find or an alias and that has to be the rule the scored baseline used. --verify
prints the name in force, so drift stays detectable.
"""

import argparse
import importlib.util
import os
import time

import matplotlib
matplotlib.use("Agg")   # no display in CI or the container; must precede pyplot
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
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

# --- the released dev split ---------------------------------------------------------------
# The figure now runs on real scenarios. They are on master (scenarios/dev/) and the map is
# one ./tools/fetch_map.sh away, so this still reproduces from a clone; it is no longer
# self-contained, which is the price of showing measured outcomes instead of chosen ones.
# build_warehouse and simulate_scan stay for --verify, whose expected numbers were derived
# against that synthetic geometry.
SCENARIO_ROOT = os.path.join(REPO_ROOT, "scenarios", "dev", "A")
GT_PATH = os.path.join(REPO_ROOT, "scenarios", "dev", "gt", "A.txt")
MAP_PATH = os.path.join(REPO_ROOT, "map", "prior_map.pcd")
# Both at bl_ga's shipped default seed, so neither panel is a seed picked to flatter or
# damn it. Over five seeds bl_ga finds 000030 on three and 000010 on none; the captions say
# so, because one run of a stochastic search is an anecdote and docs/BASELINES.md is blunt
# about that.
SUCCESS_ID, SUCCESS_HITS = "000030", 3
FAILURE_ID, FAILURE_HITS = "000010", 0
SEEDS_TRIED = 5
# The shell is 8.8 M points; drawn whole it would be a black rectangle and would take longer
# to rasterize than the search takes to run. Decimated once, reused for every frame, and
# spread evenly over the height bands rather than uniformly (see build_shell).
SHELL_POINTS = 16000
SHELL_SEED = 7


def load_map():
    """The prior map, the search bounds, and the footprint the axis limits come from.

    The bounds are the map's own min and max on each axis, exactly as bl_ga.main computes
    them, not (0, extent). The released map starts a few centimetres below zero, and seeding
    a population over a shifted rectangle draws different poses from the same seed, which is
    enough to send a stochastic search somewhere else entirely.
    """
    if not os.path.exists(MAP_PATH):
        raise SystemExit(f"no map at {MAP_PATH}; run ./tools/fetch_map.sh first")
    points = np.asarray(o3d.io.read_point_cloud(MAP_PATH).points)
    bounds = ((points[:, 0].min(), points[:, 0].max()),
               (points[:, 1].min(), points[:, 1].max()))
    return points, bounds, (float(points[:, 0].max()), float(points[:, 1].max()),
                             float(points[:, 2].max()))


def load_scan(scenario_id: str) -> np.ndarray:
    """One scenario's lidar return, in the rig's own frame, exactly as a baseline reads it."""
    path = os.path.join(SCENARIO_ROOT, scenario_id, "lidar.pcd")
    if not os.path.exists(path):
        raise SystemExit(f"no scenario at {path}")
    return np.asarray(o3d.io.read_point_cloud(path).points)


def load_truth(scenario_id: str):
    """(x, y, yaw) from the published ground truth's 3x4 row-major pose."""
    for line in open(GT_PATH):
        parts = line.split()
        if parts and parts[0] == scenario_id:
            m = np.array([float(v) for v in parts[1:13]]).reshape(3, 4)
            return float(m[0, 3]), float(m[1, 3]), float(np.arctan2(m[1, 0], m[0, 0]))
    raise SystemExit(f"{scenario_id} not in {GT_PATH}")


def build_shell(map_points: np.ndarray, bands) -> np.ndarray:
    """A decimated copy of the map to draw the hall with, sampled to equal ink per band.

    The synthetic figure drew line segments because it knew where every beam and truss was.
    A real map is only points, and thinning it uniformly draws grey noise: two thirds of the
    map is the floor slab and the roof deck, so a uniform sample is mostly those two planes
    and the racking that gives the hall its shape never appears.

    Sampling each height band to the same count instead spends the ink where the structure
    is. The floor still reads as a floor, because a plane drawn with 3,000 points is still
    obviously a plane, while 3,000 points on the racking is enough to see rows and aisles.
    This is a drawing decision only: both searches score against the full map.
    """
    rng = np.random.default_rng(SHELL_SEED)
    per_band = max(1, SHELL_POINTS // len(bands))

    chosen = []
    for z_lo, z_hi in bands:
        idx = np.flatnonzero((map_points[:, 2] >= z_lo) & (map_points[:, 2] < z_hi))
        if not len(idx):
            continue
        chosen.append(rng.choice(idx, min(per_band, len(idx)), replace=False))
    return map_points[np.concatenate(chosen)]


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

def run_ga(scan: np.ndarray, tree: cKDTree, bounds, seed: int, truth, bands, band_weights,
            banding: str):
    """bl_ga's population loop, unrolled one generation at a time so each can be drawn.

    bl_ga.evolve returns only the final population, so the loop is restated here; every
    constant and every genetic operator is imported, and the fitness is bl_ga.evaluate
    itself, so the trajectory is the baseline's and not an imitation of it.

    Mutation-only by design, matching the baseline's shipped default. bl_ga also has a
    --crossover switch now, but docs/BASELINES.md reports it as a measurement that no
    operator won, so the figure shows the search the results table describes.
    """
    rng = np.random.default_rng(seed)
    sample = (scan if len(scan) <= bl_ga.SCAN_SAMPLE
              else scan[rng.choice(len(scan), bl_ga.SCAN_SAMPLE, replace=False)])
    # Named at every call site, never defaulted: bl_ga's default is free to move and this
    # figure makes claims that depend on which rule ranked the poses. --verify asks for
    # "none", the un-demeaned raw inlier fraction its reported numbers describe; the
    # rendered scenarios ask for "per-band", because their captions call an outcome a find
    # or an alias and that has to be the fitness the shipped baseline actually scored with.
    weights = (None if banding == "none"
                else bl_ga.scan_point_weights(sample, bands, band_weights, banding))
    poses = bl_ga.random_population(rng, bl_ga.POPULATION, bounds)
    sigma_xy, sigma_yaw = bl_ga.SIGMA_XY_M, bl_ga.SIGMA_YAW_DEG
    history = []

    for _ in range(bl_ga.GENERATIONS):
        fitness = bl_ga.evaluate(poses, sample, tree, weights)
        best = poses[fitness.argmax()]
        history.append({"poses": poses, "fitness": fitness,
                        "sigma_xy": sigma_xy, "sigma_yaw": sigma_yaw,
                        "bands": per_band_inliers(best, sample, tree, bands)})

        elites = poses[np.argsort(fitness)[::-1][:bl_ga.ELITE_K]]
        n_children = bl_ga.POPULATION - bl_ga.ELITE_K - bl_ga.IMMIGRANTS
        poses = np.concatenate([
            elites,
            bl_ga.mutate(elites, rng, n_children, sigma_xy, sigma_yaw),
            bl_ga.random_population(rng, bl_ga.IMMIGRANTS, bounds),
        ], axis=0)
        sigma_xy *= bl_ga.SIGMA_DECAY
        sigma_yaw *= bl_ga.SIGMA_DECAY

    fitness = bl_ga.evaluate(poses, sample, tree, weights)
    best = poses[fitness.argmax()]
    history.append({"poses": poses, "fitness": fitness,
                    "sigma_xy": sigma_xy, "sigma_yaw": sigma_yaw,
                    "bands": per_band_inliers(best, sample, tree, bands)})

    top_poses, top_fitness = bl_ga.distinct_top(poses, fitness, bl_ga.N_HYPOTHESES,
                                                bl_ga.HYPOTHESIS_MIN_SEP_M)
    kept, _ = bl_ga.confident_subset(top_poses, top_fitness, bl_ga.HYPOTHESIS_KEEP_RATIO)
    return {"history": history, "hypotheses": top_poses, "fitness": top_fitness,
            "n_submitted": len(kept), "sample": sample,
            "error_m": float(np.hypot(best[0] - truth[0], best[1] - truth[1])),
            "banding": banding,
            "truth_fitness": float(bl_ga.evaluate(np.array([list(truth)]), sample, tree,
                                                   weights)[0])}


def run_bbs(scan: np.ndarray, map_points: np.ndarray, truth):
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
            "placements": nx * ny * n_yaws, "coverage": scan_coverage(scan, bands, truth, LENGTH_M, WIDTH_M),
            "occupancy": [float(g.mean()) for g in map_grids]}


def scan_coverage(scan: np.ndarray, bands, truth, length_m: float, width_m: float):
    """Where on the floor plan each band's scan evidence actually comes from.

    Sliced on the scan's own z against the band edges, which is what match_scan_to_map
    does, so the footprint drawn is the one that band's correlation consumed. Answers a
    question the height axis cannot: a band can be tall and still be told almost nothing,
    because the sensor's vertical limits and the racking decide what reaches it.
    """
    tx, ty, tyaw = truth
    c, s = np.cos(tyaw), np.sin(tyaw)
    world = np.stack([c * scan[:, 0] - s * scan[:, 1] + tx,
                      s * scan[:, 0] + c * scan[:, 1] + ty], axis=1)
    out = []
    for z_lo, z_hi in bands:
        xy = world[(scan[:, 2] >= z_lo) & (scan[:, 2] < z_hi)]
        touched = len(set(map(tuple, np.floor(xy).astype(np.int64)))) if len(xy) else 0
        out.append({"xy": xy, "returns": len(xy), "cells": touched,
                    "pct": 100.0 * touched / (length_m * width_m)})
    return out


# --- rendering ---------------------------------------------------------------------------
# Measured sec/scenario from the results table in docs/BASELINES.md. The animation runs on
# one clock scaled to these, so bl_bbs finishes and holds while bl_ga is still working.
BBS_SEC, GA_SEC = 2.65, 10.27
FRAMES, FPS = 46, 10
HOLD_FRAMES = 10         # pause on the final state, which is also the loop point
DPI = 80                  # the GIF is committed, so pixels are a size decision
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


def _band_mask(shell: np.ndarray, z_lo: float, z_hi: float) -> np.ndarray:
    """The shell points inside a height band, so a band lights up the geometry that actually
    lives in it rather than a floating slab. The segment clipper this replaces could split a
    truss where it crossed an edge; points either fall in the band or they do not."""
    return (shell[:, 2] >= z_lo) & (shell[:, 2] < z_hi)


def per_band_inliers(pose, scan: np.ndarray, tree: cKDTree, bands) -> np.ndarray:
    """One inlier fraction per height band for a single pose.

    bl_ga.evaluate collapses these into the one weighted number the search ranks on. The
    figure needs them apart, because the per-band average is the whole point of that fitness:
    a band holding 3% of the returns still gets a full vote, and that is what lets the sparse
    above-rack structure outvote a floor that matches nearly anywhere.

    Deliberately the un-weighted fraction per band, not the weighted contribution, so a bar
    reads as "how much of what I can see up there did I explain" and the 2x weighting stays
    a property of the caption rather than a distortion baked into the bar.
    """
    x, y, yaw = pose
    c, s_ = np.cos(yaw), np.sin(yaw)
    world = np.stack([c * scan[:, 0] - s_ * scan[:, 1] + x,
                      s_ * scan[:, 0] + c * scan[:, 1] + y,
                      scan[:, 2] + bl_ga.SENSOR_HEIGHT_M], axis=1)
    dist, _ = tree.query(world, k=1, workers=-1)
    inlier = dist <= bl_ga.INLIER_DIST_M

    out = []
    for z_lo, z_hi in bands:
        in_band = (world[:, 2] >= z_lo) & (world[:, 2] < z_hi)
        out.append(float(inlier[in_band].mean()) if in_band.any() else 0.0)
    return np.array(out)


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


def _shell(ax, shell: np.ndarray, done=None, active=None):
    """The warehouse, drawn once per frame. Bands already scored stay in a pale blue and the
    band being scored now is drawn over them in full colour, so the eye follows one sweep
    upward instead of watching the whole hall turn blue and stay there."""
    ax.scatter(shell[:, 0], shell[:, 1], shell[:, 2], s=0.35, c=SHELL_COLOR,
                depthshade=False, linewidths=0)
    if done is not None and done.any():
        ax.scatter(shell[done, 0], shell[done, 1], shell[done, 2], s=0.5,
                    c=BAND_DONE_COLOR, depthshade=False, linewidths=0)
    if active is not None and active.any():
        ax.scatter(shell[active, 0], shell[active, 1], shell[active, 2], s=0.9,
                    c=BBS_COLOR, depthshade=False, linewidths=0)


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


BAND_PANEL_RECT = (0.070, 0.052, 0.150, 0.145)   # inset over the GA column's dead corner


def _band_panel(fig):
    """The GA column's per-band inlier inset.

    Drawn as a flat 2D inset rather than another 3D panel because it answers a question the
    hall cannot show: bl_ga's fitness is the average of these five numbers, so a bar that
    stays low while the others fill is the search explaining the floor and the roof deck and
    nothing that distinguishes one bay from another. Given an opaque backing, since it sits
    over a transparent 3D axes and would otherwise be read through.
    """
    ax = fig.add_axes(BAND_PANEL_RECT)
    ax.set_facecolor("#ffffff")
    ax.patch.set_alpha(0.88)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#bbbbbb")
    ax.set_xlim(0.0, 1.0)
    ax.tick_params(labelsize=5.5, length=2, colors="#666666")
    ax.set_xticks([0.0, 0.5, 1.0])
    return ax


def render_gif(out_path: str, acts, shell: np.ndarray, frames: int = FRAMES, fps: int = FPS,
                hold: int = HOLD_FRAMES):
    """One GIF, two acts, four axes per act: one method per column, each shown top-down above
    and in perspective below, plus the GA's per-band inlier inset. Every panel is rebuilt per
    frame rather than tracked as artists: mplot3d has no usable blitting, and at these frame
    counts the redraw is not the cost.

    The two acts are the same pair of methods on two real scenarios, one bl_ga finds and one
    it aliases, so the contrast is a measured outcome rather than a claim in a caption."""
    # mplot3d draws into a square viewport inside its axes, so a wide, short cell renders a
    # small plan view with dead space either side and raising `zoom` clips rather than fills.
    # Give each axes far more height than its row needs and let the empty part of the square
    # overflow into the neighbouring row, which is transparent. The panels overlap; their ink
    # does not.
    fig = plt.figure(figsize=(10.0, 8.0))
    ax_ga_top = fig.add_axes((0.005, 0.444, 0.49, 0.6125), projection="3d")
    ax_bbs_top = fig.add_axes((0.505, 0.444, 0.49, 0.6125), projection="3d")
    ax_ga_persp = fig.add_axes((0.005, -0.010, 0.49, 0.600), projection="3d")
    ax_bbs_persp = fig.add_axes((0.505, -0.010, 0.49, 0.600), projection="3d")
    ax_bands = _band_panel(fig)   # added last, so it paints over the 3D panels

    fig.text(0.5, 0.997, "Global Localization", ha="center", va="top", fontsize=16,
             color="#222222", fontweight="bold")
    fig.text(0.25, 0.958, "Genetic Evolution", ha="center", va="top", fontsize=11,
             color=GA_COLOR)
    fig.text(0.75, 0.958, "Fast Fourier Transform", ha="center", va="top", fontsize=11,
             color=BBS_COLOR)
    caption = fig.text(0.5, 0.936, "", ha="center", va="top", fontsize=9, color="#555555")
    clock = fig.text(0.5, 0.008, "", ha="center", va="bottom", fontsize=13, color="#444444")

    act_len = frames + hold
    shell_z = shell[:, 2]

    def draw(index):
        act = acts[min(index // act_len, len(acts) - 1)]
        frame = index % act_len
        ga, bbs, truth = act["ga"], act["bbs"], act["truth"]
        history, bands = ga["history"], bbs["bands"]
        tx, ty, tyaw = truth

        verdict = frame >= frames
        t = min(frame, frames - 1) / (frames - 1)
        elapsed = t * GA_SEC
        gen = min(int(round(t * (len(history) - 1))), len(history) - 1)
        band = max(1, min(int(np.ceil(min(1.0, elapsed / BBS_SEC) * len(bands))), len(bands)))

        clock.set_text(f"{elapsed:.2f} s   (bl_bbs done, +{elapsed - BBS_SEC:.2f} s ago)"
                        if elapsed >= BBS_SEC else f"{elapsed:.2f} s")
        caption.set_text(
            f"scenario {act['sid']}: bl_ga {act['outcome']}"
            f" ({act['hits']} of {SEEDS_TRIED} seeds)   bl_bbs finds it on every seed")

        for ax in (ax_ga_top, ax_ga_persp, ax_bbs_top, ax_bbs_persp):
            ax.clear()
        ax_bands.clear()

        range_ring = act["range_ring"]

        # --- bl_ga: the population, drawn top-down and in perspective ------------------
        state = history[gen]
        poses, fitness = state["poses"], state["fitness"]
        elites = np.argsort(fitness)[::-1][:bl_ga.ELITE_K]
        floor = np.full(len(poses), bl_ga.SENSOR_HEIGHT_M)
        best = poses[fitness.argmax()]
        # Elites near-tied with a distant pose are real, but highlighting all of them makes a
        # transient secondary mode look like a rendering glitch next to the one the run
        # submits. Only the elites clustered with the current best get the elite treatment.
        near_best = elites[np.hypot(poses[elites, 0] - best[0],
                                    poses[elites, 1] - best[1]) <= ELITE_CLUSTER_RADIUS_M]

        for ax, top in ((ax_ga_top, True), (ax_ga_persp, False)):
            _setup(ax, top=top)
            _shell(ax, shell)
            ax.add_collection3d(Line3DCollection(range_ring, colors=RANGE_COLOR,
                                                 linewidths=0.8, linestyles=(0, (4, 3))))
            ax.scatter(poses[:, 0], poses[:, 1], floor, s=5.0, c=GA_COLOR,
                       depthshade=False, linewidths=0)
            ax.scatter(poses[near_best, 0], poses[near_best, 1], floor[near_best], s=16.0,
                       c=GA_COLOR, depthshade=False, edgecolors="black", linewidths=0.35)
            _pose(ax, best[0], best[1], best[2], GA_COLOR, 10, marker="o", filled=False)
            _pose(ax, tx, ty, tyaw, TRUTH_COLOR, 10)
            if verdict:
                for hx, hy, hyaw in ga["hypotheses"]:
                    _pose(ax, hx, hy, hyaw, GA_COLOR, 11, marker="o", filled=False)

        # --- the five bands bl_ga's fitness averages over ------------------------------
        fracs = state["bands"]
        ypos = np.arange(len(bands))
        ax_bands.barh(ypos, fracs, height=0.62, color=GA_COLOR, edgecolor="none")
        ax_bands.set_xlim(0.0, 1.0)
        ax_bands.set_ylim(-0.6, len(bands) - 0.4)
        ax_bands.set_yticks(ypos)
        ax_bands.set_yticklabels(
            [f"{max(lo, 0.0):.0f}-{hi:.0f} m{'  2x' if w > 1.0 else ''}"
             for (lo, hi), w in zip(bands, bbs["weights"])], fontsize=5.5, color="#666666")
        ax_bands.set_title("per-band inliers, best pose", fontsize=6.5, color="#444444",
                            pad=3)
        ax_bands.tick_params(labelsize=5.5, length=2, colors="#666666")
        ax_bands.set_xticks([0.0, 0.5, 1.0])
        for side in ("top", "right"):
            ax_bands.spines[side].set_visible(False)

        # --- bl_bbs: the band sweep, drawn top-down and in perspective ------------------
        surface = bbs["surfaces"][band - 1]
        peak = float(surface.max())
        winner = np.unravel_index(surface.argmax(), surface.shape) if peak > 0.0 else None
        contenders = np.argwhere(surface >= CONTENTION * peak) if peak > 0.0 else None
        cover_xy = bbs["coverage"][band - 1]["xy"]
        done = shell_z < bands[band - 1][0]
        active = (shell_z >= bands[band - 1][0]) & (shell_z < bands[band - 1][1])

        for ax, top in ((ax_bbs_top, True), (ax_bbs_persp, False)):
            _setup(ax, top=top)
            _shell(ax, shell, done=done, active=active)
            ax.add_collection3d(Line3DCollection(range_ring, colors=RANGE_COLOR,
                                                 linewidths=0.8, linestyles=(0, (4, 3))))
            if top and len(cover_xy):
                ax.scatter(cover_xy[:, 0], cover_xy[:, 1],
                          np.full(len(cover_xy), bl_ga.SENSOR_HEIGHT_M), s=1.2, c=BBS_COLOR,
                          depthshade=False, alpha=0.25, linewidths=0)
            if winner is not None:
                _pose(ax, winner[0] * bl_bbs.RESOLUTION_M, winner[1] * bl_bbs.RESOLUTION_M,
                      bbs["pose"][2], BBS_COLOR, 9, marker="o", filled=False)
            _pose(ax, tx, ty, tyaw, TRUTH_COLOR, 10)

        # placements still within CONTENTION of the best, in plan view only: this is what
        # collapses from "most of the band" to "one cell" as the sweep narrows, and it is
        # exactly where the aliasing bays line up.
        if contenders is not None:
            cx = contenders[:, 0] * bl_bbs.RESOLUTION_M
            cy = contenders[:, 1] * bl_bbs.RESOLUTION_M
            near = np.hypot(cx - winner[0] * bl_bbs.RESOLUTION_M,
                            cy - winner[1] * bl_bbs.RESOLUTION_M) <= CONTENTION_NEAR_M
            z = np.full(len(cx), bl_ga.SENSOR_HEIGHT_M)
            if np.any(~near):
                ax_bbs_top.scatter(cx[~near], cy[~near], z[~near], s=4.0, c=BBS_COLOR,
                                   depthshade=False, alpha=0.35, linewidths=0)
            ax_bbs_top.scatter(cx[near], cy[near], z[near], s=7.0, c=BBS_COLOR,
                               depthshade=False, edgecolors="black", linewidths=0.3)

    total = act_len * len(acts)
    animation = FuncAnimation(fig, draw, frames=total, interval=1000 // fps)
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
    print(f"   bl_ga   banding={ga['banding']}, not demeaned: fraction of the "
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


def _run_synthetic():
    """The generated warehouse, one pose, for --verify only.

    Kept because verify's expected numbers - band occupancies, the demeaning behaviour, the
    fitness a random pose already reaches - were all derived against this geometry. Pointing
    them at a real scenario would not check less, it would check nothing, since there is no
    reference value to compare against.
    """
    t0 = time.time()
    points, _ = build_warehouse()
    scan = simulate_scan(points, *TRUE_POSE)
    print(f"warehouse {len(points):,} points, scan {len(scan):,} returns "
          f"({time.time() - t0:.1f}s)")

    bands, band_weights = bl_bbs.build_slice_bands(float(points[:, 2].min()),
                                                    float(points[:, 2].max()))
    t0 = time.time()
    ga = run_ga(scan, cKDTree(points), ((0.0, LENGTH_M), (0.0, WIDTH_M)),
                 bl_ga.DEFAULT_SEED, TRUE_POSE, bands, band_weights, "none")
    print(f"bl_ga  {bl_ga.GENERATIONS} generations ({time.time() - t0:.1f}s)")

    t0 = time.time()
    bbs = run_bbs(scan, points, TRUE_POSE)
    print(f"bl_bbs {bbs['placements']:,} placements ({time.time() - t0:.1f}s)")
    return points, ga, bbs


def _run_scenario(scenario_id: str, map_points, tree, bounds, outcome: str,
                   hits: int):
    """Both searches on one released scenario, at bl_ga's shipped default seed."""
    truth = load_truth(scenario_id)
    scan = load_scan(scenario_id)
    bands, band_weights = bl_bbs.build_slice_bands(float(map_points[:, 2].min()),
                                                    float(map_points[:, 2].max()))

    t0 = time.time()
    ga = run_ga(scan, tree, bounds, bl_ga.DEFAULT_SEED,
                 truth, bands, band_weights, "per-band")
    print(f"  bl_ga  best pose {ga['error_m']:7.2f} m from truth ({time.time() - t0:.1f}s)")

    t0 = time.time()
    bbs = run_bbs(scan, map_points, truth)
    bx, by, _ = bbs["pose"]
    print(f"  bl_bbs best pose {np.hypot(bx - truth[0], by - truth[1]):7.2f} m from truth "
          f"({time.time() - t0:.1f}s)")

    return {"sid": scenario_id, "ga": ga, "bbs": bbs, "truth": truth,
            "outcome": outcome, "hits": hits, "range_ring": _range_ring(truth[0], truth[1])}


def main(argv=None):
    global LENGTH_M, WIDTH_M, HEIGHT_M

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "docs", "images",
                                                   "search_ga_vs_slices.gif"))
    ap.add_argument("--verify", action="store_true", help="print the numbers, render nothing")
    ap.add_argument("--frames", type=int, default=FRAMES,
                     help="override the frame count; for size smoke tests only, since the "
                          "shared clock is sampled at this rate")
    args = ap.parse_args(argv)

    if args.verify:
        return 0 if verify(*_run_synthetic()) else 1

    t0 = time.time()
    map_points, bounds, (LENGTH_M, WIDTH_M, HEIGHT_M) = load_map()
    # The hall's own extent replaces the synthetic constants, so every axis limit, BEV grid
    # and range ring below is the released map's rather than a stand-in for it.
    tree = cKDTree(map_points)
    shell_bands, _ = bl_bbs.build_slice_bands(float(map_points[:, 2].min()),
                                               float(map_points[:, 2].max()))
    shell = build_shell(map_points, shell_bands)
    print(f"map {len(map_points):,} points, {LENGTH_M:.1f} x {WIDTH_M:.1f} x {HEIGHT_M:.1f} m, "
          f"shell {len(shell):,} drawn ({time.time() - t0:.1f}s)")

    acts = []
    for sid, outcome, hits in ((SUCCESS_ID, "finds it", SUCCESS_HITS),
                                (FAILURE_ID, "aliases", FAILURE_HITS)):
        print(f"scenario {sid} ({outcome}):")
        acts.append(_run_scenario(sid, map_points, tree, bounds, outcome, hits))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    t0 = time.time()
    render_gif(args.out, acts, shell, frames=args.frames)
    size = os.path.getsize(args.out)
    # Pillow folds the identical hold frames into one long-duration frame, so the count
    # stored in the file is lower than the number rendered.
    with Image.open(args.out) as gif:
        stored = gif.n_frames
    rendered = (args.frames + HOLD_FRAMES) * len(acts)
    print(f"wrote {args.out}  {size / 1e6:.2f} MB, {rendered} rendered, "
          f"{stored} stored ({size / stored / 1024:.0f} KB/stored frame, "
          f"{time.time() - t0:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
