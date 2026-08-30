"""Multi-slice 2D correlative BEV matching , the shared core behind bl_bbs
and bl_retrieval_gicp. Slice a point cloud into height bands, rasterize each
to an occupancy grid, and correlate two grids at every translation in one
shot via FFT cross-correlation (not a per-pixel loop) , this is what makes an
exhaustive SE(2) search over a whole warehouse tractable on a CPU.

Candidates are free to use, copy, or replace this module; nothing here reads
from the prior map itself.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class BevGrid:
    grid: np.ndarray
    origin_x: float
    origin_y: float
    resolution: float


def rasterize_slice(points: np.ndarray, origin_x: float, origin_y: float,
                     length: float, width: float, resolution: float,
                     z_lo: float, z_hi: float) -> BevGrid:
    nx = max(1, int(np.ceil(length / resolution)))
    ny = max(1, int(np.ceil(width / resolution)))
    grid = np.zeros((nx, ny), dtype=np.float32)

    if points.shape[0] == 0:
        return BevGrid(grid, origin_x, origin_y, resolution)

    mask = (points[:, 2] >= z_lo) & (points[:, 2] < z_hi)
    pts = points[mask]
    if pts.shape[0] == 0:
        return BevGrid(grid, origin_x, origin_y, resolution)

    ix = ((pts[:, 0] - origin_x) / resolution).astype(np.int64)
    iy = ((pts[:, 1] - origin_y) / resolution).astype(np.int64)
    valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    np.add.at(grid, (ix[valid], iy[valid]), 1.0)
    return BevGrid(np.minimum(grid, 1.0), origin_x, origin_y, resolution)


def rotate_points_2d(points: np.ndarray, yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    out = points.copy()
    out[:, 0] = c * points[:, 0] - s * points[:, 1]
    out[:, 1] = s * points[:, 0] + c * points[:, 1]
    return out


def correlate_translation_full(query_grid: np.ndarray, map_grid: np.ndarray):
    """Uncropped 2D cross-correlation of shape (mnx+qnx-1, mny+qny-1):
    result[p, q] is the score when query_grid's index (0, 0) is placed at
    map-relative index (p - (qnx - 1), q - (qny - 1)) , including negative
    map indices, which a naive "valid convolution" crop would silently drop.

    The map grid is demeaned before correlating (confirmed in testing: the
    floor and roof are near-continuous surfaces occupying ~97-98% of their
    height band almost everywhere, so raw dot-product correlation is
    dominated by trivial "floor matches floor" overlap everywhere, drowning
    out the sparse features , racks, lamps, HVAC , that actually carry
    localization signal; demeaning converts a near-uniformly-occupied band
    to near-zero everywhere while leaving genuinely structured bands intact).
    """
    mnx, mny = map_grid.shape
    qnx, qny = query_grid.shape
    pad_x, pad_y = mnx + qnx, mny + qny

    demeaned_map = map_grid - map_grid.mean()
    map_f = np.fft.fft2(demeaned_map, s=(pad_x, pad_y))
    query_f = np.fft.fft2(query_grid[::-1, ::-1], s=(pad_x, pad_y))
    corr = np.fft.ifft2(map_f * query_f).real
    return corr[:mnx + qnx - 1, :mny + qny - 1], (qnx, qny)


def reference_point_scores(full_corr: np.ndarray, qnx: int, qny: int, ref_i: int, ref_j: int,
                            mnx: int, mny: int) -> np.ndarray:
    """Re-indexes a correlate_translation_full output by the position of an
    arbitrary reference point (ref_i, ref_j) inside the query grid , e.g. a
    sensor centered in a query grid that also covers points behind/left of
    it , instead of the query's own (0, 0) corner.
    """
    p0 = (qnx - 1) - ref_i
    q0 = (qny - 1) - ref_j
    out = np.zeros((mnx, mny), dtype=full_corr.dtype)

    src_i_lo, src_i_hi = max(0, -p0), min(mnx, full_corr.shape[0] - p0)
    src_j_lo, src_j_hi = max(0, -q0), min(mny, full_corr.shape[1] - q0)
    if src_i_hi > src_i_lo and src_j_hi > src_j_lo:
        out[src_i_lo:src_i_hi, src_j_lo:src_j_hi] = \
            full_corr[p0 + src_i_lo:p0 + src_i_hi, q0 + src_j_lo:q0 + src_j_hi]
    return out


def match_scan_to_map(scan_local: np.ndarray, map_points: np.ndarray,
                       length: float, width: float, slice_bands, slice_weights=None,
                       resolution: float = 0.25, yaw_step_deg: float = 3.0,
                       query_half_extent_m: float = 75.0):
    """Exhaustive SE(2) correlative match. Returns (best_x, best_y, best_yaw,
    best_score, per_slice_scores_at_best) , per_slice_scores lets a caller
    print the "ceiling saves you" breakdown.
    """
    weights = slice_weights or [1.0] * len(slice_bands)
    map_grids = [rasterize_slice(map_points, 0.0, 0.0, length, width, resolution, z_lo, z_hi).grid
                 for z_lo, z_hi in slice_bands]
    nx, ny = map_grids[0].shape
    ref_px = int(round(query_half_extent_m / resolution))
    if ref_px >= min(nx, ny):
        raise ValueError(f"query_half_extent_m={query_half_extent_m} doesn't fit inside the map grid")

    yaws = np.arange(0.0, 2 * np.pi, np.radians(yaw_step_deg))
    best = (0.0, 0.0, 0.0, -np.inf, None)

    for yaw in yaws:
        rotated = rotate_points_2d(scan_local, yaw)
        per_slice = []
        combined = np.zeros((nx, ny), dtype=np.float32)
        for (z_lo, z_hi), map_grid, w in zip(slice_bands, map_grids, weights):
            query_grid = rasterize_slice(rotated, -query_half_extent_m, -query_half_extent_m,
                                          length=2 * query_half_extent_m, width=2 * query_half_extent_m,
                                          resolution=resolution, z_lo=z_lo, z_hi=z_hi).grid
            full_corr, (qnx, qny) = correlate_translation_full(query_grid, map_grid)
            slice_scores = reference_point_scores(full_corr, qnx, qny, ref_px, ref_px, nx, ny)
            per_slice.append(slice_scores)
            combined += w * slice_scores

        i, j = np.unravel_index(np.argmax(combined), combined.shape)
        score = float(combined[i, j])
        if score > best[3]:
            best = (i * resolution, j * resolution, float(yaw), score,
                    [float(s[i, j]) for s in per_slice])

    return best
