#!/usr/bin/env python3
"""Turns the prior map into a compact inline-SVG background for the report.

The report has to browse offline as a single HTML file, so the map cannot be
a raster and cannot be forty copies of itself. It is rasterized once to a
coarse occupancy grid, the occupied cells are run-length encoded into
horizontal bars, and the result is one <path> per height band that every
scenario view reuses.

Two bands are drawn so structure reads at a glance:

    rack band  0.6-6.5 m   the pallet racking, whose beam levels sit at
                           1.5 / 3.0 / 4.5 / 6.0 m
    high band  7.0-10.5 m  structural columns and the tall landmarks

Both bands avoid the floor (0-0.5 m) and the ceiling (11-12 m). Those are
continuous surfaces covering every cell, so a band touching either renders
the whole map solid and hides everything else. The bounds are read off the
map's own height histogram, not guessed.
"""

import json
import os

import numpy as np

RACK_BAND_M = (0.6, 6.5)
HIGH_BAND_M = (7.0, 10.5)
CELL_M = 0.4          # occupancy cell; coarse enough to stay small, fine
                      # enough that individual aisles remain separable
MIN_HITS = 2          # cells with a single stray return are noise, not structure


def _band_path(points, band, x0, y0, nx, ny):
    """Run-length encodes one height band's occupied cells into an SVG path.

    Cells are emitted in grid units; the caller scales them with a transform,
    so every coordinate stays a short integer. Merging horizontal runs into
    one rect each is what keeps this small: an aisle wall becomes a handful
    of bars instead of hundreds of squares.
    """
    z = points[:, 2]
    sel = points[(z >= band[0]) & (z < band[1])]
    if len(sel) == 0:
        return ""

    ix = np.clip(((sel[:, 0] - x0) / CELL_M).astype(np.int32), 0, nx - 1)
    iy = np.clip(((sel[:, 1] - y0) / CELL_M).astype(np.int32), 0, ny - 1)
    grid = np.zeros((nx, ny), dtype=np.int32)
    np.add.at(grid, (ix, iy), 1)
    occupied = grid >= MIN_HITS

    parts = []
    for j in range(ny):
        col = occupied[:, j]
        if not col.any():
            continue

        # Run boundaries: a padded diff marks where occupancy flips.
        edges = np.flatnonzero(np.diff(np.concatenate(([0], col.view(np.int8), [0]))))
        for start, end in zip(edges[0::2], edges[1::2]):
            parts.append(f"M{start} {j}h{end - start}v1h{start - end}z")

    return "".join(parts)


def build_map_render(map_path: str, cache_path: str = None):
    """Occupancy paths plus the world extent, cached because reading a 200 MB
    point cloud to draw the same background again is pure waste."""
    if cache_path and os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)

    import open3d as o3d
    points = np.asarray(o3d.io.read_point_cloud(map_path).points)

    x0, x1 = float(points[:, 0].min()), float(points[:, 0].max())
    y0, y1 = float(points[:, 1].min()), float(points[:, 1].max())
    nx = max(1, int(np.ceil((x1 - x0) / CELL_M)))
    ny = max(1, int(np.ceil((y1 - y0) / CELL_M)))

    render = {
        "x0": x0, "y0": y0, "cell_m": CELL_M, "nx": nx, "ny": ny,
        "extent_m": [x1 - x0, y1 - y0],
        "rack_path": _band_path(points, RACK_BAND_M, x0, y0, nx, ny),
        "high_path": _band_path(points, HIGH_BAND_M, x0, y0, nx, ny),
    }

    if cache_path:
        with open(cache_path, "w") as f:
            json.dump(render, f)
    return render


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    r = build_map_render(args.map, args.out)
    print(f"wrote {args.out}  grid {r['nx']}x{r['ny']}  "
          f"rack {len(r['rack_path'])} chars  high {len(r['high_path'])} chars")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
