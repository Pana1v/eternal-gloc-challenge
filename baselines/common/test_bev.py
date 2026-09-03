import numpy as np

from bev import (
    rasterize_slice, rotate_points_2d, correlate_translation_full,
    reference_point_scores, match_scan_to_map,
)


def _cluster(cx, cy, cz, n=200, spread=0.3, *, rng):
    """Elongated blob plus an off-center satellite marker, so it has no rotational symmetry
    (an isotropic or merely-elongated blob would be ambiguous under rotation)."""
    main = rng.normal(0, 1.0, size=(n, 3)) * np.array([spread * 4, spread * 0.5, spread])
    satellite = rng.normal(0, spread * 0.3, size=(n // 4, 3)) + np.array([spread * 3, spread * 2, 0.0])
    pts = np.concatenate([main, satellite], axis=0)
    pts += np.array([cx, cy, cz])
    return pts


def test_rasterize_slice_places_points_in_correct_cells():
    points = np.array([[1.0, 1.0, 0.5], [3.0, 3.0, 0.5], [1.0, 1.0, 5.0]])
    bev = rasterize_slice(points, origin_x=0.0, origin_y=0.0, length=10.0, width=10.0,
                           resolution=1.0, z_lo=0.0, z_hi=1.0)
    assert bev.grid[1, 1] == 1.0
    assert bev.grid[3, 3] == 1.0
    assert bev.grid.sum() == 2.0


def test_rotate_points_2d_90_degrees():
    points = np.array([[1.0, 0.0, 0.0]])
    rotated = rotate_points_2d(points, yaw=np.pi / 2)
    np.testing.assert_allclose(rotated[0, :2], [0.0, 1.0], atol=1e-9)


def test_reference_point_scores_finds_feature_near_map_origin():
    query_grid = np.zeros((11, 11), dtype=np.float32)
    ref_i, ref_j = 5, 5
    query_grid[ref_i, ref_j] = 1.0

    map_grid = np.zeros((20, 20), dtype=np.float32)
    map_grid[2, 2] = 1.0

    full_corr, (qnx, qny) = correlate_translation_full(query_grid, map_grid)
    sensor_scores = reference_point_scores(full_corr, qnx, qny, ref_i, ref_j, 20, 20)

    i, j = np.unravel_index(np.argmax(sensor_scores), sensor_scores.shape)
    assert (i, j) == (2, 2)


def test_match_scan_to_map_recovers_a_known_pose():
    """Place a distinctive feature in the map at a known pose, take the
    exact same feature as a (mean-centered) local scan, and confirm the
    exhaustive search recovers the pose it was generated from."""
    rng = np.random.default_rng(0)
    true_x, true_y, true_yaw = 12.0, 7.0, np.radians(35.0)
    feature = _cluster(0.0, 0.0, 0.5, rng=rng)  # local-frame feature, at sensor origin

    c, s = np.cos(true_yaw), np.sin(true_yaw)
    R = np.array([[c, -s], [s, c]])
    world_xy = feature[:, :2] @ R.T + np.array([true_x, true_y])
    map_points = np.concatenate([world_xy, feature[:, 2:3]], axis=1)

    x, y, yaw, score, per_slice = match_scan_to_map(
        feature, map_points, length=40.0, width=40.0, slice_bands=[(0.0, 1.0)],
        resolution=0.25, yaw_step_deg=5.0, query_half_extent_m=5.0,
    )

    assert abs(x - true_x) < 0.5
    assert abs(y - true_y) < 0.5
    yaw_err_deg = np.degrees(abs(((yaw - true_yaw + np.pi) % (2 * np.pi)) - np.pi))
    assert yaw_err_deg < 10.0
