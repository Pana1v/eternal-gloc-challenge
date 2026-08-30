import numpy as np

from descriptor import build_polar_histogram, ring_key, ring_key_distance, estimate_yaw_shift


def _asymmetric_points(rng):
    """Points at several distinct angles/radii, deliberately not rotationally
    symmetric, so yaw estimation has a real, non-degenerate answer."""
    pts = []
    for angle_deg, radius in [(20, 3.0), (95, 6.0), (200, 4.5), (260, 8.0), (310, 2.0)]:
        a = np.radians(angle_deg)
        center = np.array([radius * np.cos(a), radius * np.sin(a)])
        cluster = rng.normal(0, 0.15, size=(30, 2)) + center
        pts.append(cluster)
    return np.concatenate(pts, axis=0)


def test_build_polar_histogram_bins_a_known_point():
    points = np.array([[5.0, 0.0]])  # angle=0, radius=5
    ph = build_polar_histogram(points, n_angle_bins=36, n_radius_bins=10, max_radius=10.0)
    angle_idx = int((0 + np.pi) / (2 * np.pi) * 36) % 36
    radius_idx = int(5.0 / 10.0 * 10)
    assert ph.hist[angle_idx, radius_idx] == 1.0
    assert ph.hist.sum() == 1.0


def test_points_beyond_max_radius_are_excluded():
    points = np.array([[5.0, 0.0], [50.0, 0.0]])
    ph = build_polar_histogram(points, n_angle_bins=36, n_radius_bins=10, max_radius=10.0)
    assert ph.hist.sum() == 1.0


def test_ring_key_is_rotation_invariant():
    rng = np.random.default_rng(0)
    points = _asymmetric_points(rng)
    ph_a = build_polar_histogram(points, n_angle_bins=60, n_radius_bins=20, max_radius=15.0)

    yaw = np.radians(73.0)
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    rotated = points @ R.T
    ph_b = build_polar_histogram(rotated, n_angle_bins=60, n_radius_bins=20, max_radius=15.0)

    key_a, key_b = ring_key(ph_a), ring_key(ph_b)
    # not bit-identical (discretization moves some points across radius bins
    # under rotation) but should be close relative to two unrelated scans
    assert ring_key_distance(key_a, key_b) < 0.15


def test_ring_key_distinguishes_different_places():
    rng = np.random.default_rng(1)
    points_a = _asymmetric_points(rng)
    points_b = rng.normal(0, 4.0, size=(150, 2))  # unrelated random scatter

    ph_a = build_polar_histogram(points_a, n_angle_bins=60, n_radius_bins=20, max_radius=15.0)
    ph_b = build_polar_histogram(points_b, n_angle_bins=60, n_radius_bins=20, max_radius=15.0)

    same_place_dist = ring_key_distance(ring_key(ph_a), ring_key(ph_a))
    different_place_dist = ring_key_distance(ring_key(ph_a), ring_key(ph_b))
    assert same_place_dist == 0.0
    assert different_place_dist > 0.1


def test_estimate_yaw_shift_recovers_known_rotation():
    rng = np.random.default_rng(2)
    points = _asymmetric_points(rng)
    query_ph = build_polar_histogram(points, n_angle_bins=72, n_radius_bins=20, max_radius=15.0)

    true_yaw = np.radians(40.0)
    c, s = np.cos(true_yaw), np.sin(true_yaw)
    R = np.array([[c, -s], [s, c]])
    rotated = points @ R.T
    candidate_ph = build_polar_histogram(rotated, n_angle_bins=72, n_radius_bins=20, max_radius=15.0)

    # rotating the query by true_yaw should align it onto the candidate
    est_yaw, score = estimate_yaw_shift(query_ph, candidate_ph)
    err_deg = np.degrees(abs(((est_yaw - true_yaw + np.pi) % (2 * np.pi)) - np.pi))
    assert err_deg < 10.0  # within ~2 angle bins (72 bins -> 5 deg/bin)
