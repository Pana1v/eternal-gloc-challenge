import numpy as np

from run import build_slice_bands, run_scenario


def _cluster(cx, cy, cz, n=200, spread=0.3, rng=None):
    """Elongated blob plus an off-center satellite marker, so the shape has no rotational
    symmetry (an isotropic or merely-elongated blob would leave the pose ambiguous)."""
    rng = rng or np.random.default_rng(0)
    main = rng.normal(0, 1.0, size=(n, 3)) * np.array([spread * 4, spread * 0.5, spread])
    satellite = rng.normal(0, spread * 0.3, size=(n // 4, 3)) + np.array([spread * 3, spread * 2, 0.0])
    pts = np.concatenate([main, satellite], axis=0)
    pts += np.array([cx, cy, cz])
    return pts


def test_build_slice_bands_weights_ceiling_higher():
    bands, weights = build_slice_bands(z_min=0.0, z_max=12.0)
    assert len(bands) == len(weights) == 5
    assert weights[-1] > weights[0]  # ceiling band weighted more than floor band


def test_run_scenario_recovers_known_pose_with_icp_refinement(tmp_path):
    """Regression guard: ICP must init at the rig's real sensor height, not z=0, or it
    can find zero correspondences even from an otherwise-close coarse guess."""
    import open3d as o3d

    rng = np.random.default_rng(0)
    true_x, true_y, true_yaw = 15.0, 8.0, np.radians(40.0)
    sensor_height = 1.0
    feature_low = _cluster(0.0, 0.0, 0.5 - sensor_height, rng=rng)   # local frame: sensor at z=0
    feature_high = _cluster(0.0, 0.0, 10.5 - sensor_height, rng=rng)
    scan_local = np.concatenate([feature_low, feature_high], axis=0)

    c, s = np.cos(true_yaw), np.sin(true_yaw)
    R = np.array([[c, -s], [s, c]])
    world_low = feature_low[:, :2] @ R.T + np.array([true_x, true_y])
    world_high = feature_high[:, :2] @ R.T + np.array([true_x, true_y])
    map_points = np.concatenate([
        np.concatenate([world_low, feature_low[:, 2:3] + sensor_height], axis=1),
        np.concatenate([world_high, feature_high[:, 2:3] + sensor_height], axis=1),
    ], axis=0)

    scenario_dir = tmp_path / "000000"
    scenario_dir.mkdir()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(scan_local.astype(np.float64))
    o3d.io.write_point_cloud(str(scenario_dir / "lidar.pcd"), pcd)

    bands, weights = build_slice_bands(z_min=0.0, z_max=12.0)
    x, y, yaw, score, fitness, per_slice = run_scenario(
        str(scenario_dir), map_points, length=40.0, width=40.0,
        slice_bands=bands, slice_weights=weights, query_half_extent_m=15.0,
    )

    assert fitness > 0.0, "ICP found zero correspondences, likely the init_z regression"
    assert abs(x - true_x) < 0.5
    assert abs(y - true_y) < 0.5
