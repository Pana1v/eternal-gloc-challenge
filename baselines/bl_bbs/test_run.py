import pytest
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


# --- sensor-frame vs map-frame band alignment -------------------------------
# The scan arrives in the SENSOR frame: the lidar is the origin, so the floor
# sits at z = -SENSOR_HEIGHT_M and the ceiling at ceiling_height - SENSOR_HEIGHT_M.
# The slice bands are derived from the MAP's world-frame z extent. Slicing the
# scan with those bands compares layers one sensor height apart, which put the
# 2x-weighted ceiling bands on the wrong physical structure entirely.

def _synthetic_pair(sensor_height=1.0, ceiling=12.0):
    """A map with a floor and a ceiling, and the scan the same rig would see."""
    xy = np.random.default_rng(0).uniform(0, 20, size=(4000, 2))
    floor = np.column_stack([xy, np.zeros(len(xy))])
    roof = np.column_stack([xy, np.full(len(xy), ceiling)])
    map_points = np.vstack([floor, roof])

    scan = map_points.copy()
    scan[:, 2] -= sensor_height          # into the sensor frame
    return map_points, scan


def test_scan_shares_the_maps_z_frame_before_slicing():
    """The invariant the bands rely on: once lifted, the scan's floor and
    ceiling land on the map's, so a band selects the same physical layer from
    both. Without the lift every band is off by exactly the mount height.
    """
    from run import SENSOR_HEIGHT_M

    map_points, scan = _synthetic_pair(SENSOR_HEIGHT_M)

    raw_offset = abs(scan[:, 2].min() - map_points[:, 2].min())
    assert raw_offset == pytest.approx(SENSOR_HEIGHT_M), "premise: the raw scan is offset"

    lifted = scan.copy()
    lifted[:, 2] += SENSOR_HEIGHT_M
    assert lifted[:, 2].min() == pytest.approx(map_points[:, 2].min())
    assert lifted[:, 2].max() == pytest.approx(map_points[:, 2].max())


def test_run_scenario_recovers_a_known_pose(tmp_path):
    """Drives the real run_scenario path with a sensor-frame scan on disk.
    A test that lifts the scan itself would pass either way, so this writes
    the scan exactly as a capture does and lets run_scenario handle it.

    Layer heights matter here. Bands over a 0-12 m map are
    [0,1) [1,3.5) [3.5,6.5) [6.5,9.5) [9.5,12), so a layer at 3.0 shifted down
    to 2.0 stays inside [1,3.5) and still matches by accident. The heights
    below are chosen so every layer crosses a band edge when shifted: 1.5->0.5,
    4->3, 7->6, 10->9. Each layer also carries its own pattern, so landing in
    the wrong band means correlating against structure that is genuinely
    different rather than a statistical twin.
    """
    import open3d as o3d
    from run import build_slice_bands, run_scenario, SENSOR_HEIGHT_M

    rng = np.random.default_rng(1)
    layers = []
    for z in (1.5, 4.0, 7.0, 10.0):
        n = 2500
        layers.append(np.column_stack([
            rng.uniform(0, 40, n), rng.uniform(0, 30, n), np.full(n, z)]))
    # pin the z extent so build_slice_bands sees a 0-12 m map
    layers.append(np.array([[0.0, 0.0, 0.0], [40.0, 30.0, 12.0]]))
    map_points = np.vstack(layers)

    true_x, true_y = 18.0, 13.0
    scan = map_points - np.array([true_x, true_y, SENSOR_HEIGHT_M])

    d = tmp_path / "000000"
    d.mkdir()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(scan)
    o3d.io.write_point_cloud(str(d / "lidar.pcd"), pcd)

    bands, weights = build_slice_bands(map_points[:, 2].min(), map_points[:, 2].max())
    x, y, _yaw, _score, _fit, _per = run_scenario(
        str(d), map_points, 40.0, 30.0, bands, weights, query_half_extent_m=12.0)

    assert abs(x - true_x) < 1.0 and abs(y - true_y) < 1.0, \
        f"localised at ({x:.2f}, {y:.2f}), expected ({true_x}, {true_y})"
