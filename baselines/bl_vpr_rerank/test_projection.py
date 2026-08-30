import numpy as np

from projection import project_points, hypothesis_pose_matrix, remove_occluded


def _K(fx=100.0, fy=100.0, cx=50.0, cy=50.0):
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])


def test_point_on_optical_axis_projects_to_principal_point():
    # robot at map origin, camera coincides with base (identity extrinsic)
    T_map_base = np.eye(4)
    T_base_camera = np.eye(4)
    # 2 m directly ahead (REP103 link frame: +x = forward)
    points = np.array([[2.0, 0.0, 0.0]])
    u, v, z = project_points(points, T_map_base, T_base_camera, _K(), width=100, height=100)
    assert len(u) == 1
    np.testing.assert_allclose(u[0], 50.0, atol=1e-6)
    np.testing.assert_allclose(v[0], 50.0, atol=1e-6)
    np.testing.assert_allclose(z[0], 2.0, atol=1e-6)


def test_point_to_the_left_projects_to_left_side_of_image():
    T_map_base = np.eye(4)
    T_base_camera = np.eye(4)
    # 2 m ahead, 1 m to the left (+y in REP103 link frame is left)
    points = np.array([[2.0, 1.0, 0.0]])
    u, v, z = project_points(points, T_map_base, T_base_camera, _K(), width=100, height=100)
    assert len(u) == 1
    assert u[0] < 50.0  # left of the principal point
    np.testing.assert_allclose(v[0], 50.0, atol=1e-6)


def test_point_above_projects_to_top_of_image():
    T_map_base = np.eye(4)
    T_base_camera = np.eye(4)
    # 2 m ahead, 1 m up (+z in REP103 link frame is up)
    points = np.array([[2.0, 0.0, 1.0]])
    u, v, z = project_points(points, T_map_base, T_base_camera, _K(), width=100, height=100)
    assert len(u) == 1
    assert v[0] < 50.0  # above the principal point (image v grows downward)


def test_points_behind_camera_are_dropped():
    T_map_base = np.eye(4)
    T_base_camera = np.eye(4)
    points = np.array([[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    u, v, z = project_points(points, T_map_base, T_base_camera, _K(), width=100, height=100)
    assert len(u) == 1  # only the point in front survives


def test_points_outside_fov_are_dropped():
    T_map_base = np.eye(4)
    T_base_camera = np.eye(4)
    # far to the side: projects way outside a 100px image
    points = np.array([[1.0, 100.0, 0.0]])
    u, v, z = project_points(points, T_map_base, T_base_camera, _K(), width=100, height=100)
    assert len(u) == 0


def test_hypothesis_pose_translates_and_rotates_correctly():
    T = hypothesis_pose_matrix(x=5.0, y=3.0, yaw=np.pi / 2)
    np.testing.assert_allclose(T[:3, 3], [5.0, 3.0, 1.0])
    # yaw=90deg: local +x (forward) maps to world +y
    forward_world = T[:3, :3] @ np.array([1.0, 0.0, 0.0])
    np.testing.assert_allclose(forward_world, [0.0, 1.0, 0.0], atol=1e-9)


def test_remove_occluded_keeps_only_nearest_point_per_bucket():
    # two points landing in the same coarse bucket, one closer than the other
    u = np.array([10.0, 11.0, 90.0])
    v = np.array([10.0, 11.0, 90.0])
    depth = np.array([5.0, 2.0, 8.0])
    u2, v2, d2 = remove_occluded(u, v, depth, width=100, height=100, bucket_px=4)
    assert len(d2) == 2  # the far bucket collapses the two near-duplicate points to 1
    assert 2.0 in d2 and 8.0 in d2
    assert 5.0 not in d2  # farther point in the same bucket is discarded


def test_remove_occluded_empty_input():
    u, v, d = remove_occluded(np.array([]), np.array([]), np.array([]), width=100, height=100)
    assert len(u) == 0


def test_map_frame_offset_is_correctly_removed():
    # robot sits at map (10, 0), facing +x (yaw=0); a map point at (12, 0, 0)
    # is 2 m directly ahead in the robot's own frame
    T_map_base = hypothesis_pose_matrix(x=10.0, y=0.0, yaw=0.0, z=0.0)
    T_base_camera = np.eye(4)
    points = np.array([[12.0, 0.0, 0.0]])
    u, v, z = project_points(points, T_map_base, T_base_camera, _K(), width=100, height=100)
    assert len(u) == 1
    np.testing.assert_allclose(u[0], 50.0, atol=1e-6)
    np.testing.assert_allclose(z[0], 2.0, atol=1e-6)
