"""Projects prior-map points into the scenario camera at a hypothesis pose.

Frame chain: map -> base (hypothesis pose, yaw-only) -> camera link (fixed
extrinsic from calib.json, REP103-style: x-forward, y-left, z-up) -> optical
frame (x-right, y-down, z-forward, the convention camera_info.json's K
matrix is defined in) -> pixel, via the standard pinhole model.
"""

import numpy as np

# fixed link-frame -> optical-frame rotation (REP103 forward/left/up ->
# OpenCV/ROS optical right/down/forward): x_opt = -y_link, y_opt = -z_link,
# z_opt = x_link
R_OPTICAL_FROM_LINK = np.array([
    [0.0, -1.0, 0.0],
    [0.0, 0.0, -1.0],
    [1.0, 0.0, 0.0],
])


def hypothesis_pose_matrix(x: float, y: float, yaw: float, z: float = 1.0) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    T = np.eye(4)
    T[0, 0], T[0, 1] = c, -s
    T[1, 0], T[1, 1] = s, c
    T[0, 3], T[1, 3], T[2, 3] = x, y, z
    return T


def project_points(map_points: np.ndarray, T_map_base: np.ndarray, T_base_camera: np.ndarray,
                    K: np.ndarray, width: int, height: int):
    """Returns (u, v, depth) for map points landing inside the image with positive depth;
    everything else is dropped."""
    T_map_camera = T_map_base @ T_base_camera
    T_camera_map = np.linalg.inv(T_map_camera)

    pts_h = np.concatenate([map_points, np.ones((map_points.shape[0], 1))], axis=1)
    pts_cam_link = (T_camera_map @ pts_h.T).T[:, :3]

    pts_opt = pts_cam_link @ R_OPTICAL_FROM_LINK.T

    z = pts_opt[:, 2]
    in_front = z > 0.05
    pts_opt, z = pts_opt[in_front], z[in_front]

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u = fx * pts_opt[:, 0] / z + cx
    v = fy * pts_opt[:, 1] / z + cy

    in_bounds = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return u[in_bounds], v[in_bounds], z[in_bounds]


def remove_occluded(u: np.ndarray, v: np.ndarray, depth: np.ndarray, width: int, height: int,
                     bucket_px: int = 4):
    """Keeps only the nearest point per coarse pixel bucket (a simple z-buffer / hidden-point
    removal). Without this, points behind a rack wall still get drawn, so a 180-degree-flipped
    hypothesis produces a similar silhouette to the true pose; the flip outscored GT in 2 of 3
    real scenarios before this."""
    if len(u) == 0:
        return u, v, depth

    bu = (u // bucket_px).astype(np.int64)
    bv = (v // bucket_px).astype(np.int64)
    nbx = int(np.ceil(width / bucket_px)) + 1
    bucket_id = bv * nbx + bu

    order = np.argsort(depth)
    _, first_idx = np.unique(bucket_id[order], return_index=True)
    keep = order[first_idx]
    return u[keep], v[keep], depth[keep]


def K_from_camera_info(camera_info: dict) -> np.ndarray:
    return np.array(camera_info["K"]).reshape(3, 3)


def T_base_camera_from_calib(calib: dict) -> np.ndarray:
    return np.array(calib["T_base_camera"])
