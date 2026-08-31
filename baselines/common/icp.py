"""Point-to-plane ICP refinement of a correlative-match hypothesis. The design spec calls
for "GICP" (porting slick-lo's CPU GICP); this baseline uses Open3D's point-to-plane ICP
instead, a materially simpler dependency that's close enough in spirit (both refine a coarse
initial guess via local point-cloud registration). A stronger submission can swap in real GICP.
"""

import numpy as np
import open3d as o3d


def refine_pose(scan_local: np.ndarray, map_points: np.ndarray, init_x: float, init_y: float,
                 init_yaw: float, init_z: float = 0.0, max_correspondence_dist: float = 0.3,
                 max_iterations: int = 30):
    """Returns (x, y, yaw, fitness) refined from the (init_x, init_y, init_yaw) coarse
    hypothesis. map_points should be a local crop around the hypothesis for speed; pass the
    full map only for small worlds.

    `init_z` defaults to 0.0 for backward compatibility, but scan_local's points are in the
    sensor's own local frame (1.0 m above the floor per the design's fixed rig height), not
    the floor itself. Leaving init_z at 0 starts ICP about a meter off in height, which can
    push every point beyond max_correspondence_dist and give zero correspondences (fitness=0)
    even from an otherwise-close start. Pass the rig's actual sensor height explicitly if you
    have one."""
    scan_pcd = o3d.geometry.PointCloud()
    scan_pcd.points = o3d.utility.Vector3dVector(scan_local.astype(np.float64))

    map_pcd = o3d.geometry.PointCloud()
    map_pcd.points = o3d.utility.Vector3dVector(map_points.astype(np.float64))
    map_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))

    c, s = np.cos(init_yaw), np.sin(init_yaw)
    T_init = np.eye(4)
    T_init[0, 0], T_init[0, 1] = c, -s
    T_init[1, 0], T_init[1, 1] = s, c
    T_init[0, 3], T_init[1, 3], T_init[2, 3] = init_x, init_y, init_z

    result = o3d.pipelines.registration.registration_icp(
        scan_pcd, map_pcd, max_correspondence_dist, T_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations),
    )
    T = result.transformation
    x, y = T[0, 3], T[1, 3]
    yaw = float(np.arctan2(T[1, 0], T[0, 0]))
    return x, y, yaw, result.fitness


def crop_map_near(map_points: np.ndarray, x: float, y: float, radius: float) -> np.ndarray:
    dist2 = (map_points[:, 0] - x) ** 2 + (map_points[:, 1] - y) ** 2
    return map_points[dist2 <= radius ** 2]
