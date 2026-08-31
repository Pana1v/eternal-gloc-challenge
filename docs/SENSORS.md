# Sensor Rig

One 360-degree lidar + one forward camera, rigidly mounted together.

## Lidar

| Parameter | Value |
|---|---|
| Beams | 32 |
| Horizontal samples | 1800 (0.2 deg resolution, full 360 deg) |
| Vertical FOV | -15 deg to +45 deg (asymmetric upward; reaches the ceiling from any drivable pose) |
| Range | 0.5 m to 70 m |
| Range noise | Gaussian, sigma = 0.02 m |

`lidar.pcd` is XYZ only, in the sensor's own local frame (no world/map
alignment implied).

## Camera

| Parameter | Value |
|---|---|
| Resolution | 1280 x 800 |
| Horizontal FOV | 90 deg |
| Mount pitch | +10 deg up from horizontal (sees both the aisle and some ceiling) |
| Pixel format | RGB |

`camera_info.json` carries the pinhole intrinsics matrix `K` (row-major
3x3) and distortion coefficients (zeros; the sensor is undistorted).

## Rig calibration (`calib.json`)

```json
{
  "T_base_lidar": [[4x4 identity]],
  "T_base_camera": [[4x4: translation + pitch rotation from lidar to camera]],
  "gravity_in_base": [0.0, 0.0, -9.81],
  "lidar": { "beams": 32, "horizontal_samples": 1800, "vertical_min_deg": -15.0,
             "vertical_max_deg": 45.0, "range_min_m": 0.5, "range_max_m": 70.0 },
  "camera": { "width_px": 1280, "height_px": 800, "hfov_deg": 90.0 }
}
```

`T_base_lidar` is always identity; the lidar's own frame **is** the rig's
"base" frame. `T_base_camera` places the camera ~0.10 m forward of the
lidar, pitched +10 deg up. `gravity_in_base` tells you which way is down in
the rig's own frame at capture time. A real AMR knows this from its own
IMU/leveling, so it's given rather than something you need to estimate from
a single Track A snapshot.

Track B additionally ships `imu.csv` (200 Hz: `t_rel,ax,ay,az,gx,gy,gz`,
accelerations in m/s^2, angular rates in rad/s) alongside the noisy
`odometry.txt`; fusing the two should beat wheel odometry alone.
