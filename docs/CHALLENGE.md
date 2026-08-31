# The Challenge

## Setup

A warehouse robot needs to localize in a large, deliberately repetitive
warehouse: no GPS, no prior pose estimate. You're given a **prior 3D lidar
map** of the building (geometry only, no textures, no color) and one or
more **scenarios**, each a snapshot (or short sequence) of what the robot's
sensors see at an unknown pose. Your job: recover that pose in the map's
frame.

The bottom half of the warehouse (rack level) is intentionally close to
featureless and repetitive: many aisle cells look nearly identical from
lidar alone. The ceiling (trusses, lamp grid, HVAC runs, skylights) repeats
less often, and the camera sees things lidar geometry can't (signage,
posters, floor markings). You need both.

## Track A: single-shot

Each scenario directory contains:

```
000000/
├── lidar.pcd          # one 360-degree sweep, sensor-local frame
├── camera.png          # 1280x800 RGB, forward-facing
├── camera_info.json    # pinhole intrinsics (K, distortion)
├── calib.json          # T_base_lidar, T_base_camera, gravity_in_base, rig spec
└── meta.json            # scenario_id, track, world_id
```

Everything is expressed in the scenario's own local frame; nothing ties it
to the map or to any other scenario. Answer: the rig's pose (`T_map_base`)
at the scan instant.

## Track B: motion budget

Each scenario is a short (~20 m, 40-step) driven sequence:

```
000000/
├── steps/000/ {lidar.pcd, camera.png}
├── steps/001/ {lidar.pcd, camera.png}
├── ...
├── odometry.txt   # KITTI 3x4 poses, relative to step 0, realistically noisy
├── imu.csv         # 200 Hz: t_rel,ax,ay,az,gx,gy,gz
├── camera_info.json / calib.json / meta.json
```

You may read as much or as little of the sequence as you need. Answer: the
pose of **step 0** plus your declared `steps_used` (the step index you
stopped at; reading further costs you on the budget term, see Scoring).

## Submission format

One line per (scenario, hypothesis):

```
<scenario_id> <k> <w> [<steps_used>] r11 r12 r13 tx r21 r22 r23 ty r31 r32 r33 tz
```

- `k` in `{0, 1, 2}`: up to 3 pose hypotheses per scenario, most-confident first.
- `w`: confidence weight for this hypothesis; weights for one scenario must sum to `<= 1`.
- `steps_used`: Track B only, immediately after `w`.
- The 12 trailing values are the KITTI 3x4 row-major pose matrix (`T_map_base`).

You don't have to submit 3 hypotheses; one is fine. Hedging with multiple
hypotheses is scored fairly (see below): it pays off exactly when you're
genuinely unsure, and costs you when you're not.

## Scoring

For a submitted pose `T_hat` against ground truth `T`:

- `e_t = ||trans(T^-1 T_hat)||` (meters)
- `e_r` = geodesic rotation angle (degrees)

**Success thresholds**: `S-fine` = `e_t <= 0.5 m` and `e_r <= 5 deg` (tight
enough that a local scan-matcher will converge from here). `S-coarse` =
`e_t <= 2.0 m` and `e_r <= 10 deg` (right aisle, wrong meters).

**Per-scenario bounded loss**:

```
L = 0.7 * min(e_t / 2.0, 1) + 0.3 * min(e_r / 20 deg, 1)        (Track A, in [0, 1])
L_B = 0.75 * L + 0.25 * (steps_used / 40)                         (Track B)
```

A scenario with no submitted answer scores `L = 1`.

**Multi-hypothesis**: with weights `w_k` summing to `<= 1`,
`L = sum_k(w_k * L(T_hat_k)) + (1 - sum_k w_k) * 1`. You're paid for
probability mass placed on good poses. Hedging across genuinely ambiguous
aliases caps your best-case loss instead of gambling on one guess.

**Headline score**: `Score = 100 * (1 - mean(L))` over the eval set,
reported per track (and, where available, per difficulty tier). SR@fine
and SR@coarse are also reported.

## Time-box and grading

Suggested effort: **2-4 focused days**. A strong submission improves on one
baseline in one track; you are not expected to max out both tracks.

- 40% eval score
- 30% method write-up (<= 2 pages: what you tried, why, what you'd try next)
- 20% code quality
- 10% experimental hygiene (ablations on the dev set, sane defaults)
