#!/usr/bin/env bash
# Drives every tool in eternal-gloc-challenge against a synthetic fixture.
# Self-contained: run it from the repo root, it needs no other file.
#
#   FIXTURE=/tmp/gloc-fixture ./run_all.sh
#
# Point FIXTURE outside the repo: .fixture/ and results/ are not gitignored.
set -euo pipefail

FIXTURE="${FIXTURE:-/tmp/gloc-fixture}"
IMAGE=eternal-gloc-runtime

mkdir -p "$FIXTURE"

# The image has no COPY: it is a runtime environment only, so the repo and
# the fixture both arrive by bind mount. --user keeps outputs owned by you
# rather than root.
dr() {
    docker run --rm --user "$(id -u):$(id -g)" \
        -v "$PWD:/workspace" -v "$FIXTURE:/fixture" \
        -w /workspace "$IMAGE" "$@"
}

cat > "$FIXTURE/make_fixture.py" <<'PYEOF'
"""Synthetic fixture: same geometry as ci/smoke_test.sh, plus the
camera_info.json / calib.json that bl_vpr_rerank needs and CI omits.
Sensor values follow docs/SENSORS.md.

The blobs are isotropic, so the fixture carries NO yaw information. It
exercises the plumbing, it does not measure localization quality.
"""
import json
import os

import numpy as np
import open3d as o3d
from PIL import Image

OUT, SCEN = "/fixture", "/fixture/scenarios/000000"
os.makedirs(SCEN, exist_ok=True)

WIDTH, HEIGHT, HFOV_DEG = 1280, 800, 90.0
CAM_FORWARD_M, CAM_PITCH_DEG = 0.10, 10.0

rng = np.random.default_rng(0)

# map: rack-level blob at z=0.5 and a ceiling blob at z=10.5, both at (5, 5)
map_pts = np.concatenate([
    rng.normal([5, 5, 0.5], 0.3, size=(300, 3)),
    rng.normal([5, 5, 10.5], 0.3, size=(100, 3)),
], axis=0)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(map_pts)
o3d.io.write_point_cloud(f"{OUT}/prior_map.pcd", pcd)

# scan: the same structure in the rig's own frame, so centred on the origin
scan_pts = np.concatenate([
    rng.normal([0, 0, 0.5], 0.3, size=(300, 3)),
    rng.normal([0, 0, 10.0], 0.3, size=(100, 3)),
], axis=0)
spcd = o3d.geometry.PointCloud()
spcd.points = o3d.utility.Vector3dVector(scan_pts)
o3d.io.write_point_cloud(f"{SCEN}/lidar.pcd", spcd)

Image.fromarray(rng.integers(60, 90, size=(HEIGHT, WIDTH, 3)).astype("uint8")).save(f"{SCEN}/camera.png")

# pinhole intrinsics from the horizontal FOV: fx = (w/2) / tan(hfov/2)
fx = (WIDTH / 2) / np.tan(np.deg2rad(HFOV_DEG) / 2)
with open(f"{SCEN}/camera_info.json", "w") as f:
    json.dump({
        "width": WIDTH, "height": HEIGHT,
        "K": [fx, 0.0, WIDTH / 2, 0.0, fx, HEIGHT / 2, 0.0, 0.0, 1.0],
        "distortion": [0.0] * 5,
    }, f, indent=2)

# camera sits CAM_FORWARD_M ahead of the lidar, pitched CAM_PITCH_DEG up
p = np.deg2rad(CAM_PITCH_DEG)
T_base_camera = np.eye(4)
T_base_camera[:3, :3] = [[np.cos(-p), 0, np.sin(-p)], [0, 1, 0], [-np.sin(-p), 0, np.cos(-p)]]
T_base_camera[:3, 3] = [CAM_FORWARD_M, 0.0, 0.0]
with open(f"{SCEN}/calib.json", "w") as f:
    json.dump({
        "T_base_lidar": np.eye(4).tolist(),
        "T_base_camera": T_base_camera.tolist(),
        "gravity_in_base": [0.0, 0.0, -9.81],
        "lidar": {"beams": 32, "horizontal_samples": 1800, "vertical_min_deg": -15.0,
                   "vertical_max_deg": 45.0, "range_min_m": 0.5, "range_max_m": 70.0},
        "camera": {"width_px": WIDTH, "height_px": HEIGHT, "hfov_deg": HFOV_DEG},
    }, f, indent=2)

with open(f"{SCEN}/meta.json", "w") as f:
    json.dump({"scenario_id": "000000", "track": "A", "world_id": "smoke"}, f)

with open(f"{OUT}/gt.txt", "w") as f:
    f.write("000000 1 0 0 5.0 0 1 0 5.0 0 0 1 0.5\n")

print("fixture written to", OUT)
PYEOF

echo "== fixture =="
dr python3.12 /fixture/make_fixture.py

echo "== B1 bl_bbs =="
dr python3.12 baselines/bl_bbs/run.py \
    --scenarios /fixture/scenarios --map /fixture/prior_map.pcd --out /fixture/sub_bbs.txt

echo "== B2 bl_retrieval_gicp =="
dr python3.12 baselines/bl_retrieval_gicp/run.py \
    --scenarios /fixture/scenarios --map /fixture/prior_map.pcd --out /fixture/sub_ret.txt

echo "== B3 bl_vpr_rerank (re-ranks B1's hypotheses) =="
dr python3.12 baselines/bl_vpr_rerank/run.py \
    --scenarios /fixture/scenarios --map /fixture/prior_map.pcd \
    --hypotheses /fixture/sub_bbs.txt --out /fixture/sub_rerank.txt

echo "== score =="
for m in bbs ret rerank; do
    dr python3.12 eval/score.py --submission "/fixture/sub_$m.txt" --gt /fixture/gt.txt \
        --track A --split dev --method "$m" --out-dir /fixture/results
done

echo "== plots =="
for s in "$FIXTURE"/results/*/stats.csv; do
    [ -e "$s" ] || continue
    dr python3.12 eval/plot_results.py --stats "/fixture/results/$(basename "$(dirname "$s")")/stats.csv"
done

echo "== done: outputs in $FIXTURE =="
