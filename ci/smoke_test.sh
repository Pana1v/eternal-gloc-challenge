#!/usr/bin/env bash
# CI smoke test: builds the runtime image, constructs a tiny synthetic
# fixture, runs bl_bbs against it, scores the result, and verifies nothing crashed.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FIXTURE_DIR="$(mktemp -d)"
trap 'rm -rf "$FIXTURE_DIR"' EXIT
DOCKER_USER="$(id -u):$(id -g)"  # avoid root-owned output files from container runs

echo "== building runtime image =="
docker build -f "$REPO_ROOT/docker/runtime.Dockerfile" -t eternal-gloc-runtime "$REPO_ROOT"

echo "== writing synthetic fixture =="
mkdir -p "$FIXTURE_DIR/scenarios/000000"
docker run --rm --user "$DOCKER_USER" -v "$FIXTURE_DIR:/fixture" eternal-gloc-runtime python3.12 -c "
import json
import numpy as np
import open3d as o3d
from PIL import Image

rng = np.random.default_rng(0)
map_pts = np.concatenate([
    rng.normal([5, 5, 0.5], 0.3, size=(300, 3)),
    rng.normal([5, 5, 10.5], 0.3, size=(100, 3)),
], axis=0)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(map_pts)
o3d.io.write_point_cloud('/fixture/prior_map.pcd', pcd)

scan_pts = np.concatenate([
    rng.normal([0, 0, 0.5], 0.3, size=(300, 3)),
    rng.normal([0, 0, 10.0], 0.3, size=(100, 3)),
], axis=0)
spcd = o3d.geometry.PointCloud()
spcd.points = o3d.utility.Vector3dVector(scan_pts)
o3d.io.write_point_cloud('/fixture/scenarios/000000/lidar.pcd', spcd)

img = rng.integers(60, 90, size=(64, 64, 3)).astype('uint8')
Image.fromarray(img).save('/fixture/scenarios/000000/camera.png')

with open('/fixture/scenarios/000000/meta.json', 'w') as f:
    json.dump({'scenario_id': '000000', 'track': 'A', 'world_id': 'smoke'}, f)
with open('/fixture/gt.txt', 'w') as f:
    f.write('000000 1 0 0 5.0 0 1 0 5.0 0 0 1 0.5\n')
print('fixture written')
"

echo "== running bl_bbs =="
docker run --rm --user "$DOCKER_USER" -v "$REPO_ROOT:/workspace" -v "$FIXTURE_DIR:/fixture" -w /workspace \
    eternal-gloc-runtime python3.12 baselines/bl_bbs/run.py \
    --scenarios /fixture/scenarios --map /fixture/prior_map.pcd --out /fixture/submission.txt

echo "== scoring =="
docker run --rm --user "$DOCKER_USER" -v "$REPO_ROOT:/workspace" -v "$FIXTURE_DIR:/fixture" -w /workspace \
    eternal-gloc-runtime python3.12 eval/score.py \
    --submission /fixture/submission.txt --gt /fixture/gt.txt --track A --out-dir /fixture/results

echo "== smoke test passed =="
