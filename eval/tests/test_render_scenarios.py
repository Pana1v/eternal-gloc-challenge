"""Per-scenario figures must not depend on how many workers rendered them.

build_fixture is shared with the profiling driver in the pull request
description, so the timings quoted there come from the same geometry the
tests assert on.
"""

import json
import os
import shutil

import numpy as np
from PIL import Image

import render_scenarios
import report

MAP_EXTENT_M = (100.0, 80.0)
RACK_ROWS = 8
SCAN_POINTS = 57_600         # 32 beams x 1800 samples, per docs/SENSORS.md
CAMERA_SIZE = (1280, 800)


def build_fixture(root, n_scenarios: int, map_points: int, seed: int = 0):
    """A synthetic warehouse: a floor, rack rows, and n_scenarios rigs looking
    at it. Returns (map_path, scenarios_dir, gt_path, submission_path).
    """
    import open3d as o3d
    from PIL import Image

    root = str(root)
    os.makedirs(root, exist_ok=True)
    rng = np.random.default_rng(seed)
    x_max, y_max = MAP_EXTENT_M

    # half the budget on the floor, half spread over vertical rack faces, so
    # the RACK_BAND_M crop in render_scenarios has something to draw
    n_floor = map_points // 2
    floor = np.column_stack([rng.uniform(0, x_max, n_floor), rng.uniform(0, y_max, n_floor),
                             rng.normal(0.0, 0.01, n_floor)])
    n_rack = map_points - n_floor
    rack_x = rng.integers(0, RACK_ROWS, n_rack) * (x_max / RACK_ROWS) + 2.0
    racks = np.column_stack([rack_x + rng.normal(0, 0.4, n_rack),
                             rng.uniform(0, y_max, n_rack),
                             rng.uniform(0.0, 6.0, n_rack)])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.concatenate([floor, racks]))
    map_path = os.path.join(root, "prior_map.pcd")
    o3d.io.write_point_cloud(map_path, pcd)

    scenarios_dir = os.path.join(root, "scenarios")
    gt_lines, sub_lines = [], []

    for i in range(n_scenarios):
        sid = f"{i:06d}"
        sdir = os.path.join(scenarios_dir, sid)
        os.makedirs(sdir, exist_ok=True)

        scan = rng.normal(0.0, 8.0, size=(SCAN_POINTS, 3))
        scan[:, 2] = rng.uniform(0.0, 6.0, SCAN_POINTS)
        spcd = o3d.geometry.PointCloud()
        spcd.points = o3d.utility.Vector3dVector(scan)
        o3d.io.write_point_cloud(os.path.join(sdir, "lidar.pcd"), spcd)

        w, h = CAMERA_SIZE
        img = rng.integers(60, 90, size=(h, w, 3)).astype("uint8")
        Image.fromarray(img).save(os.path.join(sdir, "camera.png"))

        with open(os.path.join(sdir, "meta.json"), "w") as f:
            json.dump({"scenario_id": sid, "track": "A", "world_id": "profiling"}, f)

        gx, gy = rng.uniform(5, x_max - 5), rng.uniform(5, y_max - 5)
        gt_lines.append(f"{sid} 1 0 0 {gx:.4f} 0 1 0 {gy:.4f} 0 0 1 0.5\n")
        # a plausible-looking estimate: right most of the time, badly wrong now
        # and then, so the figure has both the error line and the plain case
        off = rng.normal(0, 0.4, 2) if i % 7 else rng.normal(0, 12.0, 2)
        sub_lines.append(f"{sid} 0 1.0 1 0 0 {gx + off[0]:.4f} 0 1 0 {gy + off[1]:.4f} 0 0 1 0.5\n")

    gt_path = os.path.join(root, "gt.txt")
    with open(gt_path, "w") as f:
        f.writelines(gt_lines)
    sub_path = os.path.join(root, "sub_a.txt")
    with open(sub_path, "w") as f:
        f.writelines(sub_lines)

    return map_path, scenarios_dir, gt_path, sub_path


def _render(root, jobs, out_name):
    map_path, scenarios_dir, gt_path, sub_path = build_fixture(root, n_scenarios=5, map_points=4000)
    out_dir = os.path.join(root, out_name)
    render_scenarios.main([
        "--scenarios", scenarios_dir, "--map", map_path, "--gt", gt_path,
        "--out", out_dir, "--submission", f"bl_a={sub_path}", "--jobs", str(jobs),
    ])
    return out_dir


def test_jobs_one_and_many_write_the_same_png_pixels(tmp_path):
    """--jobs 1 (serial, no pool) and --jobs 3 (spawn pool) must render the
    identical scenario set to bit-identical pixels: the report links to
    whichever ran, and a reader must never be able to tell which one did.
    """
    # same seed both times, so the two runs see the same synthetic scenarios
    out1 = _render(tmp_path / "serial", jobs=1, out_name="figs1")
    out3 = _render(tmp_path / "serial", jobs=3, out_name="figs3")

    names1 = set(os.listdir(out1))
    names3 = set(os.listdir(out3))
    assert names1 == names3
    assert len(names1) == 5

    for name in names1:
        a = Image.open(os.path.join(out1, name)).tobytes()
        b = Image.open(os.path.join(out3, name)).tobytes()
        assert a == b, f"{name} differs between --jobs 1 and --jobs 3"


def test_pixel_comparison_actually_detects_a_difference(tmp_path):
    """Falsifies the comparator above: swap in a different figure and confirm
    the pixel check reports it, so a bug that renders nothing (or renders the
    same blank figure for every scenario) can't pass silently.
    """
    out_dir = _render(tmp_path, jobs=1, out_name="figs")
    names = sorted(os.listdir(out_dir))
    a = Image.open(os.path.join(out_dir, names[0])).tobytes()
    b = Image.open(os.path.join(out_dir, names[1])).tobytes()
    assert a != b


def test_report_html_identical_regardless_of_jobs(tmp_path):
    """The report only cares about the figures directory's relative path, not
    which --jobs value produced its contents, so scoring one submission with
    --jobs 1 and another with --jobs 3 must still yield the same report.html.
    """
    out1 = _render(tmp_path / "a", jobs=1, out_name="figs")
    out3 = _render(tmp_path / "b", jobs=3, out_name="figs")

    results = tmp_path / "results"
    run_dir = results / "bl_a"
    run_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text(
        '{"track":"A","overall":{"score":50.0,"sr_fine":0.5,"sr_coarse":0.5,'
        '"oracle_sr_fine":0.5,"mean_loss":0.5,"n_missing":0,"n_scenarios":5},'
        '"run":{"method":"bl_a"}}')
    (run_dir / "stats.csv").write_text(
        "scenario_id,e_t,e_r,sr_fine\n" + "\n".join(f"{i:06d},0.10,1.0,1" for i in range(5)))

    reports = []
    for out_dir in (out1, out3):
        figures = results / "figures"
        if figures.exists():
            shutil.rmtree(figures)
        shutil.copytree(out_dir, figures)
        out_html = tmp_path / "report.html"
        report.main(["--results", str(results), "--out", str(out_html), "--figures", str(figures)])
        reports.append(out_html.read_text())

    assert reports[0] == reports[1]
