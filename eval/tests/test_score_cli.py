"""End-to-end: build a tiny synthetic GT + submission, run the real CLI
(score.main), and check the produced stats.csv/summary.json.
"""

import csv
import json
import os

import numpy as np

from metrics import kitti_line_to_matrix
from score import main as score_main


def _kitti_line(T):
    return " ".join("%.9e" % v for v in T[:3, :].ravel())


def _identity_line(scenario_id):
    return f"{scenario_id} {_kitti_line(np.eye(4))}"


def _submission_line(scenario_id, k, weight, T, steps_used=None):
    extra = f" {steps_used}" if steps_used is not None else ""
    return f"{scenario_id} {k} {weight}{extra} {_kitti_line(T)}"


def test_cli_end_to_end_track_a(tmp_path):
    gt_path = tmp_path / "gt.txt"
    sub_path = tmp_path / "submission.txt"

    # s0: exact match (loss 0). s1: missing from submission (loss 1).
    gt_path.write_text(_identity_line("s0") + "\n" + _identity_line("s1") + "\n")

    T_exact = np.eye(4)
    sub_path.write_text(_submission_line("s0", 0, 1.0, T_exact) + "\n")

    out_dir = tmp_path / "results"
    result_dir = score_main([
        "--submission", str(sub_path), "--gt", str(gt_path), "--track", "A",
        "--out-dir", str(out_dir), "--split", "dev", "--method", "unittest",
    ])

    assert os.path.isdir(result_dir)
    stats_path = os.path.join(result_dir, "stats.csv")
    summary_path = os.path.join(result_dir, "summary.json")
    assert os.path.exists(stats_path)
    assert os.path.exists(summary_path)

    with open(stats_path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    by_id = {r["scenario_id"]: r for r in rows}
    assert float(by_id["s0"]["L"]) == 0.0
    assert by_id["s0"]["missing"] == "0"
    assert float(by_id["s1"]["L"]) == 1.0
    assert by_id["s1"]["missing"] == "1"

    with open(summary_path) as f:
        summary = json.load(f)
    overall = summary["overall"]
    assert overall["n_scenarios"] == 2
    assert overall["n_missing"] == 1
    # mean loss = (0 + 1) / 2 = 0.5 -> score = 100*(1-0.5) = 50
    assert abs(overall["score"] - 50.0) < 1e-6
    assert 0.0 <= overall["score"] <= 100.0
    assert summary["parameters"]["s_fine_trans_m"] == 0.5


def test_cli_end_to_end_track_b_with_steps_used(tmp_path):
    gt_path = tmp_path / "gt_b.txt"
    sub_path = tmp_path / "submission_b.txt"

    gt_path.write_text(_identity_line("s0") + "\n")
    sub_path.write_text(_submission_line("s0", 0, 1.0, np.eye(4), steps_used=10) + "\n")

    out_dir = tmp_path / "results"
    result_dir = score_main([
        "--submission", str(sub_path), "--gt", str(gt_path), "--track", "B",
        "--out-dir", str(out_dir), "--split", "dev", "--method", "unittest_b",
    ])

    with open(os.path.join(result_dir, "stats.csv")) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    # perfect pose match, steps_used=10/40 -> L = 0.75*0 + 0.25*0.25 = 0.0625
    assert abs(float(rows[0]["L"]) - 0.0625) < 1e-9
    assert rows[0]["steps_used"] == "10"


def test_kitti_line_round_trips_through_writer_format():
    """Confirms io_formats' parser matches baselines/common/submission_writer's
    actual line format byte-for-byte, not just in theory."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "baselines", "common"))
    from submission_writer import SubmissionWriter

    T = np.eye(4)
    T[0, 3], T[1, 3], T[2, 3] = 1.5, -2.5, 0.0

    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        tmp_name = f.name
    writer = SubmissionWriter(tmp_name)
    writer.add("s0", T, weight=0.75, k=0)
    writer.write()

    with open(tmp_name) as f:
        line = f.readline().strip()
    tokens = line.split()
    assert tokens[0] == "s0"
    assert tokens[1] == "0"
    assert abs(float(tokens[2]) - 0.75) < 1e-6
    assert len(tokens) == 15  # id, k, w, 12 pose values

    parsed = kitti_line_to_matrix([float(v) for v in tokens[3:]])
    np.testing.assert_allclose(parsed, T, atol=1e-6)
    os.unlink(tmp_name)
