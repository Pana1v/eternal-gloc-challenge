"""Writes the submission.txt format (design ch. 4.5):
`<scenario_id> <k> <w> r11 r12 r13 tx r21 r22 r23 ty r31 r32 r33 tz`
one line per (scenario, hypothesis) pair, k in [0, 3), weights per
scenario summing to <= 1.
"""

import json
import numpy as np

WEIGHT_DECIMALS = 4  # precision the submission format writes weights at


def pose_matrix_from_xy_yaw(x: float, y: float, yaw: float, z: float = 0.0) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    T = np.eye(4)
    T[0, 0], T[0, 1] = c, -s
    T[1, 0], T[1, 1] = s, c
    T[0, 3], T[1, 3], T[2, 3] = x, y, z
    return T


def matrix_to_kitti_line(T: np.ndarray) -> str:
    return " ".join("%.9e" % v for v in T[:3, :].ravel())


class SubmissionWriter:
    def __init__(self, out_path: str):
        self.out_path = out_path
        self._lines = []

    def add(self, scenario_id: str, T: np.ndarray, weight: float = 1.0, k: int = 0, steps_used: int = None):
        extra = f" {steps_used}" if steps_used is not None else ""
        self._lines.append(f"{scenario_id} {k} {weight:.{WEIGHT_DECIMALS}f}{extra} {matrix_to_kitti_line(T)}")

    def write(self):
        with open(self.out_path, "w") as f:
            for line in self._lines:
                f.write(line + "\n")


def peak_rss_mb() -> float:
    """Peak resident set size of this process. ru_maxrss is kilobytes on Linux
    but bytes on macOS, so the unit has to be branched on, not assumed."""
    import resource
    import sys
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024


def write_submission_meta(out_path: str, method_name: str, runtime_sec_total: float, params: dict,
                           n_scenarios: int):
    """Records what the scorer cannot measure for itself: it only ever reads a
    text file, so compute cost has to be declared by whoever produced it.
    Reported as an independent KPI, never folded into the headline score.
    """
    import platform
    with open(out_path, "w") as f:
        json.dump({
            "method_name": method_name,
            "runtime_sec_total": runtime_sec_total,
            "n_scenarios": n_scenarios,
            "runtime_sec_per_scenario": runtime_sec_total / n_scenarios if n_scenarios else None,
            "peak_rss_mb": peak_rss_mb(),
            "machine": {"cpu": platform.processor() or platform.machine(), "platform": platform.platform()},
            "params": params,
        }, f, indent=2)
