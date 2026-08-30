"""Scoring math for the Eternal GLoc Challenge (design ch. 4): pose error,
per-scenario bounded loss, multi-hypothesis weighting, and success thresholds.
Kept separate from I/O (score.py) so the formulas are easy to unit test against
hand-computed cases.
"""

import math
from dataclasses import dataclass

import numpy as np

# Success thresholds
S_FINE_TRANS_M = 0.5
S_FINE_ROT_DEG = 5.0
S_COARSE_TRANS_M = 2.0
S_COARSE_ROT_DEG = 10.0

# Per-scenario bounded loss weights/caps
LOSS_TRANS_WEIGHT = 0.7
LOSS_ROT_WEIGHT = 0.3
LOSS_TRANS_CAP_M = 2.0
LOSS_ROT_CAP_DEG = 20.0

# Track B budget weighting
TRACK_B_LOSS_WEIGHT = 0.75
TRACK_B_BUDGET_WEIGHT = 0.25
TRACK_B_MAX_STEPS = 40

MAX_HYPOTHESES = 3


def kitti_line_to_matrix(values) -> np.ndarray:
    """12 floats (3x4 row-major) -> a 4x4 homogeneous transform."""
    T = np.eye(4)
    T[:3, :] = np.asarray(values, dtype=np.float64).reshape(3, 4)
    return T


def pose_error(T_gt: np.ndarray, T_hat: np.ndarray):
    """e_t = ||trans(T_gt^-1 @ T_hat)|| (full 3D), e_r = geodesic rotation
    angle in degrees between the two rotations.
    """
    T_gt_inv = np.linalg.inv(T_gt)
    T_rel = T_gt_inv @ T_hat
    e_t = float(np.linalg.norm(T_rel[:3, 3]))

    R_rel = T_rel[:3, :3]
    # numerical safety: (trace-1)/2 can drift a hair outside [-1, 1]
    cos_angle = np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)
    e_r = math.degrees(math.acos(cos_angle))
    return e_t, e_r


def is_success(e_t: float, e_r: float, trans_thresh: float, rot_thresh: float) -> bool:
    return e_t <= trans_thresh and e_r <= rot_thresh


def bounded_loss(e_t: float, e_r: float) -> float:
    return (LOSS_TRANS_WEIGHT * min(e_t / LOSS_TRANS_CAP_M, 1.0) +
            LOSS_ROT_WEIGHT * min(e_r / LOSS_ROT_CAP_DEG, 1.0))


def track_b_loss(base_loss: float, steps_used: int, max_steps: int = TRACK_B_MAX_STEPS) -> float:
    return (TRACK_B_LOSS_WEIGHT * base_loss +
            TRACK_B_BUDGET_WEIGHT * (steps_used / max_steps))


@dataclass
class Hypothesis:
    k: int
    weight: float
    T: np.ndarray
    steps_used: int = None


@dataclass
class ScenarioScore:
    scenario_id: str
    e_t: float             # primary (highest-weight) hypothesis
    e_r: float
    loss: float             # scored weighted loss across all hypotheses
    steps_used: float = None       # primary hypothesis's declared steps (Track B only)
    sr_fine: bool = False           # primary hypothesis
    sr_coarse: bool = False
    oracle_sr_fine: bool = False    # best of all submitted hypotheses
    oracle_sr_coarse: bool = False
    missing: bool = False


def score_scenario(scenario_id: str, T_gt: np.ndarray, hypotheses, track: str) -> ScenarioScore:
    """hypotheses: list[Hypothesis], already validated to have weights
    summing to <= 1 and at most MAX_HYPOTHESES entries. Empty list means the
    scenario was missing from the submission.
    """
    if not hypotheses:
        return ScenarioScore(
            scenario_id=scenario_id, e_t=float("nan"), e_r=float("nan"), loss=1.0,
            steps_used=TRACK_B_MAX_STEPS if track == "B" else None,
            sr_fine=False, sr_coarse=False, oracle_sr_fine=False, oracle_sr_coarse=False,
            missing=True,
        )

    per_hyp = []
    for hyp in hypotheses:
        e_t, e_r = pose_error(T_gt, hyp.T)
        base = bounded_loss(e_t, e_r)
        hyp_loss = track_b_loss(base, hyp.steps_used) if track == "B" else base
        per_hyp.append((hyp, e_t, e_r, hyp_loss))

    weight_sum = sum(h.weight for h in hypotheses)
    weighted_loss = sum(w * hl for (h, _, _, hl), w in zip(per_hyp, (h.weight for h in hypotheses)))
    weighted_loss += max(0.0, 1.0 - weight_sum) * 1.0

    primary = max(per_hyp, key=lambda item: item[0].weight)
    primary_hyp, primary_e_t, primary_e_r, _ = primary

    oracle_fine = any(is_success(e_t, e_r, S_FINE_TRANS_M, S_FINE_ROT_DEG) for _, e_t, e_r, _ in per_hyp)
    oracle_coarse = any(is_success(e_t, e_r, S_COARSE_TRANS_M, S_COARSE_ROT_DEG) for _, e_t, e_r, _ in per_hyp)

    return ScenarioScore(
        scenario_id=scenario_id,
        e_t=primary_e_t, e_r=primary_e_r, loss=weighted_loss,
        steps_used=primary_hyp.steps_used if track == "B" else None,
        sr_fine=is_success(primary_e_t, primary_e_r, S_FINE_TRANS_M, S_FINE_ROT_DEG),
        sr_coarse=is_success(primary_e_t, primary_e_r, S_COARSE_TRANS_M, S_COARSE_ROT_DEG),
        oracle_sr_fine=oracle_fine, oracle_sr_coarse=oracle_coarse,
        missing=False,
    )
