"""Hand-computed pose-error / loss cases, checked against the formulas by hand,
not against the code's own output.
"""

import math
import numpy as np

from metrics import (
    pose_error, bounded_loss, track_b_loss, is_success, score_scenario, Hypothesis,
    S_FINE_TRANS_M, S_FINE_ROT_DEG, S_COARSE_TRANS_M, S_COARSE_ROT_DEG,
)


def _translation(x, y, z):
    T = np.eye(4)
    T[0, 3], T[1, 3], T[2, 3] = x, y, z
    return T


def _rotation_z(deg):
    rad = math.radians(deg)
    c, s = math.cos(rad), math.sin(rad)
    T = np.eye(4)
    T[0, 0], T[0, 1] = c, -s
    T[1, 0], T[1, 1] = s, c
    return T


def test_pose_error_pure_translation_is_3_4_5_triangle():
    T_gt = np.eye(4)
    T_hat = _translation(3.0, 4.0, 0.0)
    e_t, e_r = pose_error(T_gt, T_hat)
    assert abs(e_t - 5.0) < 1e-9
    assert abs(e_r - 0.0) < 1e-9


def test_pose_error_pure_90deg_rotation():
    T_gt = np.eye(4)
    T_hat = _rotation_z(90.0)
    e_t, e_r = pose_error(T_gt, T_hat)
    assert abs(e_t - 0.0) < 1e-9
    assert abs(e_r - 90.0) < 1e-6


def test_pose_error_is_invariant_to_gt_frame():
    """e_t should be plain Euclidean distance between translations,
    regardless of a shared rotation on both sides (sanity: T_gt^-1 @ T_hat's
    translation norm-preserving property)."""
    T_gt = _rotation_z(37.0)
    T_gt[0, 3], T_gt[1, 3] = 5.0, -2.0
    T_hat = T_gt.copy()
    T_hat[0, 3] += 3.0
    T_hat[1, 3] += 4.0
    e_t, e_r = pose_error(T_gt, T_hat)
    assert abs(e_t - 5.0) < 1e-9
    assert abs(e_r - 0.0) < 1e-9


def test_bounded_loss_hand_computed_small_errors():
    # e_t=1.0 (half of 2.0 cap), e_r=10.0 (half of 20.0 cap)
    loss = bounded_loss(1.0, 10.0)
    assert abs(loss - (0.7 * 0.5 + 0.3 * 0.5)) < 1e-9  # == 0.5


def test_bounded_loss_caps_at_one_per_term():
    # e_t=5.0 (way past 2.0 cap -> capped to 1), e_r=0
    loss = bounded_loss(5.0, 0.0)
    assert abs(loss - 0.7) < 1e-9
    # e_r=90 (way past 20 cap -> capped to 1), e_t=0
    loss2 = bounded_loss(0.0, 90.0)
    assert abs(loss2 - 0.3) < 1e-9


def test_track_b_loss_hand_computed():
    # base_loss=0.4, steps_used=20/40=0.5 -> 0.75*0.4 + 0.25*0.5 = 0.3 + 0.125 = 0.425
    loss = track_b_loss(0.4, steps_used=20, max_steps=40)
    assert abs(loss - 0.425) < 1e-9


def test_is_success_thresholds_are_inclusive_boundaries():
    assert is_success(S_FINE_TRANS_M, S_FINE_ROT_DEG, S_FINE_TRANS_M, S_FINE_ROT_DEG) is True
    assert is_success(S_FINE_TRANS_M + 0.01, S_FINE_ROT_DEG, S_FINE_TRANS_M, S_FINE_ROT_DEG) is False
    assert is_success(S_COARSE_TRANS_M, S_COARSE_ROT_DEG, S_COARSE_TRANS_M, S_COARSE_ROT_DEG) is True


def test_missing_scenario_scores_loss_one():
    T_gt = np.eye(4)
    result = score_scenario("s0", T_gt, hypotheses=[], track="A")
    assert result.missing is True
    assert result.loss == 1.0
    assert result.sr_fine is False
    assert result.oracle_sr_fine is False


def test_two_hypothesis_weighted_loss_hand_computed():
    """hyp1: e_t=2.0 e_r=0 -> loss=0.7*1.0=0.7 (capped)
    hyp2: e_t=0.0 e_r=0 -> loss=0.0
    weights 0.6 / 0.3 (sum=0.9, remainder 0.1 charged at loss=1)
    expected = 0.6*0.7 + 0.3*0.0 + 0.1*1.0 = 0.42 + 0 + 0.1 = 0.52
    """
    T_gt = np.eye(4)
    hyp1 = Hypothesis(k=0, weight=0.6, T=_translation(2.0, 0.0, 0.0))
    hyp2 = Hypothesis(k=1, weight=0.3, T=np.eye(4))
    result = score_scenario("s0", T_gt, hypotheses=[hyp1, hyp2], track="A")
    assert abs(result.loss - 0.52) < 1e-9
    # primary = highest-weight hypothesis = hyp1 (the worse one, by design of this test)
    assert abs(result.e_t - 2.0) < 1e-9
    assert result.sr_fine is False
    # oracle: hyp2 is a perfect match -> oracle_sr_fine True even though primary misses
    assert result.oracle_sr_fine is True


def test_track_b_scenario_applies_budget_penalty_per_hypothesis():
    T_gt = np.eye(4)
    hyp = Hypothesis(k=0, weight=1.0, T=np.eye(4), steps_used=20)
    result = score_scenario("s0", T_gt, hypotheses=[hyp], track="B")
    # base_loss=0 (perfect match), track_b_loss = 0.75*0 + 0.25*(20/40) = 0.125
    assert abs(result.loss - 0.125) < 1e-9
    assert result.steps_used == 20
