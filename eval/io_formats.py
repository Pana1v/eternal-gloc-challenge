"""Parsing for ground truth and submission file formats (see
baselines/common/submission_writer.py for the writer these must round-trip):

GT line:            <scenario_id> r11 r12 r13 tx r21 r22 r23 ty r31 r32 r33 tz
Submission line A:   <scenario_id> <k> <w> r11 ... tz
Submission line B:   <scenario_id> <k> <w> <steps_used> r11 ... tz
"""

from collections import defaultdict

from metrics import kitti_line_to_matrix, Hypothesis, MAX_HYPOTHESES


class FormatError(ValueError):
    pass


def load_gt(path: str) -> dict:
    gt = {}
    with open(path) as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            tokens = line.split()
            if len(tokens) != 13:
                raise FormatError(f"{path}:{lineno}: expected 13 tokens (id + 12 pose values), got {len(tokens)}")
            scenario_id = tokens[0]
            values = [float(v) for v in tokens[1:]]
            gt[scenario_id] = kitti_line_to_matrix(values)
    return gt


def load_submission(path: str, track: str) -> dict:
    """Returns {scenario_id: [Hypothesis, ...]}, at most MAX_HYPOTHESES per
    scenario, in the order they appeared in the file.
    """
    expected_tokens = 16 if track == "B" else 15
    grouped = defaultdict(list)
    with open(path) as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            tokens = line.split()
            if len(tokens) != expected_tokens:
                raise FormatError(
                    f"{path}:{lineno}: expected {expected_tokens} tokens for track {track}, got {len(tokens)}"
                )
            scenario_id = tokens[0]
            k = int(tokens[1])
            weight = float(tokens[2])
            if track == "B":
                steps_used = int(tokens[3])
                pose_values = [float(v) for v in tokens[4:]]
            else:
                steps_used = None
                pose_values = [float(v) for v in tokens[3:]]
            T = kitti_line_to_matrix(pose_values)
            grouped[scenario_id].append(Hypothesis(k=k, weight=weight, T=T, steps_used=steps_used))

    for scenario_id, hyps in grouped.items():
        if len(hyps) > MAX_HYPOTHESES:
            raise FormatError(f"{path}: scenario {scenario_id} has {len(hyps)} hypotheses, max is {MAX_HYPOTHESES}")
        weight_sum = sum(h.weight for h in hyps)
        if weight_sum > 1.0 + 1e-6:
            raise FormatError(f"{path}: scenario {scenario_id} hypothesis weights sum to {weight_sum} > 1")

    return dict(grouped)


def load_tiers(path: str) -> dict:
    """Optional private tier lookup CSV: scenario_id,tier"""
    tiers = {}
    with open(path) as f:
        header = f.readline()
        cols = [c.strip() for c in header.strip().split(",")]
        id_idx = cols.index("scenario_id")
        tier_idx = cols.index("tier")
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            tiers[parts[id_idx]] = parts[tier_idx]
    return tiers
