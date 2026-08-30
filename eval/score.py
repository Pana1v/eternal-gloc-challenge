#!/usr/bin/env python3
"""Official scorer (design ch. 4). Scores a submission against ground truth
and writes results/<split>_<method>_<timestamp>/{stats.csv,summary.json}.

Usage:
    python score.py --submission submission.txt --gt gt/A.txt --track A \\
        --out-dir results --split dev [--method my_method] [--tiers tiers.csv]
"""

import argparse
import csv
import json
import math
import os
import platform
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import (
    score_scenario, S_FINE_TRANS_M, S_FINE_ROT_DEG, S_COARSE_TRANS_M, S_COARSE_ROT_DEG,
    LOSS_TRANS_WEIGHT, LOSS_ROT_WEIGHT, LOSS_TRANS_CAP_M, LOSS_ROT_CAP_DEG,
    TRACK_B_LOSS_WEIGHT, TRACK_B_BUDGET_WEIGHT, TRACK_B_MAX_STEPS, MAX_HYPOTHESES,
)
from io_formats import load_gt, load_submission, load_tiers


def score_all(gt: dict, submissions: dict, track: str):
    scores = []
    for scenario_id, T_gt in gt.items():
        hyps = submissions.get(scenario_id, [])
        scores.append(score_scenario(scenario_id, T_gt, hyps, track))
    return scores


def _mean(values):
    values = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return sum(values) / len(values) if values else float("nan")


def aggregate(scores, group_key=None):
    """group_key(scenario_id) -> group label, or None for a single overall
    group. Returns {group_label: {metrics...}}.
    """
    groups = {}
    for s in scores:
        label = group_key(s.scenario_id) if group_key else "overall"
        groups.setdefault(label, []).append(s)

    result = {}
    for label, group_scores in groups.items():
        mean_loss = _mean([s.loss for s in group_scores])
        result[label] = {
            "n_scenarios": len(group_scores),
            "n_missing": sum(1 for s in group_scores if s.missing),
            "mean_loss": mean_loss,
            "score": 100.0 * (1.0 - mean_loss),
            "sr_fine": _mean([1.0 if s.sr_fine else 0.0 for s in group_scores]),
            "sr_coarse": _mean([1.0 if s.sr_coarse else 0.0 for s in group_scores]),
            "oracle_sr_fine": _mean([1.0 if s.oracle_sr_fine else 0.0 for s in group_scores]),
            "oracle_sr_coarse": _mean([1.0 if s.oracle_sr_coarse else 0.0 for s in group_scores]),
        }
    return result


def write_stats_csv(scores, out_path: str):
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scenario_id", "e_t", "e_r", "L", "steps_used", "sr_fine", "sr_coarse",
                          "oracle_sr_fine", "oracle_sr_coarse", "missing"])
        for s in scores:
            writer.writerow([
                s.scenario_id,
                "" if math.isnan(s.e_t) else s.e_t,
                "" if math.isnan(s.e_r) else s.e_r,
                s.loss,
                "" if s.steps_used is None else s.steps_used,
                int(s.sr_fine), int(s.sr_coarse),
                int(s.oracle_sr_fine), int(s.oracle_sr_coarse),
                int(s.missing),
            ])


def write_summary_json(out_path: str, track: str, overall: dict, per_tier: dict, args):
    summary = {
        "generated_at": datetime.now().isoformat(),
        "track": track,
        "overall": overall["overall"],
        "per_tier": per_tier,
        "parameters": {
            "s_fine_trans_m": S_FINE_TRANS_M, "s_fine_rot_deg": S_FINE_ROT_DEG,
            "s_coarse_trans_m": S_COARSE_TRANS_M, "s_coarse_rot_deg": S_COARSE_ROT_DEG,
            "loss_trans_weight": LOSS_TRANS_WEIGHT, "loss_rot_weight": LOSS_ROT_WEIGHT,
            "loss_trans_cap_m": LOSS_TRANS_CAP_M, "loss_rot_cap_deg": LOSS_ROT_CAP_DEG,
            "track_b_loss_weight": TRACK_B_LOSS_WEIGHT, "track_b_budget_weight": TRACK_B_BUDGET_WEIGHT,
            "track_b_max_steps": TRACK_B_MAX_STEPS, "max_hypotheses": MAX_HYPOTHESES,
        },
        "run": {
            "submission": os.path.abspath(args.submission),
            "gt": os.path.abspath(args.gt),
            "method": args.method,
            "machine": {"cpu": platform.processor() or platform.machine(), "platform": platform.platform()},
        },
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", required=True)
    parser.add_argument("--gt", required=True)
    parser.add_argument("--track", required=True, choices=["A", "B"])
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--method")
    parser.add_argument("--tiers", help="optional private CSV: scenario_id,tier")
    args = parser.parse_args(argv)

    if args.method is None:
        args.method = os.path.splitext(os.path.basename(args.submission))[0]

    gt = load_gt(args.gt)
    submissions = load_submission(args.submission, args.track)
    scores = score_all(gt, submissions, args.track)

    overall = aggregate(scores)
    per_tier = {}
    if args.tiers:
        tiers = load_tiers(args.tiers)
        per_tier = aggregate(scores, group_key=lambda sid: tiers.get(sid, "unknown"))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(args.out_dir, f"{args.split}_{args.method}_{timestamp}")
    os.makedirs(result_dir, exist_ok=True)

    write_stats_csv(scores, os.path.join(result_dir, "stats.csv"))
    write_summary_json(os.path.join(result_dir, "summary.json"), args.track, overall, per_tier, args)

    print(f"scored {len(scores)} scenarios ({overall['overall']['n_missing']} missing)")
    print(f"headline score: {overall['overall']['score']:.2f}")
    print(f"SR@fine: {overall['overall']['sr_fine']:.3f}  SR@coarse: {overall['overall']['sr_coarse']:.3f}")
    print(f"wrote {result_dir}/stats.csv")
    print(f"wrote {result_dir}/summary.json")
    return result_dir


if __name__ == "__main__":
    main()
