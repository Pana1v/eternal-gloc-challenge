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
    score_scenario, random_baseline_score, RANDOM_TRIALS,
    S_FINE_TRANS_M, S_FINE_ROT_DEG, S_COARSE_TRANS_M, S_COARSE_ROT_DEG,
    LOSS_TRANS_WEIGHT, LOSS_ROT_WEIGHT, LOSS_TRANS_CAP_M, LOSS_ROT_CAP_DEG,
    TRACK_B_LOSS_WEIGHT, TRACK_B_BUDGET_WEIGHT, TRACK_B_MAX_STEPS, MAX_HYPOTHESES,
)
from io_formats import load_gt, load_submission, load_tiers, load_compute_meta

# plotting is optional: matplotlib is present in the runtime image but a bare
# scoring environment may not have it, and a missing plot must not fail a run
try:
    import plot_results
except ImportError:
    plot_results = None

# the HTML report is regenerated after every scoring run, so it always covers
# every method scored into --out-dir so far, not just this one
import report


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


def write_summary_json(out_path: str, track: str, overall: dict, per_tier: dict, args,
                        random_baseline=None, compute=None):
    summary = {
        "generated_at": datetime.now().isoformat(),
        "track": track,
        "overall": overall["overall"],
        "random_baseline": random_baseline,
        # reported alongside the score, never folded into it: this figure is
        # self-declared by the submitting method, so ranking on it would be
        # trivially gameable
        "compute": compute,
        "margin_over_random": (None if random_baseline is None
                                else overall["overall"]["score"] - random_baseline["score"]),
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


def print_compute(compute):
    """Compute cost is an independent KPI: reported next to the score, never
    part of it. A method that wins on accuracy while costing 100x the compute
    should be visibly doing so, not silently penalized.
    """
    if compute is None:
        print("compute: n/a (no <submission>.meta.json sidecar)")
        return

    parts = []
    if compute["runtime_sec_total"] is not None:
        parts.append(f"{compute['runtime_sec_total']:.1f}s total")
    if compute["runtime_sec_per_scenario"] is not None:
        parts.append(f"{compute['runtime_sec_per_scenario']:.2f}s/scenario")
    if compute["peak_rss_mb"] is not None:
        parts.append(f"{compute['peak_rss_mb']:.0f} MB peak RSS")
    print("compute (independent KPI): " + (", ".join(parts) if parts else "n/a"))


def write_plots(stats_path: str, tiers_path, random_baseline):
    """Renders the plots into the result directory alongside stats.csv."""
    if plot_results is None:
        print("skipping plots: matplotlib not installed", file=sys.stderr)
        return

    argv = ["--stats", stats_path]
    if tiers_path:
        argv += ["--tiers", tiers_path]
    if random_baseline is not None:
        argv += ["--random-loss", str(random_baseline["mean_loss"])]
    plot_results.main(argv)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", required=True)
    parser.add_argument("--gt", required=True)
    parser.add_argument("--track", required=True, choices=["A", "B"])
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--method")
    parser.add_argument("--tiers", help="optional private CSV: scenario_id,tier")
    parser.add_argument("--random-trials", type=int, default=RANDOM_TRIALS,
                         help="random-guess reference draws per scenario; 0 disables")
    parser.add_argument("--no-plots", action="store_true", help="skip rendering plots")
    args = parser.parse_args(argv)

    if args.method is None:
        args.method = os.path.splitext(os.path.basename(args.submission))[0]

    gt = load_gt(args.gt)
    submissions = load_submission(args.submission, args.track)
    scores = score_all(gt, submissions, args.track)

    overall = aggregate(scores)
    tiers = load_tiers(args.tiers, args.track) if args.tiers else {}
    per_tier = {}
    if tiers:
        per_tier = aggregate(scores, group_key=lambda sid: tiers.get(sid, "unknown"))

    compute = load_compute_meta(args.submission)

    random_baseline = None
    if args.random_trials > 0:
        random_baseline = random_baseline_score(list(gt.values()), args.track,
                                                 trials=args.random_trials)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(args.out_dir, f"{args.split}_{args.method}_{timestamp}")
    os.makedirs(result_dir, exist_ok=True)

    stats_path = os.path.join(result_dir, "stats.csv")
    write_stats_csv(scores, stats_path)
    write_summary_json(os.path.join(result_dir, "summary.json"), args.track, overall, per_tier,
                        args, random_baseline, compute)

    print(f"scored {len(scores)} scenarios ({overall['overall']['n_missing']} missing)")
    print(f"headline score: {overall['overall']['score']:.2f}")
    if random_baseline is not None:
        margin = overall["overall"]["score"] - random_baseline["score"]
        print(f"random baseline: {random_baseline['score']:.2f}  (margin {margin:+.2f})")
    elif args.random_trials > 0:
        print("random baseline: n/a (ground truth extent too small to sample)")
    print(f"SR@fine: {overall['overall']['sr_fine']:.3f}  SR@coarse: {overall['overall']['sr_coarse']:.3f}")
    print_compute(compute)
    print(f"wrote {stats_path}")
    print(f"wrote {result_dir}/summary.json")

    if not args.no_plots:
        write_plots(stats_path, args.tiers, random_baseline)

    report_path = os.path.join(args.out_dir, "report.html")
    report.main(["--results", args.out_dir, "--out", report_path,
                 *(["--tiers", args.tiers] if args.tiers else []),
                 "--title", f"GLoc Eval - {args.split} Track {args.track}"])

    return result_dir


if __name__ == "__main__":
    main()
