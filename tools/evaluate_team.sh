#!/usr/bin/env bash
# One-shot: run a team's submission against a dataset, score it with this
# repo's own eval/score.py, and merge the result with their self-reported
# numbers into one JSON record. Wraps run_submission.sh; see that script for
# what COMMAND and the $SCENARIOS/$MAP/$OUT env vars mean.
#
# Usage:
#   tools/evaluate_team.sh LABEL SUBMISSION_ROOT DATASET_DIR \
#       SELF_SCORE SELF_SEC_PER_SCENARIO SELF_SOURCE -- COMMAND...
#
#   LABEL                  short id, e.g. aborrt
#   SELF_SCORE             team's own reported dev-set score, or "NA"
#   SELF_SEC_PER_SCENARIO  team's own reported per-scenario compute time, or "NA"
#   SELF_SOURCE            one-line pointer to where those two came from, e.g.
#                          "REPORT.pdf table 3"
set -euo pipefail

if [ $# -lt 7 ]; then
    echo "usage: $0 LABEL SUBMISSION_ROOT DATASET_DIR SELF_SCORE SELF_SEC_PER_SCENARIO SELF_SOURCE -- COMMAND..." >&2
    exit 2
fi

LABEL=$1 SUBMISSION_ROOT=$2 DATASET_DIR=$3 SELF_SCORE=$4 SELF_SEC=$5 SELF_SOURCE=$6
shift 6
[ "$1" = "--" ] || { echo "usage: $0 LABEL SUBMISSION_ROOT DATASET_DIR SELF_SCORE SELF_SEC_PER_SCENARIO SELF_SOURCE -- COMMAND..." >&2; exit 2; }
shift

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$HERE/results/tiebreaker/$LABEL"

"$HERE/tools/run_submission.sh" "$SUBMISSION_ROOT" "$DATASET_DIR" "$LABEL" -- "$@"

# score.py nests its own output under results/tiebreaker/<label>/dev_<label>_<timestamp>/;
# take the most recent one in case this label was ever run before.
FRESH_SUMMARY="$(find "$OUT_DIR" -mindepth 2 -maxdepth 2 -name "summary.json" | sort | tail -n1)"
if [ -z "$FRESH_SUMMARY" ]; then
    echo "evaluate_team: no summary.json found under $OUT_DIR after scoring" >&2
    exit 1
fi

python3 - "$LABEL" "$SELF_SCORE" "$SELF_SEC" "$SELF_SOURCE" "$FRESH_SUMMARY" "$OUT_DIR/timing.json" \
    > "$OUT_DIR/evaluation.json" <<'PYEOF'
import json, sys

label, self_score, self_sec, self_source, summary_path, timing_path = sys.argv[1:7]
summary = json.load(open(summary_path))
timing = json.load(open(timing_path))
overall = summary["overall"]
compute = summary.get("compute")  # from the submission's own <file>.meta.json sidecar, if it wrote one

def maybe_float(s):
    try:
        return float(s)
    except ValueError:
        return None

record = {
    "label": label,
    "self_reported": {
        "score": maybe_float(self_score) if self_score != "NA" else None,
        "sec_per_scenario": maybe_float(self_sec) if self_sec != "NA" else None,
        "source": self_source,
    },
    "tb_d100": {
        "score": overall.get("score"),
        "sr_fine": overall.get("sr_fine"),
        "sr_coarse": overall.get("sr_coarse"),
        "n_missing": overall.get("n_missing"),
        "margin_over_random": summary.get("margin_over_random"),
        # self-declared by the submission's own meta.json sidecar (not from our stopwatch);
        # never used for ranking, only reported alongside the score
        "self_declared_compute": compute,
        # our own external wall-clock measurement, independent of what the submission reports
        "measured_wall_s": timing.get("wall_s"),
        "measured_sec_per_scenario": timing.get("sec_per_scenario"),
    },
}
json.dump(record, sys.stdout, indent=2)
PYEOF

echo
echo "[$LABEL] wrote $OUT_DIR/evaluation.json"
cat "$OUT_DIR/evaluation.json"
