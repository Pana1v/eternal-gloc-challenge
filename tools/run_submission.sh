#!/usr/bin/env bash
# Run one Track A submission against a dataset, time it, and score it with
# THIS repo's own eval/score.py -- not whatever copy the submission bundles --
# so every team is judged by one unmodified scorer.
#
# Usage:
#   tools/run_submission.sh SUBMISSION_ROOT DATASET_DIR METHOD_LABEL -- COMMAND...
#
# SUBMISSION_ROOT   the team's checked-out repo (COMMAND runs with this as cwd)
# DATASET_DIR       holds map/prior_map.pcd, scenarios/A/, gt/A.txt, [gt/tiers.csv]
# METHOD_LABEL      short name; results land in results/tiebreaker/METHOD_LABEL/
# COMMAND           however the team invokes their own pipeline; reference the
#                   dataset via the $SCENARIOS / $MAP / $OUT env vars this
#                   script exports, e.g.:
#
#   tools/run_submission.sh ~/teamx/repo ~/tb_d100 teamx -- \
#       python3 run.py --scenarios "$SCENARIOS" --map "$MAP" --out "$OUT"
set -euo pipefail

if [ $# -lt 4 ]; then
    echo "usage: $0 SUBMISSION_ROOT DATASET_DIR METHOD_LABEL -- COMMAND..." >&2
    exit 2
fi

SUBMISSION_ROOT=$1
DATASET_DIR=$2
METHOD_LABEL=$3
shift 3
[ "$1" = "--" ] || { echo "usage: $0 SUBMISSION_ROOT DATASET_DIR METHOD_LABEL -- COMMAND..." >&2; exit 2; }
shift

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_DIR="$(cd "$DATASET_DIR" && pwd)"
OUT_DIR="$HERE/results/tiebreaker/$METHOD_LABEL"
mkdir -p "$OUT_DIR"

export SCENARIOS="$DATASET_DIR/scenarios/A"
export MAP="$DATASET_DIR/map/prior_map.pcd"
export OUT="$OUT_DIR/submission.txt"

echo "[$METHOD_LABEL] cwd=$SUBMISSION_ROOT"
echo "[$METHOD_LABEL] $*"
cd "$SUBMISSION_ROOT"
START=$(date +%s.%N)
"$@"
END=$(date +%s.%N)

N=$(find "$SCENARIOS" -mindepth 1 -maxdepth 1 -type d | wc -l)
python3 - "$START" "$END" "$N" > "$OUT_DIR/timing.json" <<'PYEOF'
import json, sys
start, end, n = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
wall = end - start
json.dump({"wall_s": round(wall, 2), "n_scenarios": n,
           "sec_per_scenario": round(wall / n, 4)}, sys.stdout, indent=2)
PYEOF
echo

TIERS_ARGS=()
[ -f "$DATASET_DIR/gt/tiers.csv" ] && TIERS_ARGS=(--tiers "$DATASET_DIR/gt/tiers.csv")

python3 "$HERE/eval/score.py" \
    --submission "$OUT" \
    --gt "$DATASET_DIR/gt/A.txt" \
    --track A \
    --method "$METHOD_LABEL" \
    --out-dir "$OUT_DIR" \
    --no-plots \
    "${TIERS_ARGS[@]}"

echo "[$METHOD_LABEL] timing: $(cat "$OUT_DIR/timing.json")"
