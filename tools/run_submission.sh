#!/usr/bin/env bash
# Run one Track A submission against a dataset, time it, and score it with
# THIS repo's own eval/score.py -- not whatever copy the submission bundles --
# so every team is judged by one unmodified scorer.
#
# Usage:
#   tools/run_submission.sh SUBMISSION_ROOT DATASET_DIR METHOD_LABEL \
#       [--map PATH] [--scenarios DIR] [--gt PATH] [--tiers PATH] -- COMMAND...
#
# SUBMISSION_ROOT   the team's checked-out repo (COMMAND runs with this as cwd)
# DATASET_DIR       base dir; by default map/scenarios/gt are derived from it
#                   as map/prior_map.pcd, scenarios/A/, gt/A.txt, [gt/tiers.csv]
#                   -- the layout of a packaged tiebreaker dataset. The
#                   shipped dev set does NOT follow that layout (it's
#                   scenarios/dev/A and scenarios/dev/gt/A.txt), so override
#                   whichever of the four don't match with the flags above.
# METHOD_LABEL      short name; results land in results/tiebreaker/METHOD_LABEL/
# COMMAND           however the team invokes their own pipeline; reference the
#                   dataset via the $SCENARIOS / $MAP / $OUT env vars this
#                   script exports. Wrap COMMAND in `bash -c '...'` (single
#                   quotes) so those $VARs are expanded AFTER this script sets
#                   them, not by your own shell beforehand -- unquoted, your
#                   shell expands "$SCENARIOS" to empty right now, before this
#                   script has even run, and the baseline gets called with no
#                   dataset path at all. e.g.:
#
#   tools/run_submission.sh ~/teamx/repo ~/tb_d100 teamx -- \
#       bash -c 'python3 run.py --scenarios "$SCENARIOS" --map "$MAP" --out "$OUT"'
#
#   # against the repo's own dev set, whose scenarios/gt live under scenarios/dev/:
#   tools/run_submission.sh . . bl_bbs_dev \
#       --scenarios scenarios/dev/A --gt scenarios/dev/gt/A.txt -- \
#       bash -c 'python3 baselines/bl_bbs/run.py --scenarios "$SCENARIOS" --map "$MAP" --out "$OUT"'
set -euo pipefail

if [ $# -lt 4 ]; then
    echo "usage: $0 SUBMISSION_ROOT DATASET_DIR METHOD_LABEL [--map PATH] [--scenarios DIR] [--gt PATH] [--tiers PATH] -- COMMAND..." >&2
    exit 2
fi

SUBMISSION_ROOT=$1
DATASET_DIR=$2
METHOD_LABEL=$3
shift 3

MAP_OVERRIDE=""
SCENARIOS_OVERRIDE=""
GT_OVERRIDE=""
TIERS_OVERRIDE=""
while [ "${1:-}" != "--" ]; do
    if [ $# -lt 2 ]; then
        echo "usage: $0 SUBMISSION_ROOT DATASET_DIR METHOD_LABEL [--map PATH] [--scenarios DIR] [--gt PATH] [--tiers PATH] -- COMMAND..." >&2
        exit 2
    fi
    case "$1" in
        --map) MAP_OVERRIDE=$2 ;;
        --scenarios) SCENARIOS_OVERRIDE=$2 ;;
        --gt) GT_OVERRIDE=$2 ;;
        --tiers) TIERS_OVERRIDE=$2 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
    shift 2
done
shift  # consume --

# resolve to an absolute path before we cd into SUBMISSION_ROOT below
abspath() {
    if [ -d "$1" ]; then
        (cd "$1" && pwd)
    else
        echo "$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
    fi
}

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_DIR="$(cd "$DATASET_DIR" && pwd)"
OUT_DIR="$HERE/results/tiebreaker/$METHOD_LABEL"
mkdir -p "$OUT_DIR"

export SCENARIOS="${SCENARIOS_OVERRIDE:+$(abspath "$SCENARIOS_OVERRIDE")}"
export SCENARIOS="${SCENARIOS:-$DATASET_DIR/scenarios/A}"
export MAP="${MAP_OVERRIDE:+$(abspath "$MAP_OVERRIDE")}"
export MAP="${MAP:-$DATASET_DIR/map/prior_map.pcd}"
export OUT="$OUT_DIR/submission.txt"
GT="${GT_OVERRIDE:+$(abspath "$GT_OVERRIDE")}"
GT="${GT:-$DATASET_DIR/gt/A.txt}"
TIERS="${TIERS_OVERRIDE:+$(abspath "$TIERS_OVERRIDE")}"
TIERS="${TIERS:-$DATASET_DIR/gt/tiers.csv}"

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
[ -f "$TIERS" ] && TIERS_ARGS=(--tiers "$TIERS")

python3 "$HERE/eval/score.py" \
    --submission "$OUT" \
    --gt "$GT" \
    --track A \
    --method "$METHOD_LABEL" \
    --out-dir "$OUT_DIR" \
    --no-plots \
    "${TIERS_ARGS[@]}"

echo "[$METHOD_LABEL] timing: $(cat "$OUT_DIR/timing.json")"
