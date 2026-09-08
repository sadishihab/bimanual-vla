#!/bin/sh
# Run the pick primitive over a grid of props and seeds, one process per case.
#
# One seed per process is the rule in this repo, so the loop lives here in the
# shell rather than inside the Python: each case is a fresh interpreter with a
# fresh model, and nothing carries over between them.  Each case's full report is
# written to $OUT/<prop>-<seed>.json for scripts/sweep_report.py to summarize.
#
# usage: scripts/sweep.sh [out_dir] [seeds...]
set -e
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
PY=${PY:-"$ROOT/.venv/bin/python"}
OUT=${1:-"$ROOT/.cache/sweep"}
if [ $# -gt 0 ]; then shift; fi
SEEDS=${*:-"0 1 2 3 4 5 6 7 8 9"}
PROPS=${PROPS:-"plate mug spoon fork"}

mkdir -p "$OUT"
for seed in $SEEDS; do
  for prop in $PROPS; do
    "$PY" "$ROOT/scripts/test_pick.py" --seed "$seed" --object "$prop" --dump \
      > "$OUT/$prop-$seed.json" 2>&1 || true
    printf '.'
  done
  printf ' seed %s\n' "$seed"
done
echo "wrote $OUT"
