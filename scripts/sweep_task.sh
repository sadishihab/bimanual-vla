#!/bin/sh
# Run the whole table-setting task over a range of seeds, one process per seed.
#
# One seed per process is the rule in this repo, so the loop lives here in the
# shell rather than inside the Python: each case is a fresh interpreter with a
# fresh model, and nothing carries over between them.  Each seed's full report is
# written to $OUT/task-<seed>.json for scripts/sweep_task_report.py to summarize.
#
# The companion to sweep.sh, which sweeps the pick primitive alone.  A seed here
# is a whole four-prop setting, handovers included, so it costs minutes rather
# than seconds; JOBS seeds run at once, in batches.  Nothing this calls builds a
# renderer.
#
# usage: scripts/sweep_task.sh [out_dir] [seeds...]
set -e
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
PY=${PY:-"$ROOT/.venv/bin/python"}
OUT=${1:-"$ROOT/.cache/task_sweep"}
if [ $# -gt 0 ]; then shift; fi
SEEDS=${*:-"0 1 2 3 4 5 6 7 8 9"}
JOBS=${JOBS:-4}

mkdir -p "$OUT"
n=0
for seed in $SEEDS; do
  # A non-zero exit just means the layout missed, which is data, not an error.
  { "$PY" "$ROOT/scripts/run_task.py" --seed "$seed" --dump \
      > "$OUT/task-$seed.json" 2>&1 || true; } &
  n=$((n + 1))
  if [ "$n" -ge "$JOBS" ]; then wait; n=0; fi
done
wait
echo "wrote $OUT"
