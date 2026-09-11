#!/bin/sh
# Record the scripted expert over a range of seeds, one process per seed.
#
# One seed per process is the rule in this repo, so the loop lives here in the
# shell rather than inside the Python.  Each seed stages its successful episodes
# under $OUT/seed-<NNN>/<prop>/ for scripts/pack_lerobot.py to assemble.
#
# Rendering is the bottleneck and it does not parallelise: measured on this
# machine, one process renders a 256 px camera pair in 52 ms and four concurrent
# processes take 200 ms each, so total render throughput is ~19 pairs/s however
# many run at once.  JOBS is still worth more than 1 because the expert's
# planning -- IK, reach maps, the handover's meeting-pose search -- does
# parallelise, and on a handover seed that dominates.
#
# usage: scripts/record_demos.sh [out_dir] [seeds...]
set -e
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
PY=${PY:-"$ROOT/.venv/bin/python"}
OUT=${1:-"$ROOT/.cache/demos"}
if [ $# -gt 0 ]; then shift; fi
SEEDS=${*:-$(seq 0 49)}
JOBS=${JOBS:-4}
export MUJOCO_GL=${MUJOCO_GL:-egl}

mkdir -p "$OUT"
n=0
for seed in $SEEDS; do
  # A seed whose props all failed records nothing, which is data, not an error.
  { "$PY" "$ROOT/scripts/record_demos.py" --seed "$seed" --out "$OUT" \
      || echo "seed $seed: FAILED"; } &
  n=$((n + 1))
  if [ "$n" -ge "$JOBS" ]; then wait; n=0; fi
done
wait
echo "staged in $OUT"
