#!/bin/sh
# Evaluate the trained policy in closed loop over a range of seeds.
#
# One seed per process, as the rule here requires, and strictly one at a time:
# unlike the scripted sweeps there is nothing to gain from parallelism, because
# both halves of the work -- rendering the observation and running the policy --
# serialize on this machine anyway.
#
# Each seed's report goes to $OUT/eval-<seed>.json.
#
# usage: scripts/eval_sweep.sh [out_dir] [seeds...]
set -e
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
PY=${PY:-"$HOME/.cache/lerobot-verify/venv/bin/python"}   # needs mujoco AND lerobot
OUT=${1:-"$ROOT/.cache/eval"}
if [ $# -gt 0 ]; then shift; fi
SEEDS=${*:-"0 1 2 3 4 5 6 7 8 9"}
CKPT=${CKPT:-"$ROOT/checkpoints/step_0044000"}
export MUJOCO_GL=${MUJOCO_GL:-egl}

mkdir -p "$OUT"
for seed in $SEEDS; do
  # A non-zero exit just means it did not place all four, which is the point.
  "$PY" "$ROOT/scripts/eval_policy.py" --seed "$seed" --checkpoint "$CKPT" --dump \
    > "$OUT/eval-$seed.json" 2>&1 || true
  printf '.'
done
echo " wrote $OUT"
