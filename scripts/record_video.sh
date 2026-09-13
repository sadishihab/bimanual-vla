#!/bin/sh
# Render the demonstration video: one mp4 segment per seed, then concatenate.
#
# One seed per process, as the rule here requires, and strictly one at a time.  That
# is not only the rule: rendering does not parallelise on this machine, and the point
# of doing it serially is to keep the footprint to a single interpreter while the
# retraining recorder is running alongside.
#
# usage: scripts/record_video.sh [out_dir] [seeds...]
set -e
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
PY=${PY:-"$ROOT/.venv/bin/python"}
OUT=${1:-"$ROOT/.cache/video"}
if [ $# -gt 0 ]; then shift; fi
# Seed 2 first: it has a hand-off, so the part most worth checking appears early.
SEEDS=${*:-"2 0 1 3 4 5 6 7 8 9"}
export MUJOCO_GL=${MUJOCO_GL:-egl}

mkdir -p "$OUT"
for seed in $SEEDS; do
  "$PY" "$ROOT/scripts/record_video.py" --seed "$seed" --out "$OUT"
done
"$PY" "$ROOT/scripts/record_video.py" --summary --out "$OUT" \
  --placed "${PLACED:-24}" --total "${TOTAL:-40}" --median-mm "${MEDIAN_MM:-1.6}"

# Concatenate in seed order, summary last.  All segments share codec and geometry,
# so this is a stream copy and re-encodes nothing.
LIST="$OUT/segments.txt"
: > "$LIST"
for f in $(ls "$OUT"/seg-*.mp4 | sort); do
  printf "file '%s'\n" "$(basename "$f")" >> "$LIST"
done
ffmpeg -y -hide_banner -loglevel error -f concat -safe 0 -i "$LIST" -c copy "$OUT/demo.mp4"
echo "wrote $OUT/demo.mp4"
ffprobe -v error -show_entries format=duration,size -of default=nw=1 "$OUT/demo.mp4"
