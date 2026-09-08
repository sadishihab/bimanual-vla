#!/usr/bin/env python3
"""Summarize a sweep run by scripts/sweep.sh: a prop x seed table and the rates.

Reads only what the sweep wrote; runs no physics of its own.  Every failure is
put in one of three buckets, each of which is a distinct mechanism rather than a
guess at one:

``approach shove``       the prop was pushed off its spot before the grasp, so
                         the descent aimed where it used to be.  Measured as the
                         prop's horizontal displacement at the end of the
                         approach, before the jaws have touched anything.
``far end never clears``  the grasp held, but the prop is still touching the
                         table.  For cutlery held at the handle this is the far
                         end: it pivots down as the handle goes up.
``lift clamped short``   the prop came up clear of the table and stayed held,
                         but by less than LIFT_THRESHOLD, because that is as far
                         as the arm reaches above that grasp point.
"""

import argparse
import collections
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from control.primitives import LIFT_THRESHOLD  # noqa: E402

SHOVE = 0.005            # m of pre-grasp displacement that counts as a shove
TAGS = {"pass": "PASS", "approach shove": "shove",
        "far end never clears": "far-end", "lift clamped short": "short"}


def classify(report):
    if report["lifted"]:
        return "pass"
    if report["approach_shift"] >= SHOVE:
        return "approach shove"
    if report["contacts"]["table"] > 0:
        return "far end never clears"
    return "lift clamped short"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out_dir", type=pathlib.Path, nargs="?",
                        default=pathlib.Path(__file__).resolve().parent.parent
                        / ".cache" / "sweep")
    args = parser.parse_args()

    runs = {}
    for path in sorted(args.out_dir.glob("*-*.json")):
        prop, seed = path.stem.rsplit("-", 1)
        if not seed.isdigit():
            continue
        text = path.read_text()
        brace = text.find("{")
        if brace < 0:
            print(f"  {path.name}: no report ({text.strip().splitlines()[-1][:60]})")
            continue
        runs[prop, int(seed)] = json.loads(text[brace:])
    if not runs:
        raise SystemExit(f"no sweep output in {args.out_dir}")

    props = sorted({p for p, _ in runs}, key=lambda p: [q for q, _ in runs].index(p))
    seeds = sorted({s for _, s in runs})

    print(f"rise in mm; a pass needs rise > {LIFT_THRESHOLD * 1000:.0f} mm,"
          f" no table contact, and the prop still held\n")
    head = f"{'seed':>4}  " + "".join(f"{p:^17}" for p in props)
    print(head)
    print("-" * len(head))
    for seed in seeds:
        row = f"{seed:>4}  "
        for prop in props:
            r = runs.get((prop, seed))
            row += "       --        " if r is None else \
                f"{r['rise'] * 1000:9.1f} {TAGS[classify(r)]:<7}"
        print(row)

    print()
    lifted = grasped = 0
    for prop in props:
        rows = [runs[prop, s] for s in seeds if (prop, s) in runs]
        n = sum(r["lifted"] for r in rows)
        g = sum(1 for r in rows
                if next(x for x in r["stages"] if x["stage"] == "close")
                ["grip_force"]["moving"] > 1.0)
        lifted, grasped = lifted + n, grasped + g
        print(f"  {prop:6s} lifted {n:2d}/{len(rows)} ({n / len(rows) * 100:3.0f}%)"
              f"   grasp formed {g:2d}/{len(rows)}")
    total = len(runs)
    print(f"\n  overall lifted       {lifted:2d}/{total} = {lifted / total * 100:.1f}%")
    print(f"  overall grasp formed {grasped:2d}/{total} = {grasped / total * 100:.1f}%")

    counts = collections.Counter(classify(r) for r in runs.values())
    print("\nfailure modes:")
    for mode, n in counts.most_common():
        if mode == "pass":
            continue
        cases = sorted(f"{p}-{s}" for (p, s), r in runs.items() if classify(r) == mode)
        print(f"  {mode:22s} {n:2d}   {', '.join(cases)}")


if __name__ == "__main__":
    main()
