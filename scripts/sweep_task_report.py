#!/usr/bin/env python3
"""Summarize a sweep run by scripts/sweep_task.sh: a prop x seed table and the rates.

The companion to sweep_report.py, which does the same for the pick primitive
alone.  Reads only what the sweep wrote; runs no physics of its own.

A prop passes when it ends within the tolerance of its slot *and* is resting on
the table -- a prop still in the jaws is not placed, wherever its xy reads.  Every
failure is put in one of these buckets, each a distinct mechanism rather than a
guess at one:

``no route``        no arm reaches one of the two ends, so nothing was attempted.
``pick failed``     the prop never came off the table, so nothing downstream ran.
``set down short``  the giving arm could not put it down at the transfer spot.
``not received``    it was set down at the transfer spot, but the receiving arm
                    failed to pick it up again.
``release unreachable``  the arm held it over the slot and the release pose was
                    outside its envelope, so it refused to let go rather than
                    drop the prop somewhere else entirely.  The arm keeps hold,
                    and the prop may still creep out of the jaws afterwards, so
                    this is read off the refusal and not off where it ended up.
``dropped before place``  the prop came off the table but was gone from the jaws
                    by the time the place was planned.
``off target``      it was let go and landed outside the tolerance.
``left held``       the run ended with the prop still in an arm's jaws.

The ``route`` column says which arm did the job, or ``a>b`` for a handover.
"""

import argparse
import collections
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from envs.task import PLACE_ORDER  # noqa: E402

TAGS = {"pass": "ok", "no route": "no-route", "pick failed": "pick",
        "set down short": "putdown", "not received": "receive",
        "release unreachable": "release", "off target": "off",
        "dropped before place": "shed", "left held": "held"}


def _route(step) -> str:
    """The arm that did the job: ``l``/``r``, or ``l>r`` for a handover."""
    arm = (step or {}).get("arm")
    if not arm:
        return "--"
    if ">" in arm:
        giver, taker = arm.split(">")
        return f"{giver[0]}>{taker[0]}"
    return arm[0]


def classify(step, placed):
    """Which bucket a prop's outcome falls in, from the step that produced it."""
    if step is None or step.get("skipped"):
        return "no route"
    if placed["settled"]:
        return "pass"
    if not step["pick"]["lifted"]:
        return "pick failed"
    hand = step.get("handover")
    if hand is not None and not hand["handed"]:
        # A handover that never found a transfer spot has no put-down to report.
        put = hand.get("put_down")
        return "not received" if put and put["placed"] else "set down short"
    # What the place refused on says more than where the prop ended up.  An arm
    # that refuses to release keeps hold, and over the rest of the run the prop
    # can still creep out of the jaws -- so the prop's final position does not
    # tell you which of these happened, but the reason does.
    reason = (step.get("place") or {}).get("reason") or ""
    if "release pose" in reason:
        return "release unreachable"
    if "nothing held" in reason:
        return "dropped before place"
    if placed["held_by"] is not None:
        return "left held"
    return "off target"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out_dir", type=pathlib.Path, nargs="?",
                        default=pathlib.Path(__file__).resolve().parent.parent
                        / ".cache" / "task_sweep")
    args = parser.parse_args()

    runs = {}
    for path in sorted(args.out_dir.glob("task-*.json")):
        seed = path.stem.rsplit("-", 1)[1]
        if not seed.isdigit():
            continue
        text = path.read_text()
        brace = text.find("{")
        if brace < 0:
            tail = (text.strip().splitlines() or ["(empty)"])[-1]
            print(f"  {path.name}: no report ({tail[:70]})")
            continue
        runs[int(seed)] = json.loads(text[brace:])
    if not runs:
        raise SystemExit(f"no sweep output in {args.out_dir}")

    seeds = sorted(runs)
    tol = runs[seeds[0]]["tolerance"]
    print(f"error to slot in mm; a pass needs error <= {tol * 1000:.0f} mm and the"
          f" prop resting on the table\n")
    head = f"{'seed':>4}  " + "".join(f"{p:^18}" for p in PLACE_ORDER)
    print(head)
    print("-" * len(head))

    cells = {}
    for seed in seeds:
        run = runs[seed]
        steps = {s["object"]: s for s in run["steps"]}
        row = f"{seed:>4}  "
        for prop in PLACE_ORDER:
            got = classify(steps.get(prop), run["layout"][prop])
            cells[prop, seed] = got
            err = run["layout"][prop]["error"] * 1000
            shown = f"{err:7.1f}" if err < 1000 else "    big"
            row += f"{shown} {TAGS[got]:<8}{_route(steps.get(prop)):<4}"
        print(row)

    print()
    passes = 0
    for prop in PLACE_ORDER:
        got = [cells[prop, s] for s in seeds]
        n = got.count("pass")
        passes += n
        errs = [runs[s]["layout"][prop]["error"] * 1000 for s in seeds
                if cells[prop, s] == "pass"]
        spread = (f"   placed error median {sorted(errs)[len(errs) // 2]:4.1f} mm,"
                  f" worst {max(errs):4.1f} mm" if errs else "")
        print(f"  {prop:6s} placed {n:2d}/{len(got)} ({n / len(got) * 100:3.0f}%){spread}")

    total = len(seeds) * len(PLACE_ORDER)
    complete = sum(1 for s in seeds if runs[s]["matched"])
    handovers = [(s, st["object"]) for s in seeds for st in runs[s]["steps"]
                 if st.get("handover") is not None]
    handed = [(s, st["object"]) for s in seeds for st in runs[s]["steps"]
              if (st.get("handover") or {}).get("handed")]
    print(f"\n  props placed        {passes:2d}/{total} = {passes / total * 100:.1f}%")
    print(f"  settings complete   {complete:2d}/{len(seeds)}"
          f" = {complete / len(seeds) * 100:.1f}%   (all four props in tolerance)")
    if handovers:
        print(f"  handovers completed {len(handed):2d}/{len(handovers)}"
              f" = {len(handed) / len(handovers) * 100:.1f}%")

    counts = collections.Counter(cells.values())
    print("\nfailure modes:")
    if set(counts) == {"pass"}:
        print("  none")
    for mode, n in counts.most_common():
        if mode == "pass":
            continue
        cases = sorted(f"{p}-{s}" for (p, s), g in cells.items() if g == mode)
        print(f"  {mode:20s} {n:2d}   {', '.join(cases)}")


if __name__ == "__main__":
    main()
