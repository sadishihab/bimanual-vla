#!/usr/bin/env python3
"""Run the whole table-setting task on one seed: pick and place all four props.

Each prop is picked with whichever arm can reach it and put down at its place in
the setting, in an order that lays the anchor first.  A prop whose slot is across
the midline from where it lies is routed through a handover -- the arm that can
reach it picks it up, sets it down where both arms can reach, and the arm that
can reach the slot takes it from there.  Reports each step and then whether the
layout that came out matches the goal.

Deliberately minimal: one seed, one process, no renderer and no video.
"""

import argparse
import json
import pathlib
import sys

import mujoco
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from control.primitives import (PLACE_TOL, _table_top_z, arm_clash,  # noqa: E402
                                best_arm, handover, held_object, park, pick, place)
from envs.randomize import PROPS, randomize  # noqa: E402
from envs.scene import SCENE, load_scene  # noqa: E402
from envs.task import (PLACE_ORDER, arms_reaching, check_setting,  # noqa: E402
                       crossings, goal_layout)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True, help="the one seed to run")
    parser.add_argument("--scene", type=pathlib.Path, default=SCENE)
    parser.add_argument("--tolerance", type=float, default=PLACE_TOL,
                        help="m; how close to its goal a prop must land")
    parser.add_argument("--order", default=None,
                        help="comma-separated prop order to work in; default lays the "
                             "anchor first. The order decides what the table looks like "
                             "when each prop's turn comes, which is what a policy "
                             "trained on recordings of this sees")
    parser.add_argument("--dump", action="store_true", help="print the full report as JSON")
    args = parser.parse_args()

    order = tuple(PLACE_ORDER)
    if args.order:
        order = tuple(x.strip() for x in args.order.split(",") if x.strip())
        if sorted(order) != sorted(PLACE_ORDER):
            raise SystemExit(f"--order must be a permutation of {list(PLACE_ORDER)},"
                             f" got {list(order)}")

    model = load_scene(args.scene)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    randomize(model, data, args.seed)
    mujoco.mj_forward(model, data)

    goal = goal_layout(model, data)
    slack = check_setting(model, goal)
    print(f"seed {args.seed}: table setting, tolerance {args.tolerance * 1000:.0f} mm")
    if order != tuple(PLACE_ORDER):
        print(f"  order     {' -> '.join(order)}")
    print(f"  goal      " + "  ".join(f"{n} {_fmt(goal[n])}" for n in PLACE_ORDER))
    print(f"  tightest pair {min(slack, key=slack.get)} clears by"
          f" {min(slack.values()) * 1000:.0f} mm")
    start = {n: data.xpos[_bid(model, n)][:2].copy() for n in PLACE_ORDER}

    # The arms split the table at y = 0, so a prop lying on the far side of its
    # own slot cannot be carried there by either of them alone.  Those props go
    # through a handover instead of being skipped; saying which up front still
    # beats working it out from the step lines.
    blocked = crossings(model, data, goal)
    for name, (here, there) in blocked.items():
        reach = f"{'/'.join(here) or 'no arm'} reaches it where it lies, " \
                f"{'/'.join(there) or 'no arm'} reaches its slot"
        how = "handover" if (here and there) else "unreachable at one end"
        print(f"  {name} crosses the midline: {reach}  ->  {how}")

    radius = {s.name: s.radius for s in PROPS}
    steps = []
    print("\n  step                     pick                            place")
    for name in order:
        if name in blocked:
            here, there = blocked[name]
            if not here or not there:
                steps.append({"object": name, "arm": None, "pick": None, "place": None,
                              "skipped": "no arm reaches one of the two ends"})
                print(f"  {name:6s} --     skipped, no arm reaches"
                      f" {'it where it lies' if not here else 'its slot'}")
                continue
            # Blocked means the two arm sets are disjoint, so the arm that can
            # reach the slot is necessarily the other one.  Everything already
            # spoken for -- the other props' slots -- is kept clear of the spot
            # the prop is put down on in between.
            from_arm, to_arm = here[0], there[0]
            avoid = [(goal[o][0], goal[o][1], radius[o]) for o in goal if o != name]
            got = pick(model, data, from_arm, name)
            hand = handover(model, data, from_arm, to_arm, avoid=avoid) \
                if got["lifted"] else None
            put = (place(model, data, to_arm, goal[name])
                   if hand is not None and hand["handed"]
                   else _no_place(to_arm, name, hand, got))
            steps.append({"object": name, "arm": f"{from_arm}>{to_arm}", "pick": got,
                          "handover": hand, "place": put})
            _report(name, f"{from_arm[0]}>{to_arm[0]}", got, put, hand)
            _clear(model, data, to_arm)
            continue

        # One arm does both halves, so it has to be an arm that can reach the goal
        # as well as the prop.
        #
        # This is asked again here rather than taken from the crossings computed
        # before anything moved, and it can come back empty even for a prop those
        # crossings called movable.  Reachability is answered against the prop's
        # *current* orientation, because the tool site's offset from the prop
        # rotates with it, and an earlier step can nudge a prop: measured on seed
        # 48, turning the fork 90 degrees takes its slot from both arms to the
        # right arm only.  The flatware has a single grasp yaw and so only two
        # site offsets, which is what makes it fragile that way; the round plate
        # and mug have fourteen or more and keep their answer.  Left unhandled
        # this reached best_arm with nothing to choose from and crashed the run.
        reaching = arms_reaching(model, data, name, goal[name])
        if not reaching:
            steps.append({"object": name, "arm": None, "pick": None, "place": None,
                          "skipped": "no arm reaches its slot any more"})
            print(f"  {name:6s} --     skipped, no arm reaches its slot any more"
                  f" (it was turned by an earlier step)")
            continue
        arm = best_arm(model, data, name, arms=reaching)[0]
        got = pick(model, data, arm, name)
        put = place(model, data, arm, goal[name])
        steps.append({"object": name, "arm": arm, "pick": got, "place": put})
        _report(name, arm, got, put, None)
        _clear(model, data, arm)

    print("\n  final layout")
    ok = True
    layout = {}
    for name in PLACE_ORDER:
        bid = _bid(model, name)
        at = data.xpos[bid][:2]
        err = float(np.linalg.norm(at - goal[name]))
        moved = float(np.linalg.norm(at - start[name]))
        # A prop still in the jaws is not on the table, wherever its xy reads:
        # a place that refused to fly leaves it hanging, and reporting that as a
        # position on the table would be the report lying about the layout.
        holder = next((a for a in ("left", "right")
                       if held_object(model, data, a) == bid), None)
        settled = err <= args.tolerance and holder is None
        ok &= settled
        layout[name] = {"at": at.tolist(), "error": err, "moved": moved,
                        "held_by": holder, "settled": bool(settled),
                        "z_above_table": float(data.xpos[bid][2] - _table_top_z(model))}
        print(f"  {name:6s} at {_fmt(at)}  goal {_fmt(goal[name])}"
              f"  err {err * 1000:6.1f} mm  moved {moved * 1000:6.1f} mm"
              f"  {'ok' if settled else 'OUT OF TOLERANCE'}"
              f"{'' if holder is None else f'  (still in the {holder} jaws, {(data.xpos[bid][2] - _table_top_z(model)) * 1000:.0f} mm up)'}")

    tried = [s for s in steps if s.get("pick") is not None]
    picked = sum(s["pick"]["lifted"] for s in tried)
    placed = sum(s["place"]["placed"] for s in tried)
    handed = [s for s in tried if s.get("handover") is not None]
    clash, deepest = arm_clash(model, data)
    print(f"\n  attempted {len(tried)}/{len(steps)}   picks {picked}/{len(tried)}"
          f"   places {placed}/{len(tried)}"
          f"   handovers {sum(s['handover']['handed'] for s in handed)}/{len(handed)}"
          f"   layout matches goal: {ok}")
    if clash:
        print(f"  the arms are touching: {clash} contacts,"
              f" {deepest * 1000:.1f} mm deepest")

    if args.dump:
        print(json.dumps({"seed": args.seed,
                          "order": list(order),
                          "tolerance": args.tolerance,
                          "goal": {k: v.tolist() for k, v in goal.items()},
                          "blocked": {k: [list(v[0]), list(v[1])]
                                      for k, v in blocked.items()},
                          "steps": steps, "layout": layout, "matched": ok},
                         indent=2, sort_keys=True, default=float))
    raise SystemExit(0 if ok else 1)


def _clear(model, data, arm: str) -> None:
    """Send ``arm`` home once it has let go, so it is not standing over the table.

    Nothing in the corridor checks looks at the other arm -- they ask only about
    props -- and both arms work the same square of table here.  Measured on seed
    0: with the left arm left hovering over the plate's slot, the right arm's
    approach to the fork ended 11.6 mm off its standoff and its descent 15.6 mm
    off the grasp, driving the fixed finger into the fork at 10.3 N and shoving it
    12.6 mm before the jaws closed on nothing.  :func:`control.primitives.handover`
    already parks the giving arm for the same reason; this does it after every
    step rather than only in the middle of one.

    An arm that still has hold of something is left alone: parking it would carry
    the prop away and drop it wherever the keyframe points.
    """
    if held_object(model, data, arm) is None:
        park(model, data, arm)


def _no_place(arm: str, name: str, hand, got) -> dict:
    """The place that never happened, in the shape the report expects."""
    why = "pick failed" if not got["lifted"] else (hand or {}).get("reason") or "handover failed"
    return {"arm": arm, "object": name, "placed": False, "reason": why,
            "error": float("inf"), "fouls": 0, "final_xy": None}


def _report(name, arm, got, put, hand) -> None:
    note = ""
    if hand is not None:
        note = (f"  via {_fmt(hand['spot'])}"
                f"{'' if hand.get('spot_pickable', True) else ' (not pickable)'}"
                if hand.get("spot") is not None else "  no transfer spot")
    print(f"  {name:6s} {arm:5s}  "
          f"rise {got['rise'] * 1000:+6.1f} mm {'lifted ' if got['lifted'] else 'FAILED '}"
          f"  |  err {put['error'] * 1000:6.1f} mm"
          f" {'placed' if put['placed'] else 'FAILED'}"
          f"  fouls {put['fouls']:2d}"
          f"{'' if put['placed'] else '  (' + str(put.get('reason', 'off target')) + ')'}"
          f"{note}")


def _bid(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def _fmt(v) -> str:
    return "[" + ", ".join(f"{x:+.3f}" for x in v) + "]"


if __name__ == "__main__":
    main()
