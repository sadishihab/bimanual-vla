#!/usr/bin/env python3
"""Run the whole table-setting task on one seed: pick and place all four props.

Each prop is picked with whichever arm can reach it and put down at its place in
the setting, in an order that lays the anchor first.  Reports each step and then
whether the layout that came out matches the goal.

Deliberately minimal: one seed, one process, no renderer and no video.
"""

import argparse
import json
import pathlib
import sys

import mujoco
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from control.primitives import PLACE_TOL, best_arm, pick, place  # noqa: E402
from envs.randomize import randomize  # noqa: E402
from envs.scene import SCENE, load_scene  # noqa: E402
from envs.task import (PLACE_ORDER, arms_reaching, check_setting,  # noqa: E402
                       crossings, goal_layout)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True, help="the one seed to run")
    parser.add_argument("--scene", type=pathlib.Path, default=SCENE)
    parser.add_argument("--tolerance", type=float, default=PLACE_TOL,
                        help="m; how close to its goal a prop must land")
    parser.add_argument("--dump", action="store_true", help="print the full report as JSON")
    args = parser.parse_args()

    model = load_scene(args.scene)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    randomize(model, data, args.seed)
    mujoco.mj_forward(model, data)

    goal = goal_layout(model, data)
    slack = check_setting(model, goal)
    print(f"seed {args.seed}: table setting, tolerance {args.tolerance * 1000:.0f} mm")
    print(f"  goal      " + "  ".join(f"{n} {_fmt(goal[n])}" for n in PLACE_ORDER))
    print(f"  tightest pair {min(slack, key=slack.get)} clears by"
          f" {min(slack.values()) * 1000:.0f} mm")
    start = {n: data.xpos[_bid(model, n)][:2].copy() for n in PLACE_ORDER}

    # The arms split the table at y = 0 and nothing hands a prop across it, so a
    # prop lying on the far side of its own slot cannot be moved by either of
    # them.  Saying so up front beats watching it fail four times.
    blocked = crossings(model, data, goal)
    for name, (here, there) in blocked.items():
        print(f"  {name} cannot be moved: {'/'.join(here)} reaches it where it lies,"
              f" {'/'.join(there)} reaches its slot")

    steps = []
    print("\n  step                     pick                            place")
    for name in PLACE_ORDER:
        if name in blocked:
            steps.append({"object": name, "arm": None, "pick": None, "place": None,
                          "skipped": "no arm reaches both ends"})
            print(f"  {name:6s} --     skipped, no arm reaches both ends")
            continue
        # One arm does both halves, so it has to be an arm that can reach the goal
        # as well as the prop.  Nothing here hands a prop over between arms.
        reaching = arms_reaching(model, data, name, goal[name])
        arm = best_arm(model, data, name, arms=reaching)[0]
        got = pick(model, data, arm, name)
        put = place(model, data, arm, goal[name])
        steps.append({"object": name, "arm": arm, "pick": got, "place": put})
        print(f"  {name:6s} {arm:5s}  "
              f"rise {got['rise'] * 1000:+6.1f} mm {'lifted ' if got['lifted'] else 'FAILED '}"
              f"  |  err {put['error'] * 1000:6.1f} mm"
              f" {'placed' if put['placed'] else 'FAILED'}"
              f"  fouls {put['fouls']:2d}"
              f"{'' if put['placed'] else '  (' + str(put.get('reason', 'off target')) + ')'}")

    print("\n  final layout")
    ok = True
    for name in PLACE_ORDER:
        at = data.xpos[_bid(model, name)][:2]
        err = float(np.linalg.norm(at - goal[name]))
        moved = float(np.linalg.norm(at - start[name]))
        ok &= err <= args.tolerance
        print(f"  {name:6s} at {_fmt(at)}  goal {_fmt(goal[name])}"
              f"  err {err * 1000:6.1f} mm  moved {moved * 1000:6.1f} mm"
              f"  {'ok' if err <= args.tolerance else 'OUT OF TOLERANCE'}")

    tried = [s for s in steps if s.get("pick") is not None]
    picked = sum(s["pick"]["lifted"] for s in tried)
    placed = sum(s["place"]["placed"] for s in tried)
    print(f"\n  attempted {len(tried)}/{len(steps)}   picks {picked}/{len(tried)}"
          f"   places {placed}/{len(tried)}   layout matches goal: {ok}")

    if args.dump:
        print(json.dumps({"seed": args.seed,
                          "goal": {k: v.tolist() for k, v in goal.items()},
                          "steps": steps, "matched": ok},
                         indent=2, sort_keys=True, default=float))
    raise SystemExit(0 if ok else 1)


def _bid(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def _fmt(v) -> str:
    return "[" + ", ".join(f"{x:+.3f}" for x in v) + "]"


if __name__ == "__main__":
    main()
