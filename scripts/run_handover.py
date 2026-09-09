#!/usr/bin/env python3
"""Hand one prop from one arm to the other on a single seed, and report each step.

The arms split the table at y = 0, so a prop that starts on one side of a slot on
the other side cannot be moved by either arm alone.  This runs the whole move for
one such prop: the arm that can reach it picks it up, hands it over, and the
receiving arm places it in its slot.

Deliberately minimal: one seed, one process, no renderer and no video.
"""

import argparse
import json
import pathlib
import sys

import mujoco
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from control.primitives import handover, pick, place  # noqa: E402
from envs.randomize import PROPS, randomize  # noqa: E402
from envs.scene import SCENE, load_scene  # noqa: E402
from envs.task import arms_reaching, crossings, goal_layout  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True, help="the one seed to run")
    parser.add_argument("--object", default=None,
                        help="which prop to hand over; default is the first that crosses")
    parser.add_argument("--scene", type=pathlib.Path, default=SCENE)
    parser.add_argument("--dump", action="store_true", help="print the full report as JSON")
    args = parser.parse_args()

    model = load_scene(args.scene)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    randomize(model, data, args.seed)
    mujoco.mj_forward(model, data)

    goal = goal_layout(model, data)
    blocked = crossings(model, data, goal)
    print(f"seed {args.seed}: props that no single arm can both pick up and put down")
    for name, (here, there) in blocked.items():
        bid = _bid(model, name)
        print(f"  {name:6s} at {_fmt(data.xpos[bid][:2])} ({'/'.join(here)} reaches it)"
              f"  ->  slot {_fmt(goal[name])} ({'/'.join(there)} reaches that)")
    if not blocked:
        print("  none: every prop's pick and place are on one arm")

    name = args.object or (next(iter(blocked)) if blocked else "fork")
    bid = _bid(model, name)
    here = arms_reaching(model, data, name, data.xpos[bid][:2])
    there = arms_reaching(model, data, name, goal[name])
    if not here:
        print(f"\nno arm reaches {name} where it lies; nothing to hand over")
        raise SystemExit(1)
    from_arm = here[0]
    to_arm = ({"left", "right"} - {from_arm}).pop()
    print(f"\nhanding {name} from the {from_arm} arm to the {to_arm} arm"
          f"  ({'/'.join(there) if there else 'no arm'} reaches its slot)")

    radius = {s.name: s.radius for s in PROPS}
    avoid = [(goal[o][0], goal[o][1], radius[o]) for o in goal if o != name]

    got = pick(model, data, from_arm, name)
    print(f"\n  1. pick   {from_arm:5s}  rise {got['rise'] * 1000:+6.1f} mm"
          f"  {'lifted' if got['lifted'] else 'FAILED'}"
          f"  contacts table={got['contacts']['table']} arm={got['contacts']['arm']}")
    if not got["lifted"]:
        raise SystemExit(1)

    hand = handover(model, data, from_arm, to_arm, avoid=avoid)
    m = hand["meeting"]
    print(f"\n  2. handover")
    print(f"     in-air pose: {'found' if m.get('found') else 'none'}"
          f"   ({m.get('reason', '')})")
    if not m.get("found"):
        print(f"       shared cells {m.get('shared_cells')},"
              f" grasp feature {m.get('feature_length', 0) * 1000:.0f} mm,"
              f" shallowest clash {(m.get('deepest') or 0) * 1000:.1f} mm"
              f" over {m.get('clash')} contacts")
    if hand.get("spot") is None:
        print(f"     FAILED: {hand['reason']}")
        raise SystemExit(1)
    print(f"     transfer spot {_fmt(hand['spot'])}")
    put = hand["put_down"]
    print(f"     put down {hand['from']:5s}  err {put['error'] * 1000:6.1f} mm"
          f"  {'placed' if put['placed'] else 'FAILED'}"
          f"  fouls {put['fouls']:2d}  regrips {put.get('regrips', 0)}")
    up = hand["picked_up"]
    if up is None:
        print(f"     pick up  {hand['to']:5s}  not attempted ({hand['reason']})")
    else:
        print(f"     pick up  {hand['to']:5s}  rise {up['rise'] * 1000:+6.1f} mm"
              f"  {'lifted' if up['lifted'] else 'FAILED'}")
    print(f"     HANDED: {hand['handed']}")
    if not hand["handed"]:
        raise SystemExit(1)

    final = place(model, data, to_arm, goal[name])
    print(f"\n  3. place  {to_arm:5s}  err {final['error'] * 1000:6.1f} mm"
          f"  {'placed' if final['placed'] else 'FAILED'}"
          f"  fouls {final['fouls']:2d}  regrips {final.get('regrips', 0)}"
          f"{'' if final['placed'] else '  (' + str(final.get('reason', 'off target')) + ')'}")

    at = data.xpos[bid][:2]
    err = float(np.linalg.norm(at - goal[name]))
    print(f"\n  {name} ended at {_fmt(at)}, slot {_fmt(goal[name])}, err {err * 1000:.1f} mm")

    if args.dump:
        print(json.dumps({"seed": args.seed, "object": name, "pick": got,
                          "handover": hand, "place": final},
                         indent=2, sort_keys=True, default=float))
    raise SystemExit(0 if final["placed"] else 1)


def _bid(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def _fmt(v) -> str:
    return "[" + ", ".join(f"{x:+.3f}" for x in v) + "]"


if __name__ == "__main__":
    main()
