#!/usr/bin/env python3
"""Run the scripted pick primitive on a single seed and report whether it lifted.

Deliberately minimal: one seed, one process, no renderer and no video.
"""

import argparse
import json
import pathlib
import sys

import mujoco

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from control.primitives import pick  # noqa: E402
from envs.randomize import randomize  # noqa: E402
from envs.scene import SCENE, load_scene  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True, help="the one seed to run")
    parser.add_argument("--arm", choices=("left", "right"), default="right")
    parser.add_argument("--object", default="plate")
    parser.add_argument("--scene", type=pathlib.Path, default=SCENE)
    parser.add_argument("--dump", action="store_true", help="print the full report as JSON")
    args = parser.parse_args()

    model = load_scene(args.scene)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    randomize(model, data, args.seed)
    mujoco.mj_forward(model, data)

    report = pick(model, data, args.arm, args.object)

    print(f"seed {args.seed}: {args.arm} arm picking {args.object}")
    print(f"  grasp target   {_fmt(report['grasp_target'])}  yaw {report['grasp_yaw']:+.3f} rad")
    print(f"  grasp width    {report['geometry']['grasp_width'] * 1000:.1f} mm")
    for stage in report["stages"]:
        line = f"  {stage['stage']:<9}"
        if "ik" in stage:
            ik = stage["ik"]
            line += (f" ik pos_err {ik['pos_err'] * 1000:6.2f} mm"
                     f"  rot_err {ik['rot_err']:5.3f} rad"
                     f"  {'ok' if ik['converged'] else 'NOT CONVERGED'}")
            line += f" | site_err {stage['site_err'] * 1000:6.2f} mm"
        else:
            line += " " * 58
        c = stage["contacts"]
        g = stage["grip_force"]
        line += (f" | obj z {stage['object_z']:.4f} contacts table={c['table']} arm={c['arm']}"
                 f" | jaw {stage['jaw_gap'] * 1000:5.1f} mm"
                 f" grip {g['fixed']:6.2f}/{g['moving']:6.2f} N")
        print(line)

    if report["lift_trace"]:
        print("  lift trace (grip force per Cartesian sub-step, fixed/moving jaw):")
        for t in report["lift_trace"]:
            g = t["grip_force"]
            print(f"    step {t['step']:2d}  +{t['height'] * 1000:5.1f} mm"
                  f"  obj z {t['object_z']:.4f}  jaw {t['jaw_gap'] * 1000:5.1f} mm"
                  f"  grip {g['fixed']:6.2f} / {g['moving']:6.2f} N")
    print(f"  rest z {report['rest_z']:.4f} -> final z {report['final_z']:.4f}"
          f"  (rise {report['rise'] * 1000:+.1f} mm)")
    print(f"  final contacts table={report['contacts']['table']} arm={report['contacts']['arm']}")
    print(f"  LIFTED: {report['lifted']}")

    if args.dump:
        print(json.dumps(report, indent=2, sort_keys=True, default=float))

    raise SystemExit(0 if report["lifted"] else 1)


def _fmt(v) -> str:
    return "[" + ", ".join(f"{x:+.4f}" for x in v) + "]"


if __name__ == "__main__":
    main()
