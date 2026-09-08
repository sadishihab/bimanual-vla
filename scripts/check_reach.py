#!/usr/bin/env python3
"""Verify that every prop placed by one seed admits a top-down IK solve.

Checks each prop at two heights: the grasp height the reach map is built at
(``PropSpec.grasp_z``, which is what the randomizer guarantees) and the height
``control.primitives.grasp_pose`` actually commands.  One seed per process.
"""

import argparse
import pathlib
import sys

import mujoco
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from control.ik import IKSolver, solve_top_down  # noqa: E402
from control.primitives import grasp_pose  # noqa: E402
from envs.randomize import IK_POS_TOL, IK_ROT_TOL, PROPS, randomize  # noqa: E402

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SCENE = REPO_ROOT / "scenes" / "bimanual_table.xml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True, help="the one seed to check")
    parser.add_argument("--scene", type=pathlib.Path, default=SCENE)
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(args.scene))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    randomize(model, data, args.seed)
    mujoco.mj_forward(model, data)

    solvers = {arm: IKSolver(model, arm) for arm in ("left", "right")}
    ok = True
    print(f"seed {args.seed}")
    for spec in PROPS:
        pos, yaw, _ = grasp_pose(model, data, spec.name)
        for label, z in (("map", spec.grasp_z), ("primitive", pos[2])):
            target = np.array([pos[0], pos[1], z])
            results = {}
            for arm, solver in solvers.items():
                _, info = solve_top_down(solver, data, target, yaw,
                                         pos_tol=IK_POS_TOL, rot_tol=IK_ROT_TOL)
                results[arm] = info
            if spec.reach == "both":
                passed = all(i["converged"] for i in results.values())
            else:
                passed = any(i["converged"] for i in results.values())
            ok &= passed
            detail = "  ".join(
                f"{arm}: {i['pos_err'] * 1000:5.1f} mm/{i['rot_err']:.3f} rad"
                f" {'ok ' if i['converged'] else 'no '}"
                for arm, i in results.items())
            print(f"  {spec.name:6s} {label:9s} z={z:.3f} need={spec.reach:4s}"
                  f" [{'PASS' if passed else 'FAIL'}]  {detail}")
    print(f"  seed {args.seed}: {'all props reachable top-down' if ok else 'FAILURES above'}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
