#!/usr/bin/env python3
"""Run the trained ACT policy in closed loop on the table-setting scene.

    MUJOCO_GL=egl scripts/eval_policy.py --seed 0 --checkpoint checkpoints/step_0044000

The policy drives the actuators directly: its twelve outputs are written to
``data.ctrl`` and held for one control period, in place of the scripted
primitives.  Scoring is the same test scripts/run_task.py applies -- a prop counts
only if it ends within the tolerance of its slot *and* resting on the table -- so
the numbers are comparable with the expert's sweep.

Two facts about this checkpoint shape the evaluation, and neither is a choice made
here:

**The policy is not task-conditioned.**  Its inputs are the two camera images and
the twelve joint positions; the dataset's task strings were recorded but ACT never
consumes them.  So it cannot be told which prop to move, and the only honest
evaluation is to start it from the randomized layout and score whatever it does to
all four props.  That is the same 4-props-per-seed scoring the expert gets.

**It executes 100 actions per inference.**  ``n_action_steps`` is 100, four seconds
at 25 Hz, so the loop is open between inferences by the checkpoint's own
configuration.  ``--horizon`` shortens it for comparison.  Observations are only
needed when an inference happens, which is why this renders a handful of frames per
seed rather than one per control step.

A renderer is unavoidable -- the policy's observation *is* camera RGB -- so exactly
one offscreen renderer is built, shared by both cameras, and freed before exit.  No
video is written and no window is opened.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import mujoco
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "openvino"))
from act_io import ACTInference, build_policy, dataset_meta  # noqa: E402
from control.primitives import (LIFT_THRESHOLD, _object_contacts,  # noqa: E402
                                _table_top_z, prop_bodies)
from envs.randomize import randomize  # noqa: E402
from envs.scene import SCENE, load_scene  # noqa: E402
from envs.task import PLACE_ORDER, check_setting, crossings, goal_layout  # noqa: E402

CAMERAS = ("observation.images.overhead", "observation.images.front")
CAMERA_NAMES = ("overhead", "front")
RESOLUTION = 256
CONTROL_HZ = 25          # the rate the demonstrations were recorded at
SETTLE_TIME = 0.5        # s, so the props drop the couple of mm they spawn above
GRIP_CONTACT = 1         # arm contacts on a prop that count as "the jaws are on it"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    repo = pathlib.Path(__file__).resolve().parent.parent
    parser.add_argument("--seed", type=int, required=True, help="the one seed to run")
    parser.add_argument("--checkpoint", type=pathlib.Path,
                        default=repo / "checkpoints" / "step_0044000")
    parser.add_argument("--dataset", type=pathlib.Path,
                        default=repo / ".cache" / "lerobot" / "bimanual_table_setting")
    parser.add_argument("--repo-id", default="local/bimanual_table_setting")
    parser.add_argument("--scene", type=pathlib.Path, default=SCENE)
    parser.add_argument("--tolerance", type=float, default=0.020,
                        help="m; the scripted sweep's tolerance, for comparability")
    parser.add_argument("--seconds", type=float, default=60.0,
                        help="sim seconds of policy control; the expert takes 50-120 s "
                             "for a whole seed")
    parser.add_argument("--horizon", type=int, default=None,
                        help="actions executed per inference; default is the "
                             "checkpoint's n_action_steps")
    parser.add_argument("--dump", action="store_true")
    args = parser.parse_args()

    model = load_scene(args.scene)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    randomize(model, data, args.seed)
    mujoco.mj_forward(model, data)

    goal = goal_layout(model, data)
    check_setting(model, goal)
    blocked = crossings(model, data, goal)

    # Settle before handing over control, exactly as the expert's pick does.
    for _ in range(int(round(SETTLE_TIME / model.opt.timestep))):
        mujoco.mj_step(model, data)

    bid = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in PLACE_ORDER}
    start = {n: data.xpos[b][:2].copy() for n, b in bid.items()}
    rest_z = {n: float(data.xpos[b][2]) for n, b in bid.items()}
    table_z = _table_top_z(model)

    meta = dataset_meta(args.dataset if (args.dataset / "meta" / "info.json").exists()
                        else None, args.repo_id)
    policy, cfg, stats, source = build_policy(
        args.checkpoint, meta, chunk=100, cameras=list(CAMERAS),
        resolution=RESOLUTION, state_dim=12)
    module = ACTInference(policy, stats, list(CAMERAS)).eval()
    horizon = args.horizon or int(cfg.n_action_steps)

    qadr, act_ids = [], np.arange(model.nu)
    from control.ik import ARM_JOINTS
    for arm in ("left", "right"):
        for j in (*ARM_JOINTS, "gripper"):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{arm}_{j}")
            qadr.append(model.jnt_qposadr[jid])
    qadr = np.asarray(qadr)
    lo = model.actuator_ctrlrange[:, 0].copy()
    hi = model.actuator_ctrlrange[:, 1].copy()

    print(f"seed {args.seed}: closed-loop ACT, tolerance {args.tolerance * 1000:.0f} mm")
    print(f"  checkpoint     : {args.checkpoint}")
    print(f"  statistics     : {source}")
    print(f"  horizon        : {horizon} actions per inference"
          f" ({horizon / CONTROL_HZ:.1f} s open loop)")
    print(f"  budget         : {args.seconds:.0f} s of sim"
          f" = {int(args.seconds * CONTROL_HZ)} control steps")
    print(f"  goal           " + "  ".join(f"{n} {_fmt(goal[n])}" for n in PLACE_ORDER))
    if blocked:
        print(f"  the expert would hand over: {', '.join(sorted(blocked))}")

    renderer = mujoco.Renderer(model, height=RESOLUTION, width=RESOLUTION)
    per_step = max(1, int(round(1.0 / (CONTROL_HZ * model.opt.timestep))))
    total_steps = int(args.seconds * CONTROL_HZ)

    # Behaviour tracking: enough to say what it did, not merely that it failed.
    watch = {n: {"max_shift": 0.0, "max_rise": 0.0, "gripped": 0, "lifted": False,
                 "off_table": False} for n in PLACE_ORDER}
    # The whole evaluation is only as good as the observation.  A renderer that
    # silently hands back black frames would still produce plausible-looking motion
    # -- the policy would just emit its mean action -- so the first frames are
    # checked against what the dataset's own image statistics say to expect.
    seen_stats = None
    joint_travel = np.zeros(12)
    prev_q = data.qpos[qadr].copy()
    chunk, inferences, infer_s = None, 0, 0.0

    try:
        for step in range(total_steps):
            if chunk is None or step % horizon == 0:
                obs = [torch.from_numpy(data.qpos[qadr].astype(np.float32)[None])]
                for cam in CAMERA_NAMES:
                    renderer.update_scene(data, camera=cam)
                    px = renderer.render().astype(np.float32) / 255.0
                    obs.append(torch.from_numpy(px.transpose(2, 0, 1)[None]))
                if seen_stats is None:
                    seen_stats = [(float(o.mean()), float(o.std())) for o in obs[1:]]
                    for cam, (m, sd) in zip(CAMERA_NAMES, seen_stats):
                        want = stats.get(f"observation.images.{cam}", {}).get("mean")
                        want = float(np.asarray(want).mean()) if want is not None else None
                        note = "" if want is None else f", dataset mean {want:.3f}"
                        print(f"  rendered {cam:8s}: mean {m:.3f} std {sd:.3f}{note}")
                        if sd < 1e-3:
                            raise SystemExit(
                                f"the {cam} render is a constant image (std {sd:.2e}). "
                                "The policy would be acting blind, so this evaluation "
                                "would be meaningless. Check MUJOCO_GL / the EGL driver.")
                t0 = time.perf_counter()
                with torch.inference_mode():
                    chunk = module(*obs).cpu().numpy()[0]     # (chunk_size, 12)
                infer_s += time.perf_counter() - t0
                inferences += 1

            a = np.clip(chunk[min(step % horizon, len(chunk) - 1)], lo, hi)
            data.ctrl[act_ids] = a
            for _ in range(per_step):
                mujoco.mj_step(model, data)

            joint_travel += np.abs(data.qpos[qadr] - prev_q)
            prev_q = data.qpos[qadr].copy()
            for n, b in bid.items():
                w = watch[n]
                w["max_shift"] = max(w["max_shift"],
                                     float(np.linalg.norm(data.xpos[b][:2] - start[n])))
                w["max_rise"] = max(w["max_rise"], float(data.xpos[b][2] - rest_z[n]))
                c = _object_contacts(model, data, b)
                if c["arm"] >= GRIP_CONTACT:
                    w["gripped"] += 1
                    if data.xpos[b][2] - rest_z[n] > LIFT_THRESHOLD and c["table"] == 0:
                        w["lifted"] = True
                if data.xpos[b][2] < table_z - 0.05:
                    w["off_table"] = True
    finally:
        renderer.close()

    print(f"\n  ran {total_steps} control steps, {inferences} inferences"
          f" ({infer_s / max(1, inferences) * 1000:.0f} ms each)")
    print(f"  arm joint travel (rad, per joint): "
          f"{np.round(joint_travel, 1).tolist()}")

    print("\n  final layout")
    results, ok = {}, 0
    for n in PLACE_ORDER:
        b = bid[n]
        at = data.xpos[b][:2]
        err = float(np.linalg.norm(at - goal[n]))
        moved = float(np.linalg.norm(at - start[n]))
        c = _object_contacts(model, data, b)
        settled = err <= args.tolerance and c["arm"] == 0 and not watch[n]["off_table"]
        ok += settled
        w = watch[n]
        # A prop can satisfy the tolerance without the policy having done anything:
        # the goal layout is searched per seed and sometimes lands a slot on top of
        # where its prop already lies.  Scored the same way as the expert either
        # way -- that is what makes the numbers comparable -- but an untouched pass
        # is not evidence the policy can place, so it is reported separately.
        earned = bool(settled and (w["gripped"] > 0 or moved > 0.005))
        results[n] = {"error": err, "moved": moved, "settled": bool(settled),
                      "earned": earned,
                      "max_shift": w["max_shift"], "max_rise": w["max_rise"],
                      "grip_steps": w["gripped"], "lifted": w["lifted"],
                      "off_table": w["off_table"], "contacts": c,
                      "goal": goal[n].tolist(), "final_xy": at.tolist()}
        mark = "no"
        if settled:
            mark = "ok" if earned else "ok (untouched, started in tolerance)"
        print(f"  {n:6s} err {err * 1000:7.1f} mm  moved {moved * 1000:6.1f} mm"
              f"  max shift {w['max_shift'] * 1000:6.1f}  max rise {w['max_rise'] * 1000:+6.1f}"
              f"  jaws on it {w['gripped']:4d} steps  lifted {str(w['lifted']):5s}"
              f"  {mark}"
              f"{'  OFF TABLE' if w['off_table'] else ''}")

    verdict = _verdict(results, joint_travel)
    earned = sum(1 for r in results.values() if r["earned"])
    note = "" if earned == ok else f" ({earned} earned, {ok - earned} already in tolerance)"
    print(f"\n  seed {args.seed}: {ok}/4 props placed{note}   -- {verdict}")

    if args.dump:
        print(json.dumps({"seed": args.seed, "placed": ok, "earned": earned,
                          "verdict": verdict,
                          "horizon": horizon, "inferences": inferences,
                          "joint_travel": joint_travel.tolist(),
                          "props": results}, indent=2, sort_keys=True, default=float))
    raise SystemExit(0 if ok == len(PLACE_ORDER) else 1)


def _verdict(results: dict, travel: np.ndarray) -> str:
    """One line on what the policy actually did, not just whether it scored."""
    if float(travel.max()) < 0.2:
        return "the arms barely moved"
    touched = [n for n, r in results.items() if r["grip_steps"] > 0]
    lifted = [n for n, r in results.items() if r["lifted"]]
    nudged = [n for n, r in results.items() if r["max_shift"] > 0.01]
    placed = [n for n, r in results.items() if r["earned"]]
    if placed:
        return f"placed {', '.join(placed)}; lifted {', '.join(lifted) or 'nothing'}"
    if lifted:
        return f"lifted {', '.join(lifted)} but placed none"
    if touched:
        return (f"the arms moved and touched {', '.join(touched)}"
                f" but lifted nothing; {len(nudged)} prop(s) shifted >10 mm")
    if nudged:
        return f"the arms moved and knocked {', '.join(nudged)} without grasping"
    return "the arms moved but never reached a prop"


def _fmt(v) -> str:
    return "[" + ", ".join(f"{x:+.3f}" for x in v) + "]"


if __name__ == "__main__":
    main()
