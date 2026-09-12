#!/usr/bin/env python3
"""Run a trained ACT policy in closed loop on the table-setting scene.

    MUJOCO_GL=egl scripts/eval_policy.py --seed 0 --checkpoint checkpoints/step_0044000_lang

The policy drives the actuators directly: its twelve outputs are written to
``data.ctrl`` and held for one control period, in place of the scripted primitives.
Scoring is the same test scripts/run_task.py applies -- a prop counts only if it ends
within the tolerance of its slot *and* resting on the table -- so the numbers are
comparable with the expert's sweep and between policies.

Two modes, because the two kinds of checkpoint admit different questions:

``directed`` (default)
    One episode per prop, each from a fresh reset of the same randomized layout, each
    given that prop's own task string.  Only a language-conditioned policy can use
    this, and it is the controlled experiment: identical observation, different
    sentence, so any change in behaviour is the conditioning and nothing else.
    Four episodes per seed, so the per-seed score is still out of four.

``undirected``
    One episode, scoring whatever happened to all four props.  The only honest option
    for a policy that cannot be told which prop to move, and what the unconditioned
    checkpoint was measured with.

One caveat on ``directed``, stated rather than hidden: every episode starts from the
pristine layout, but in the demonstrations the later props were picked from a
partially set table.  So a directed fork episode is slightly off the distribution its
own demonstrations came from.  The alternative -- running the four in order from one
reset -- keeps the distribution but lets a failure on one prop poison the next, which
would confound exactly the question being asked.

A renderer is unavoidable here -- the policy's observation *is* camera RGB -- so
exactly one offscreen renderer is built, shared by both cameras, and freed before
exit.  No video is written and no window is opened.  Observations are only consumed at
an inference, so this renders a handful of frames per episode rather than one per
control step.
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

_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "openvino"))
from act_io import (ACTInference, ENV_KEY, build_policy,  # noqa: E402
                    dataset_meta, task_embedder)
from control.ik import ARM_JOINTS  # noqa: E402
from control.primitives import (LIFT_THRESHOLD, _object_contacts,  # noqa: E402
                                _table_top_z)
from envs.randomize import randomize  # noqa: E402
from envs.scene import SCENE, load_scene  # noqa: E402
from envs.task import PLACE_ORDER, check_setting, crossings, goal_layout  # noqa: E402

CAMERAS = ("observation.images.overhead", "observation.images.front")
CAMERA_NAMES = ("overhead", "front")
RESOLUTION = 256
CONTROL_HZ = 25          # the rate the demonstrations were recorded at
SETTLE_TIME = 0.5        # s, so the props drop the couple of mm they spawn above
GRIP_CONTACT = 1         # arm contacts on a prop that count as "the jaws are on it"
JOINTS = tuple(f"{arm}_{j}" for arm in ("left", "right")
               for j in (*ARM_JOINTS, "gripper"))


def task_for(name: str, handed: bool) -> str:
    """The task string the recorder would have labelled this episode with."""
    if handed:
        return f"Hand the {name} to the other arm and place it in the table setting"
    return f"Place the {name} in the table setting"


def fresh_state(model, seed: int):
    """A settled, randomized scene: what every episode starts from."""
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    randomize(model, data, seed)
    mujoco.mj_forward(model, data)
    for _ in range(int(round(SETTLE_TIME / model.opt.timestep))):
        mujoco.mj_step(model, data)
    return data


def run_episode(model, data, module, renderer, qadr, *, task_vec, seconds, horizon,
                bids, sanity) -> dict:
    """Drive the actuators from the policy, watching every prop.  Returns the watch."""
    lo = model.actuator_ctrlrange[:, 0]
    hi = model.actuator_ctrlrange[:, 1]
    table_z = _table_top_z(model)
    start = {n: data.xpos[b][:2].copy() for n, b in bids.items()}
    rest_z = {n: float(data.xpos[b][2]) for n, b in bids.items()}
    watch = {n: {"max_shift": 0.0, "max_rise": 0.0, "gripped": 0, "lifted": False,
                 "off_table": False} for n in bids}
    travel = np.zeros(len(qadr))
    prev = data.qpos[qadr].copy()
    chunk, inferences, infer_s = None, 0, 0.0

    for step in range(int(seconds * CONTROL_HZ)):
        if chunk is None or step % horizon == 0:
            obs = [torch.from_numpy(data.qpos[qadr].astype(np.float32)[None])]
            for cam in CAMERA_NAMES:
                renderer.update_scene(data, camera=cam)
                px = renderer.render().astype(np.float32) / 255.0
                obs.append(torch.from_numpy(px.transpose(2, 0, 1)[None]))
            if task_vec is not None:
                obs.append(task_vec)
            sanity(obs[1:1 + len(CAMERA_NAMES)])
            t0 = time.perf_counter()
            with torch.inference_mode():
                chunk = module(*obs).cpu().numpy()[0]
            infer_s += time.perf_counter() - t0
            inferences += 1

        data.ctrl[:] = np.clip(chunk[min(step % horizon, len(chunk) - 1)], lo, hi)
        for _ in range(max(1, int(round(1.0 / (CONTROL_HZ * model.opt.timestep))))):
            mujoco.mj_step(model, data)

        travel += np.abs(data.qpos[qadr] - prev)
        prev = data.qpos[qadr].copy()
        for n, b in bids.items():
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
    return {"watch": watch, "travel": travel, "start": start,
            "inferences": inferences, "infer_s": infer_s}


def make_sanity(stats):
    """Abort if a camera returns a constant image.

    The whole evaluation is only as good as the observation.  A renderer handing back
    black frames would still produce plausible motion -- the policy would emit its
    mean action -- so the first frames are checked against what the dataset's own
    image statistics say to expect.
    """
    state = {"done": False}

    def check(images):
        if state["done"]:
            return
        state["done"] = True
        for cam, img in zip(CAMERA_NAMES, images):
            m, sd = float(img.mean()), float(img.std())
            want = stats.get(f"observation.images.{cam}", {}).get("mean")
            want = float(np.asarray(want).mean()) if want is not None else None
            note = "" if want is None else f", dataset mean {want:.3f}"
            print(f"  rendered {cam:8s}: mean {m:.3f} std {sd:.3f}{note}")
            if sd < 1e-3:
                raise SystemExit(
                    f"the {cam} render is a constant image (std {sd:.2e}). The policy "
                    "would be acting blind, so this evaluation would be meaningless. "
                    "Check MUJOCO_GL / the EGL driver.")
    return check


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, required=True, help="the one seed to run")
    parser.add_argument("--checkpoint", type=pathlib.Path,
                        default=_ROOT / "checkpoints" / "step_0044000_lang")
    parser.add_argument("--dataset", type=pathlib.Path,
                        default=_ROOT / ".cache" / "lerobot" / "bimanual_table_setting")
    parser.add_argument("--repo-id", default="local/bimanual_table_setting")
    parser.add_argument("--scene", type=pathlib.Path, default=SCENE)
    parser.add_argument("--tolerance", type=float, default=0.020)
    parser.add_argument("--mode", choices=("directed", "undirected"), default="directed")
    parser.add_argument("--seconds", type=float, default=20.0,
                        help="sim seconds per episode; the expert's own episodes run 11-27 s")
    parser.add_argument("--horizon", type=int, default=None,
                        help="actions per inference; default is the checkpoint's n_action_steps")
    parser.add_argument("--dump", action="store_true")
    args = parser.parse_args()

    model = load_scene(args.scene)
    probe = fresh_state(model, args.seed)
    goal = goal_layout(model, probe)
    check_setting(model, goal)
    blocked = crossings(model, probe, goal)
    bids = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in PLACE_ORDER}
    qadr = np.asarray([model.jnt_qposadr[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in JOINTS])

    meta = dataset_meta(args.dataset if (args.dataset / "meta" / "info.json").exists()
                        else None, args.repo_id)
    policy, cfg, stats, source = build_policy(
        args.checkpoint, meta, chunk=100, cameras=list(CAMERAS),
        resolution=RESOLUTION, state_dim=12)
    conditioned = bool(cfg.env_state_feature)
    module = ACTInference(policy, stats, list(CAMERAS),
                          env_key=ENV_KEY if conditioned else None).eval()
    horizon = args.horizon or int(cfg.n_action_steps)

    embed = label = None
    if conditioned:
        embed, label = task_embedder(args.checkpoint, cfg.env_state_feature.shape[0])
    if args.mode == "directed" and not conditioned:
        raise SystemExit(
            "--mode directed needs a language-conditioned checkpoint; this one has no "
            "task input, so the task string would be ignored and the four episodes "
            "would be identical. Use --mode undirected.")

    print(f"seed {args.seed}: closed-loop ACT, {args.mode}, tolerance "
          f"{args.tolerance * 1000:.0f} mm")
    print(f"  checkpoint     : {args.checkpoint.name}")
    print(f"  conditioned    : {conditioned}"
          + (f", {cfg.env_state_feature.shape[0]}-d task embedding from {label}"
             if conditioned else ""))
    print(f"  statistics     : {source}")
    print(f"  horizon        : {horizon} actions per inference "
          f"({horizon / CONTROL_HZ:.1f} s open loop)")
    print(f"  episodes       : {'4 (one per prop)' if args.mode == 'directed' else '1'}"
          f" x {args.seconds:.0f} s")
    if blocked:
        print(f"  crosses the midline (handover phrasing): {', '.join(sorted(blocked))}")

    sanity = make_sanity(stats)
    renderer = mujoco.Renderer(model, height=RESOLUTION, width=RESOLUTION)
    episodes = []
    try:
        targets = list(PLACE_ORDER) if args.mode == "directed" else [None]
        for target in targets:
            data = fresh_state(model, args.seed)
            task = task_for(target, target in blocked) if target else None
            vec = None
            if conditioned:
                # An undirected episode on a conditioned policy still has to say
                # something; the plate's string is what the opening observation was
                # most often labelled with.
                vec = torch.from_numpy(
                    np.asarray(embed(task or task_for("plate", False)),
                               dtype=np.float32)[None])
            out = run_episode(model, data, module, renderer, qadr, task_vec=vec,
                              seconds=args.seconds, horizon=horizon, bids=bids,
                              sanity=sanity)
            out.update({"target": target, "task": task, "data": data})
            episodes.append(out)
    finally:
        renderer.close()

    results, placed, earned = {}, 0, 0
    print()
    for ep in episodes:
        data, w = ep["data"], ep["watch"]
        scored = [ep["target"]] if ep["target"] else list(PLACE_ORDER)
        if ep["target"]:
            others = [n for n in PLACE_ORDER if n != ep["target"] and w[n]["gripped"] > 0]
            print(f"  episode: {ep['task']}")
            print(f"    {ep['inferences']} inferences"
                  f" ({ep['infer_s'] / max(1, ep['inferences']) * 1000:.0f} ms each),"
                  f" joint travel {ep['travel'].sum():.1f} rad"
                  f", other props touched: {', '.join(others) or 'none'}")
        for n in scored:
            b = bids[n]
            at = data.xpos[b][:2]
            err = float(np.linalg.norm(at - goal[n]))
            moved = float(np.linalg.norm(at - ep["start"][n]))
            c = _object_contacts(model, data, b)
            ok = err <= args.tolerance and c["arm"] == 0 and not w[n]["off_table"]
            got = bool(ok and (w[n]["gripped"] > 0 or moved > 0.005))
            placed += ok
            earned += got
            mark = "no" if not ok else ("ok" if got else "ok (untouched, started in tolerance)")
            results[n] = {"error": err, "moved": moved, "settled": bool(ok), "earned": got,
                          "max_shift": w[n]["max_shift"], "max_rise": w[n]["max_rise"],
                          "grip_steps": w[n]["gripped"], "lifted": w[n]["lifted"],
                          "off_table": w[n]["off_table"], "task": ep["task"],
                          "others_touched": [x for x in PLACE_ORDER
                                             if x != n and w[x]["gripped"] > 0],
                          "goal": goal[n].tolist(), "final_xy": at.tolist()}
            print(f"    {n:6s} err {err * 1000:7.1f} mm  moved {moved * 1000:6.1f} mm"
                  f"  max rise {w[n]['max_rise'] * 1000:+6.1f}"
                  f"  jaws on it {w[n]['gripped']:4d} steps"
                  f"  lifted {str(w[n]['lifted']):5s}  {mark}"
                  f"{'  OFF TABLE' if w[n]['off_table'] else ''}")

    note = "" if earned == placed else f" ({earned} earned, {placed - earned} already in tolerance)"
    contacted = sum(1 for n, r in results.items() if r["grip_steps"] > 0)
    print(f"\n  seed {args.seed}: {placed}/{len(results)} props placed{note};"
          f" target contacted in {contacted}/{len(results)} episodes")

    if args.dump:
        print(json.dumps({"seed": args.seed, "mode": args.mode,
                          "conditioned": conditioned, "placed": placed,
                          "earned": earned, "horizon": horizon,
                          "props": results}, indent=2, sort_keys=True, default=float))
    raise SystemExit(0 if earned == len(results) else 1)


if __name__ == "__main__":
    main()
