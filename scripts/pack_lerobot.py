#!/usr/bin/env python3
"""Assemble episodes staged by scripts/record_demos.py into a LeRobotDataset.

Writes the v3.0 layout through the ``lerobot`` library itself rather than
reproducing it by hand, so the result is whatever the current reader expects and
needs no conversion.  That means this script must be run by a python that has
``lerobot`` installed, which is deliberately *not* the project's own venv: the
project pins mujoco and numpy for the simulator and nothing here should disturb
them.

    <lerobot venv>/bin/python scripts/pack_lerobot.py --stage .cache/demos \
        --out .cache/lerobot/bimanual_table_setting

Reads only what the recorder staged; runs no physics and builds no renderer.
"""

import argparse
import json
import pathlib
import shutil
import sys

import numpy as np
from PIL import Image

FEATURE_STATE = "observation.state"
FEATURE_ACTION = "action"


def episodes(stage: pathlib.Path):
    """Every staged episode, ordered by seed then by the order it was recorded."""
    for seed_dir in sorted(stage.glob("seed-*")):
        metas = []
        for ep_dir in sorted(seed_dir.iterdir()):
            meta_path = ep_dir / "meta.json"
            if ep_dir.is_dir() and meta_path.exists():
                metas.append((json.loads(meta_path.read_text()), ep_dir))
        # PLACE_ORDER is the order they were actually performed in.
        order = {"plate": 0, "fork": 1, "spoon": 2, "mug": 3}
        for meta, ep_dir in sorted(metas, key=lambda m: order.get(m[0]["object"], 9)):
            yield meta, ep_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    root = pathlib.Path(__file__).resolve().parent.parent
    parser.add_argument("--stage", type=pathlib.Path, default=root / ".cache" / "demos")
    parser.add_argument("--out", type=pathlib.Path,
                        default=root / ".cache" / "lerobot" / "bimanual_table_setting")
    parser.add_argument("--repo-id", default="local/bimanual_table_setting")
    parser.add_argument("--robot-type", default="so101_bimanual")
    parser.add_argument("--video", action="store_true", default=True,
                        help="encode camera frames to mp4 (the LeRobot default)")
    parser.add_argument("--no-video", dest="video", action="store_false",
                        help="keep camera frames as png instead")
    args = parser.parse_args()

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:                                   # noqa: BLE001
        raise SystemExit(
            f"{exc}\n\nThis script needs the lerobot package, which is intentionally not\n"
            f"installed in the project venv.  Run it with the python of an environment\n"
            f"that has it, e.g.  ~/.cache/lerobot-verify/venv/bin/python {sys.argv[0]} ...")

    staged = list(episodes(args.stage))
    if not staged:
        raise SystemExit(f"no staged episodes under {args.stage}")
    first, _ = staged[0]
    cameras, joints = first["cameras"], first["joints"]
    res, fps = first["resolution"], first["fps"]

    features = {
        FEATURE_STATE: {"dtype": "float32", "shape": (len(joints),),
                        "names": list(joints)},
        FEATURE_ACTION: {"dtype": "float32", "shape": (len(joints),),
                         "names": list(joints)},
    }
    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": "video" if args.video else "image",
            "shape": (res, res, 3),
            "names": ["height", "width", "channel"],
        }

    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=fps,
        root=args.out,
        robot_type=args.robot_type,
        features=features,
        use_videos=args.video,
    )

    total = 0
    for meta, ep_dir in staged:
        state = np.load(ep_dir / "state.npy")
        action = np.load(ep_dir / "action.npy")
        n = len(state)
        if len(action) != n:
            raise SystemExit(f"{ep_dir}: {n} states but {len(action)} actions")
        for i in range(n):
            # ``task`` travels inside the frame; add_frame pops it back out.
            frame = {FEATURE_STATE: state[i].astype(np.float32),
                     FEATURE_ACTION: action[i].astype(np.float32),
                     "task": meta["task"]}
            for cam in cameras:
                path = ep_dir / cam / f"frame-{i:06d}.png"
                frame[f"observation.images.{cam}"] = np.asarray(
                    Image.open(path).convert("RGB"), dtype=np.uint8)
            dataset.add_frame(frame)
        dataset.save_episode()
        total += n
        print(f"  packed seed {meta['seed']:3d} {meta['object']:6s}"
              f" {'handover' if meta['handover'] else 'direct  '}"
              f" {n:4d} frames  err {meta['error_mm']:5.1f} mm")

    write_card(args.out, dataset, staged, total, args)
    print(f"\n{len(staged)} episodes, {total} frames -> {args.out}")


def write_card(out: pathlib.Path, dataset, staged, total: int, args) -> None:
    """A dataset card, chiefly so the action's units are not guessed at.

    Ten of the twelve action channels are joint *position* targets in radians and
    two are gripper *torques* in newton-metres -- the gripper is force-controlled
    so that the squeeze survives a lift instead of decaying as a servo settles.
    A policy trained on this has to reproduce both, and nothing in the LeRobot
    schema itself records the distinction.
    """
    handovers = sum(1 for meta, _ in staged if meta["handover"])
    seeds = sorted({meta["seed"] for meta, _ in staged})
    props = {}
    for meta, _ in staged:
        props[meta["object"]] = props.get(meta["object"], 0) + 1
    errs = sorted(meta["error_mm"] for meta, _ in staged)
    lines = [
        "# Bimanual table setting -- scripted expert demonstrations", "",
        "Demonstrations recorded in MuJoCo from the scripted expert in this repo",
        "(`control/primitives.py`), driven by `scripts/run_task.py`.  Two SO-101 arms",
        "set a table: a plate, fork, spoon and mug are moved from a randomized start",
        "layout into a place setting.  A prop whose slot is across the midline from",
        "where it lies is handed from one arm to the other via the table.", "",
        f"- episodes: **{len(staged)}** over seeds {seeds[0]}-{seeds[-1]}"
        f" ({len(seeds)} seeds contributed)",
        f"- frames: **{total}** at {dataset.fps} Hz",
        f"- of these, **{handovers}** episodes include a handover",
        "- per prop: " + ", ".join(f"{k} {v}" for k, v in sorted(props.items())), "",
        "## What an episode is", "",
        "One prop's manipulation: from the start of its pick to the end of its final",
        "place, the handover in between included.  **Only successful episodes are",
        "recorded** -- the prop finished within 20 mm of its slot and resting on the",
        "table.  Failed attempts are discarded, so this is not a dataset of recoveries.",
        f"Placement error across the kept episodes: median {errs[len(errs) // 2]:.1f} mm,"
        f" worst {errs[-1]:.1f} mm.", "",
        "## Action and state units", "",
        "Both are 12-vectors over the same joints, in this order:", "",
        "```", ", ".join(dataset.meta.features["action"]["names"]), "```", "",
        "`observation.state` is joint position (`qpos`) throughout: radians.", "",
        "`action` is the actuator command (`ctrl`), and **its channels are not all the",
        "same quantity**.  The ten arm joints are position targets in radians.  The two",
        "`*_gripper` channels are **torques in newton-metres**, negative to close: the",
        "gripper is force-controlled rather than position-controlled, because a",
        "commanded jaw angle relaxes its grip as the servo settles and drops the prop.",
        "Normalizing all twelve channels together will mix radians with newton-metres.",
        "", "## Cameras", "",
        f"`overhead` and `front`, {args.robot_type}, "
        f"{dataset.meta.features['observation.images.overhead']['shape'][0]} px square.",
        "Both are rendered from the simulator, not captured from hardware.", "",
        "## Provenance", "",
        "Recorded by `scripts/record_demos.py` (one seed per process) and assembled by",
        "`scripts/pack_lerobot.py`.  The expert was not modified to record: the capture",
        "is a hook on `mujoco.mj_step`, so these are the same trajectories the sweep in",
        "`scripts/sweep_task_report.py` measures.",
    ]
    (out / "README.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
