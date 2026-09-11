#!/usr/bin/env python3
"""Record the scripted expert's successful manipulations on one seed.

Stages raw episodes for scripts/pack_lerobot.py to assemble into a
LeRobotDataset.  Nothing here changes the expert: the whole run is driven by
calling scripts/run_task.py's own ``main`` with the primitives wrapped, and the
per-timestep capture is a hook on ``mujoco.mj_step``, which every one of the
expert's legs already steps through.  So the trajectories recorded are exactly
the trajectories the sweep measured.

An *episode* is one prop's manipulation: from the moment the pick for that prop
begins to the moment its final place returns, the handover in the middle
included.  A whole seed is not an episode -- no seed yet gets all four props
into tolerance, so that would record nothing at all.  An episode is kept only if
that prop ended within tolerance of its slot and resting on the table, which is
the same test scripts/run_task.py applies in its final layout.

Deliberately minimal: one seed, one process.  A renderer is genuinely needed
here -- the observations are camera RGB -- so exactly one is built, shared by
both cameras, and freed before the process exits.

usage: MUJOCO_GL=egl scripts/record_demos.py --seed N [--out DIR]
"""

import argparse
import contextlib
import io
import json
import pathlib
import shutil
import sys

import mujoco
import numpy as np
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import scripts.run_task as run_task  # noqa: E402
from control.ik import ARM_JOINTS  # noqa: E402

CAMERAS = ("overhead", "front")
RESOLUTION = 256          # px; the render cost here is a fixed per-call sync,
                          # measured flat from 96 px to 256 px, so this is free
FPS = 25                  # Hz; the physics runs at 500 Hz, so every 20th step
JOINTS = tuple(f"{arm}_{j}" for arm in ("left", "right")
               for j in (*ARM_JOINTS, "gripper"))


def _decimation(model) -> int:
    every = round(1.0 / (FPS * model.opt.timestep))
    if abs(1.0 / (every * model.opt.timestep) - FPS) > 1e-6:
        raise ValueError(f"{FPS} Hz does not divide the {1 / model.opt.timestep:.0f} Hz physics")
    return every


class Recorder:
    """Captures (state, action, images) inside each prop's manipulation."""

    def __init__(self, out: pathlib.Path, seed: int):
        self.out, self.seed = out, seed
        self.model = self.data = self.renderer = None
        self.every = None
        self.qadr = self.ctrl_of = None
        self.depth, self.tick = 0, 0
        self.episodes, self.current, self.abandoned = [], None, []

    # -- lifecycle ---------------------------------------------------------
    def bind(self, model, data) -> None:
        if self.model is not None:
            return
        self.model, self.data = model, data
        self.every = _decimation(model)
        qadr = []
        for name in JOINTS:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise ValueError(f"scene has no joint {name}")
            qadr.append(model.jnt_qposadr[jid])
        self.qadr = np.asarray(qadr)
        self.renderer = mujoco.Renderer(model, height=RESOLUTION, width=RESOLUTION)

    def close(self) -> None:
        # Memory has been a problem in this repo; the renderer does not outlive
        # the recording it was built for.
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    # -- capture -----------------------------------------------------------
    def step(self) -> None:
        """Called before each physics step, so state and action are a matched pair.

        ``data.ctrl`` at this moment is the command about to be applied and
        ``data.qpos`` the state it is applied to, which is the (s_t, a_t) a
        policy is asked to reproduce.
        """
        if self.current is None:
            return
        if self.tick % self.every == 0:
            ep = self.current
            ep["state"].append(self.data.qpos[self.qadr].astype(np.float32))
            ep["action"].append(self.data.ctrl.astype(np.float32).copy())
            for cam in CAMERAS:
                self.renderer.update_scene(self.data, camera=cam)
                Image.fromarray(self.renderer.render()).save(
                    ep["dir"] / cam / f"frame-{len(ep['state']) - 1:06d}.png")
        self.tick += 1

    # -- episode boundaries ------------------------------------------------
    def begin(self, name: str) -> None:
        # A prop whose pick failed never reaches its place, so its episode never
        # ends; drop whatever it staged rather than leaving it orphaned.
        self.discard()
        path = self.out / f"seed-{self.seed:03d}" / f"{name}"
        if path.exists():
            shutil.rmtree(path)
        for cam in CAMERAS:
            (path / cam).mkdir(parents=True, exist_ok=True)
        self.current = {"object": name, "dir": path, "state": [], "action": []}
        self.tick = 0

    def end(self) -> None:
        if self.current is None:
            return
        self.episodes.append(self.current)
        self.current = None

    def discard(self) -> None:
        """Throw away an episode that never reached its place, and say so.

        A prop whose pick fails short-circuits the place, so its episode has no
        end; it is a failure either way, but it should be counted as one rather
        than vanish.
        """
        if self.current is not None:
            shutil.rmtree(self.current["dir"], ignore_errors=True)
            self.abandoned.append(self.current["object"])
            self.current = None


def wrap(recorder: Recorder, fn, kind: str):
    """Wrap a primitive so it marks episode boundaries without changing it."""
    def inner(model, data, *args, **kwargs):
        recorder.bind(model, data)
        top = recorder.depth == 0
        if top and kind == "pick":
            recorder.begin(args[1])
        recorder.depth += 1
        try:
            return fn(model, data, *args, **kwargs)
        finally:
            recorder.depth -= 1
            # A handover calls place and pick of its own; only the outermost
            # place ends the prop's episode.
            if recorder.depth == 0 and kind == "place":
                recorder.end()
    return inner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, required=True, help="the one seed to run")
    parser.add_argument("--out", type=pathlib.Path,
                        default=pathlib.Path(__file__).resolve().parent.parent
                        / ".cache" / "demos")
    args = parser.parse_args()

    recorder = Recorder(args.out, args.seed)
    real_step = mujoco.mj_step

    def hooked(model, data, nstep=1):
        if data is recorder.data:
            recorder.step()
        real_step(model, data, nstep)

    mujoco.mj_step = hooked
    run_task.pick = wrap(recorder, run_task.pick, "pick")
    run_task.place = wrap(recorder, run_task.place, "place")
    run_task.handover = wrap(recorder, run_task.handover, "handover")

    argv, buf = sys.argv, io.StringIO()
    sys.argv = ["run_task.py", "--seed", str(args.seed), "--dump"]
    try:
        with contextlib.redirect_stdout(buf):
            run_task.main()
    except SystemExit:
        pass
    finally:
        sys.argv = argv
        mujoco.mj_step = real_step
        recorder.discard()      # a trailing prop whose place never ran
        recorder.close()

    text = buf.getvalue()
    if "{" not in text:
        raise SystemExit(f"seed {args.seed}: run_task produced no report\n{text[-500:]}")
    report = json.loads(text[text.index("{"):])
    steps = {s["object"]: s for s in report["steps"]}

    kept, dropped, frames = [], [], 0
    for ep in recorder.episodes:
        name = ep["object"]
        placed = report["layout"][name]
        route = steps[name].get("arm") or "?"
        if not placed["settled"]:
            shutil.rmtree(ep["dir"])
            dropped.append(name)
            continue
        n = len(ep["state"])
        np.save(ep["dir"] / "state.npy", np.stack(ep["state"]))
        np.save(ep["dir"] / "action.npy", np.stack(ep["action"]))
        handed = steps[name].get("handover") is not None
        (ep["dir"] / "meta.json").write_text(json.dumps({
            "seed": args.seed, "object": name, "route": route, "handover": handed,
            "frames": n, "fps": FPS, "resolution": RESOLUTION,
            "cameras": list(CAMERAS), "joints": list(JOINTS),
            "error_mm": placed["error"] * 1000.0,
            "task": task_for(name, handed),
        }, indent=2, sort_keys=True))
        kept.append(name)
        frames += n

    dropped += recorder.abandoned
    print(f"seed {args.seed}: kept {len(kept)} episodes ({', '.join(kept) or '-'}),"
          f" {frames} frames; dropped {len(dropped)} ({', '.join(sorted(set(dropped))) or '-'})")


def task_for(name: str, handed: bool) -> str:
    if handed:
        return f"Hand the {name} to the other arm and place it in the table setting"
    return f"Place the {name} in the table setting"


if __name__ == "__main__":
    main()
