#!/usr/bin/env python3
"""Voice command -> transcript -> matched task string -> the conditioned policy.

    export SPEECHMATICS_API_KEY=...
    MUJOCO_GL=egl <lerobot-env>/bin/python scripts/voice_command_demo.py \
        --audio speechmatics_test.mp4 --seed 0

Three isolated pieces, run in sequence and never modifying each other:
``speechmatics/transcribe.py`` (audio -> text), ``speechmatics/match_task.py``
(text -> one of the seven task strings the checkpoint knows), and
``scripts/eval_policy.py``'s own :func:`run_episode` (task string -> the
language-conditioned policy driving the scene) -- the exact machinery the directed
closed-loop evaluation uses, imported rather than re-implemented, so a voice-driven
episode is not a second, subtly different code path from the one the numbers in
README.md were measured on.

This file adds nothing to ``control/``, ``envs/``, or the eval scripts themselves;
it is additive, in its own ``speechmatics/`` package plus this one driver.

One seed per process, no rendering beyond the policy's own 256x256 observation,
which ``eval_policy``'s sanity check still verifies is not a constant image before
any control step runs.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

import mujoco
import numpy as np
import torch

_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "openvino"))
sys.path.insert(0, str(_ROOT / "speechmatics"))
from act_io import ACTInference, ENV_KEY, build_policy, dataset_meta, task_embedder  # noqa: E402
from envs.scene import SCENE  # noqa: E402
import scripts.eval_policy as EP  # noqa: E402
from match_task import match_task  # noqa: E402
from transcribe import SpeechmaticsError, transcribe  # noqa: E402


class _ViewerClosed(RuntimeError):
    """Raised to unwind cleanly when the user closes the viewer window mid-episode."""


HOLD_MAX_S = 600.0    # cap on how long the finished window is held open unattended


def _run_with_viewer(model, data, module, renderer, qadr, task_vec, args, bids, sanity,
                     horizon):
    """Run the episode with a live, synced MuJoCo window, and hand back the viewer too.

    ``eval_policy.run_episode`` calls ``mujoco.mj_step`` as a module attribute
    lookup on every physics step (``mujoco.mj_step(model, data)``, not a name bound
    at import time), which is what ``scripts/record_video.py`` already relies on to
    hook the scripted expert's primitives without editing them.  The same trick
    works here without touching eval_policy.py: swap ``mujoco.mj_step`` for a
    wrapper that also syncs the viewer and paces itself to wall-clock time, run the
    unmodified ``run_episode``, and put the real function back.

    Real-time pacing matters because ``run_episode`` steps physics as fast as the
    CPU allows -- a 20 s episode would otherwise flash past in a couple of seconds
    of wall time, which is unwatchable and unrecordable.  Pacing sleeps by the
    model's own timestep divided by ``--speed``, so 1.0 is real time and 0.5 is
    half speed.

    Two things about ``viewer.sync()`` on this machine shape the pacing below, and
    both were measured, not assumed.  It costs **~21 ms a call** here (GLFW is
    running a degraded path -- no ``libdecor-gtk``, so no native window
    decorations, on Wayland) -- so syncing on every physics step, at the model's
    500 Hz, would spend *ten seconds of pure render overhead per second of sim
    time*.  A first version did exactly that and looked hung: the process was
    genuinely alive and busy, not stuck, but 20 s of sim time took several minutes
    of wall time, and it did not even respond to SIGINT while inside that string of
    slow native calls.  Syncing at a fixed ~30 Hz instead -- plainly enough for a
    screen recording -- and pacing sleeps against *cumulative* sim time rather than
    a fixed per-step amount (so one slow sync does not compound into permanent
    drift) brought the same episode back to within a few percent of real time.

    Separately: this deliberately does **not** use ``launch_passive`` as a context
    manager, and never calls ``viewer.close()``.  Even with sync throttled, the
    viewer's own teardown occasionally does not return, so ``main()`` holds the
    finished window open by hand (:func:`_hold_until_closed`) and the whole
    process ends via ``os._exit``, which needs no cooperation from anything --
    it ends every thread at once at the OS level.  The cost is that Python's
    normal cleanup (flushing buffers, atexit hooks) is skipped, which is why every
    print in this script happens before that point.
    """
    import mujoco.viewer

    SYNC_HZ = 30.0

    viewer = mujoco.viewer.launch_passive(model, data)
    real_step = mujoco.mj_step
    start = time.time()
    sim_elapsed = 0.0
    last_sync = 0.0

    def paced_step(m, d, nstep=1):
        nonlocal sim_elapsed, last_sync
        real_step(m, d, nstep)
        if d is not data:
            return
        if not viewer.is_running():
            raise _ViewerClosed("viewer window was closed")
        sim_elapsed += nstep * m.opt.timestep
        now = time.time()
        if now - last_sync >= 1.0 / SYNC_HZ:
            viewer.sync()
            last_sync = now
        target = start + sim_elapsed / args.speed
        delay = target - time.time()
        if delay > 0:
            time.sleep(delay)

    mujoco.mj_step = paced_step
    try:
        out = EP.run_episode(model, data, module, renderer, qadr,
                             task_vec=task_vec, seconds=args.seconds,
                             horizon=horizon, bids=bids, sanity=sanity)
    except _ViewerClosed:
        out = None
    finally:
        mujoco.mj_step = real_step
    return out, viewer


def _hold_until_closed(viewer, speed: float, max_s: float = HOLD_MAX_S) -> None:
    """Keep the finished episode's final frame on screen until the window is closed.

    Polls rather than blocking on the viewer's own machinery, for the same reason
    :func:`_run_with_viewer` avoids ``close()``: nothing here calls into whatever
    native path hangs.  Capped at :data:`HOLD_MAX_S` so an unattended run still ends.
    """
    dt = (1.0 / 30.0) / max(speed, 1e-6)
    deadline = time.time() + max_s
    while viewer.is_running() and time.time() < deadline:
        viewer.sync()
        time.sleep(dt)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audio", type=pathlib.Path, required=True,
                        help="an audio file with a spoken command")
    parser.add_argument("--seed", type=int, default=0,
                        help="which randomized table layout to run the command against")
    parser.add_argument("--checkpoint", type=pathlib.Path,
                        default=_ROOT / "checkpoints" / "step_0044000_shuffled")
    parser.add_argument("--dataset", type=pathlib.Path,
                        default=_ROOT / ".cache" / "lerobot" / "bimanual_table_setting")
    parser.add_argument("--repo-id", default="local/bimanual_table_setting")
    parser.add_argument("--scene", type=pathlib.Path, default=SCENE)
    parser.add_argument("--tolerance", type=float, default=0.020)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--dump", action="store_true")
    parser.add_argument("--viewer", action="store_true",
                        help="open a live, interactive MuJoCo window and play the "
                             "episode in it in real time -- for screen recording. "
                             "Needs a display (DISPLAY set, X/Wayland reachable) and "
                             "MUJOCO_GL must NOT be set to egl/osmesa: measured on "
                             "this machine, an EGL offscreen context plus a GLFW "
                             "window in one process segfaults at teardown, after "
                             "the episode has already finished and its result "
                             "printed, but before the process exits cleanly.")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="--viewer playback speed; 0.5 runs at half speed, which "
                             "is often easier to record and narrate over")
    parser.add_argument("--hold-seconds", type=float, default=HOLD_MAX_S,
                        help="--viewer only: how long to keep the finished episode's "
                             "final frame on screen before force-exiting, if the "
                             "window is not closed first")
    args = parser.parse_args()
    if args.viewer and args.speed <= 0:
        raise SystemExit("--speed must be positive")
    if args.viewer and os.environ.get("MUJOCO_GL", "").lower() in ("egl", "osmesa"):
        raise SystemExit(
            f"--viewer needs a live window, but MUJOCO_GL={os.environ['MUJOCO_GL']!r} "
            "selects a headless backend. Measured on this machine, mixing that "
            "offscreen backend with the viewer's GLFW window segfaults the process "
            "at exit. Run with MUJOCO_GL unset (do not export it at all) instead.")

    if not args.audio.exists():
        raise SystemExit(f"no such file: {args.audio}")

    print(f"1. transcribing {args.audio.name} via Speechmatics ...")
    try:
        heard = transcribe(args.audio)
    except SpeechmaticsError as exc:
        raise SystemExit(f"transcription failed: {exc}")
    print(f"   transcript: {heard['text']!r}")

    print("2. matching against the seven known task strings ...")
    matched = match_task(heard["text"])
    print(f"   matched   : {matched['matched_task']!r}")
    print(f"   confidence: {matched['confidence']:.3f}"
          f" (runner-up {matched['runner_up_confidence']:.3f}, margin {matched['margin']:.3f}"
          f"{', AMBIGUOUS' if matched['ambiguous'] else ''})")
    if matched["ambiguous"]:
        print("   the top two candidates were close; proceeding with the best match anyway,"
              " but check --dump if the run does something unexpected.")

    task_string = matched["matched_task"]
    # Which prop the matched string names, so the episode can be scored against
    # that prop's own slot -- the same accounting eval_policy.py's directed mode uses.
    target = next((p for p in EP.PLACE_ORDER if p in task_string.lower()), None)
    if target is None:
        raise SystemExit(f"matched task {task_string!r} does not name a known prop")

    print(f"3. running the language-conditioned policy on seed {args.seed},"
          f" instructed: {task_string!r}")
    model = EP.load_scene(args.scene)
    data = EP.fresh_state(model, args.seed)
    goal = EP.goal_layout(model, data)
    bids = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in EP.PLACE_ORDER}
    qadr = np.asarray([model.jnt_qposadr[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in EP.JOINTS])

    meta = dataset_meta(args.dataset if (args.dataset / "meta" / "info.json").exists()
                        else None, args.repo_id)
    policy, cfg, stats, source = build_policy(
        args.checkpoint, meta, chunk=100, cameras=list(EP.CAMERAS),
        resolution=EP.RESOLUTION, state_dim=12)
    if not cfg.env_state_feature:
        raise SystemExit(f"{args.checkpoint} is not language-conditioned; "
                         "voice commands need a checkpoint with a task input")
    module = ACTInference(policy, stats, list(EP.CAMERAS), env_key=ENV_KEY).eval()
    horizon = args.horizon or int(cfg.n_action_steps)

    embed, embed_source = task_embedder(args.checkpoint, cfg.env_state_feature.shape[0])
    print(f"   task embedding from: {embed_source}")
    task_vec = torch.from_numpy(np.asarray(embed(task_string), dtype=np.float32)[None])

    sanity = EP.make_sanity(stats)
    renderer = mujoco.Renderer(model, height=EP.RESOLUTION, width=EP.RESOLUTION)
    try:
        viewer = None
        if args.viewer:
            out, viewer = _run_with_viewer(model, data, module, renderer, qadr, task_vec,
                                           args, bids, sanity, horizon)
        else:
            out = EP.run_episode(model, data, module, renderer, qadr, task_vec=task_vec,
                                 seconds=args.seconds, horizon=horizon, bids=bids,
                                 sanity=sanity)
    finally:
        renderer.close()

    if out is None:
        print("\nstopped: the viewer window was closed before the episode finished; "
              "no result to report.")
        os._exit(0)

    w = out["watch"][target]
    b = bids[target]
    at = data.xpos[b][:2]
    err = float(np.linalg.norm(at - goal[target]))
    from control.primitives import _object_contacts

    c = _object_contacts(model, data, b)
    settled = err <= args.tolerance and c["arm"] == 0 and not w["off_table"]
    earned = bool(settled and (w["gripped"] > 0 or w["max_shift"] > 0.005))

    print(f"\n4. result: {target}")
    print(f"   error       : {err * 1000:.1f} mm  (tolerance {args.tolerance * 1000:.0f} mm)")
    print(f"   contacted   : {'yes' if w['gripped'] > 0 else 'no'}"
          f"  ({w['gripped']} control steps)")
    print(f"   lifted      : {w['lifted']}")
    print(f"   placed      : {'yes' if settled else 'no'}"
          f"{'  (earned)' if earned else '  (started in tolerance)' if settled else ''}")

    if args.dump:
        print(json.dumps({
            "audio": str(args.audio), "transcript": heard["text"], "match": matched,
            "target": target, "seed": args.seed, "error_mm": err * 1000.0,
            "contacted": w["gripped"] > 0, "lifted": w["lifted"], "settled": settled,
            "earned": earned,
        }, indent=2, sort_keys=True, default=float))

    if args.viewer and viewer is not None:
        print(f"\nholding the final frame in the viewer window -- close it "
              f"(or wait up to {args.hold_seconds:.0f}s) to end.")
        _hold_until_closed(viewer, args.speed, max_s=args.hold_seconds)
        # Not a plain return: see _run_with_viewer's docstring for why the viewer's
        # own teardown hangs on this machine and does not even respond to SIGINT.
        # os._exit ends every thread at the OS level, needing no cooperation from
        # whatever is stuck, at the cost of skipping Python's normal shutdown --
        # harmless here because every result has already been printed above.
        os._exit(0)


if __name__ == "__main__":
    main()
