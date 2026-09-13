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
import pathlib
import sys

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
    args = parser.parse_args()

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
        out = EP.run_episode(model, data, module, renderer, qadr, task_vec=task_vec,
                             seconds=args.seconds, horizon=horizon, bids=bids, sanity=sanity)
    finally:
        renderer.close()

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


if __name__ == "__main__":
    main()
