#!/usr/bin/env python3
"""Render one seed of the scripted expert to an mp4 segment, with overlays.

    MUJOCO_GL=egl scripts/record_video.py --seed 6 --out .cache/video
    MUJOCO_GL=egl scripts/record_video.py --summary --out .cache/video

One seed per process, as the rule here requires; scripts/record_video.sh runs the
range and concatenates the segments.  That also keeps memory flat: each seed is a
fresh interpreter, and frames are streamed straight into ffmpeg rather than collected,
so nothing grows with the length of the video.

The expert is not modified.  The run is driven by calling scripts/run_task.py's own
``main``, and the overlay state comes from wrapping the primitives -- ``pick``,
``place`` and ``handover`` on the driver for the outer phases, and
``control.primitives``' own ``pick`` and ``place`` for the two halves of a hand-off,
which ``handover`` calls internally.  So what is on screen is the run, not a
re-enactment of it.

Two renderers are genuinely needed and both are freed: the cinematic view at video
resolution, and the policy's own 256x256 observation, which is shown inset because
"perception operating" means the frames the policy would actually be given.  Nothing
learned is running here -- this is the scripted expert, and the overlay says so.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import pathlib
import subprocess
import sys
import time

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
import scripts.run_task as run_task  # noqa: E402
import control.primitives as prim  # noqa: E402
from envs.task import PLACE_ORDER, crossings, goal_layout  # noqa: E402

W, H = 960, 540
FPS = 30
OBS = 256                 # the policy's real observation size
INSET = 200               # how big that observation is drawn
STRIDE_S = 0.15           # sim seconds between frames, normally
HANDOVER_STRIDE_S = 0.05  # ... and during a hand-off, so it is legible
TITLE_S, FINAL_S, SUMMARY_S = 2.0, 2.0, 6.0
FONT_DIR = pathlib.Path("/usr/share/fonts/truetype/dejavu")
# The scene's own cameras are framed for the policy, not for a viewer: 'front' sits
# far enough back that the table fills a third of the shot.  The main view therefore
# uses a free camera, chosen by looking at candidates.  The insets keep the real
# cameras, because those are what the policy is actually given.
CAM_LOOKAT = (0.0, 0.0, 0.80)
CAM_AZIMUTH, CAM_ELEVATION, CAM_DISTANCE = 180.0, -30.0, 0.85

INK = (245, 246, 250)
DIM = (165, 172, 186)
PANEL = (18, 20, 27)
ACCENT = (255, 193, 86)
HAND = (120, 210, 255)


def font(size: int, mono: bool = False):
    name = "DejaVuSansMono.ttf" if mono else "DejaVuSans.ttf"
    path = FONT_DIR / name
    try:
        return ImageFont.truetype(str(path), size)
    except Exception:                                              # noqa: BLE001
        return ImageFont.load_default()


F_TITLE, F_BIG, F_TEXT = font(30), font(24), font(18)
F_MONO, F_SMALL = font(17, mono=True), font(14)


def ffmpeg(path: pathlib.Path):
    """A streaming encoder.  Frames go in as raw rgb24 and never pile up in RAM."""
    return subprocess.Popen(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", f"{W}x{H}",
         "-framerate", str(FPS), "-i", "-",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
         "-pix_fmt", "yuv420p", str(path)],
        stdin=subprocess.PIPE)


def task_for(name: str, handed: bool) -> str:
    if handed:
        return f"Hand the {name} to the other arm and place it in the table setting"
    return f"Place the {name} in the table setting"


def panel(draw, xy, wh, alpha_fill=PANEL):
    x, y = xy
    w, h = wh
    draw.rectangle([x, y, x + w, y + h], fill=alpha_fill)


def compose(main: np.ndarray, obs: dict, state: dict) -> Image.Image:
    """One video frame: the scene, the policy's observation, and the overlay."""
    img = Image.fromarray(main).convert("RGB")
    d = ImageDraw.Draw(img)

    # header
    panel(d, (0, 0), (W, 44))
    d.text((16, 9), "Scripted expert  ·  bimanual SO-101 table setting",
           font=F_BIG, fill=INK)
    d.text((W - 150, 15), f"seed {state['seed']}", font=F_TEXT, fill=ACCENT)

    # the policy's observation, inset
    x0 = W - INSET - 14
    y0 = 56
    for label, key in (("overhead", "overhead"), ("front", "front")):
        if key not in obs:
            continue
        thumb = Image.fromarray(obs[key]).resize((INSET, INSET), Image.BILINEAR)
        img.paste(thumb, (x0, y0))
        d.rectangle([x0, y0, x0 + INSET, y0 + INSET], outline=DIM)
        d.text((x0 + 6, y0 + INSET - 20), f"obs · {label} {OBS}x{OBS}",
               font=F_SMALL, fill=INK)
        y0 += INSET + 10

    # command + phase block
    box_h = 118
    panel(d, (0, H - box_h), (W - INSET - 30, box_h))
    d.text((16, H - box_h + 10), "COMMAND", font=F_SMALL, fill=DIM)
    cmd = state.get("task") or "Set the table"
    d.text((16, H - box_h + 28), cmd, font=F_TEXT, fill=INK)
    d.text((16, H - box_h + 60), "PHASE", font=F_SMALL, fill=DIM)
    d.text((16, H - box_h + 78), state.get("phase", "-"), font=F_MONO, fill=INK)
    d.text((520, H - box_h + 60), "ARM", font=F_SMALL, fill=DIM)
    d.text((520, H - box_h + 78), state.get("arm", "-"), font=F_MONO,
           fill=HAND if state.get("hand") else INK)

    if state.get("hand"):
        band = "HAND-OFF  ·  " + state["hand"]
        w = d.textlength(band, font=F_BIG)
        panel(d, ((W - w) / 2 - 18, 50), (w + 36, 40), alpha_fill=(12, 42, 60))
        d.text(((W - w) / 2, 58), band, font=F_BIG, fill=HAND)
    return img


def card(lines, sub=None) -> Image.Image:
    """A full-frame text card."""
    img = Image.new("RGB", (W, H), PANEL)
    d = ImageDraw.Draw(img)
    y = 120
    for text, f, fill in lines:
        w = d.textlength(text, font=f)
        d.text(((W - w) / 2, y), text, font=f, fill=fill)
        y += f.size + 18
    if sub:
        y += 12
        for text in sub:
            w = d.textlength(text, font=F_MONO)
            d.text(((W - w) / 2, y), text, font=F_MONO, fill=DIM)
            y += F_MONO.size + 8
    return img


def write(proc, img: Image.Image, seconds: float = None, frames: int = 1) -> int:
    n = frames if seconds is None else max(1, int(round(seconds * FPS)))
    buf = np.asarray(img, dtype=np.uint8).tobytes()
    for _ in range(n):
        proc.stdin.write(buf)
    return n


# ---------------------------------------------------------------------------


def run_seed(args) -> tuple[pathlib.Path, int]:
    out = args.out / f"seg-{args.seed:02d}.mp4"
    args.out.mkdir(parents=True, exist_ok=True)

    # Probe the scene once for the title card and the hand-off list.
    from envs.randomize import randomize
    from envs.scene import load_scene

    model = load_scene(args.scene)
    probe = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, probe, 0)
    randomize(model, probe, args.seed)
    mujoco.mj_forward(model, probe)
    goal = goal_layout(model, probe)
    blocked = crossings(model, probe, goal)
    del probe

    state = {"seed": args.seed, "phase": "settling", "arm": "-", "task": None,
             "hand": None, "depth": 0, "object": None}
    view = mujoco.MjvCamera()
    view.type = mujoco.mjtCamera.mjCAMERA_FREE
    view.lookat[:] = CAM_LOOKAT
    view.azimuth, view.elevation, view.distance = (
        CAM_AZIMUTH, CAM_ELEVATION, CAM_DISTANCE)
    big = mujoco.Renderer(model, height=H, width=W)
    small = mujoco.Renderer(model, height=OBS, width=OBS)
    proc = ffmpeg(out)
    written = [0]
    real_step = mujoco.mj_step
    live = {"data": None}

    def grab(data):
        big.update_scene(data, camera=view)
        main = big.render()
        obs = {}
        for cam in ("overhead", "front"):
            small.update_scene(data, camera=cam)
            obs[cam] = small.render()
        written[0] += write(proc, compose(main, obs, state))

    def hooked(m, data, nstep=1):
        real_step(m, data, nstep)
        if data is live["data"]:
            state["ticks"] = state.get("ticks", 0) + 1
            stride = HANDOVER_STRIDE_S if state["hand"] else STRIDE_S
            every = max(1, int(round(stride / m.opt.timestep)))
            if state["ticks"] % every == 0:
                grab(data)

    def wrap(fn, kind, inner=False):
        def call(m, data, *a, **k):
            if live["data"] is None:
                live["data"] = data
            prev = dict(state)
            state["depth"] += 1
            if kind == "pick":
                state.update(phase="PICK · grasp and lift", arm=a[0], object=a[1])
                if not inner:
                    state["task"] = task_for(a[1], a[1] in blocked)
            elif kind == "place":
                state.update(phase="PLACE · carry and release", arm=a[0])
            elif kind == "handover":
                state.update(hand=f"{a[0]} sets it down, {a[1]} collects the "
                                  f"{state.get('object') or 'prop'}",
                             phase="HAND-OFF via the table", arm=f"{a[0]} -> {a[1]}")
            if inner and state["hand"]:
                half = "set down" if kind == "place" else "collect"
                state["phase"] = f"HAND-OFF · {half}  ({a[0]})"
                state["arm"] = a[0]
            try:
                return fn(m, data, *a, **k)
            finally:
                state["depth"] -= 1
                if kind == "handover":
                    state["hand"] = None
                if state["depth"] == 0:
                    state.update(phase=prev["phase"], arm=prev["arm"])
        return call

    run_task.pick = wrap(run_task.pick, "pick")
    run_task.place = wrap(run_task.place, "place")
    run_task.handover = wrap(run_task.handover, "handover")
    prim.pick = wrap(prim.pick, "pick", inner=True)
    prim.place = wrap(prim.place, "place", inner=True)
    mujoco.mj_step = hooked

    try:
        # Title: the command and the randomized scene it applies to.
        scene0 = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, scene0, 0)
        randomize(model, scene0, args.seed)
        mujoco.mj_forward(model, scene0)
        big.update_scene(scene0, camera=view)
        title = Image.fromarray(big.render()).convert("RGB")
        dt = ImageDraw.Draw(title)
        panel(dt, (0, 0), (W, 44))
        dt.text((16, 9), f"seed {args.seed}  ·  randomized initial scene",
                font=F_BIG, fill=INK)
        panel(dt, (0, H - 150), (W, 150))
        dt.text((16, H - 140), "COMMAND", font=F_SMALL, fill=DIM)
        dt.text((16, H - 120), "Set the table: plate, fork, spoon, mug",
                font=F_BIG, fill=INK)
        y = H - 84
        for n in PLACE_ORDER:
            mark = "hand-off" if n in blocked else "one arm"
            dt.text((16, y), f"  {task_for(n, n in blocked)}", font=F_SMALL, fill=DIM)
            dt.text((W - 120, y), mark, font=F_SMALL,
                    fill=HAND if n in blocked else DIM)
            y += 18
        written[0] += write(proc, title, seconds=TITLE_S)
        del scene0

        argv, buf = sys.argv, io.StringIO()
        sys.argv = ["run_task.py", "--seed", str(args.seed), "--dump"]
        try:
            with contextlib.redirect_stdout(buf):
                run_task.main()
        except SystemExit:
            pass
        finally:
            sys.argv = argv
        text = buf.getvalue()
        report = json.loads(text[text.index("{"):]) if "{" in text else {}

        # Final state, held, with the per-prop outcome.
        if live["data"] is not None and report:
            state.update(phase="final state", arm="-", hand=None,
                         task="Set the table: plate, fork, spoon, mug")
            big.update_scene(live["data"], camera=view)
            final = Image.fromarray(big.render()).convert("RGB")
            df = ImageDraw.Draw(final)
            panel(df, (0, 0), (W, 44))
            df.text((16, 9), f"seed {args.seed}  ·  final state", font=F_BIG, fill=INK)
            panel(df, (0, H - 132), (W, 132))
            y = H - 122
            placed = 0
            for n in PLACE_ORDER:
                lay = report["layout"][n]
                ok = lay["settled"]
                placed += ok
                df.text((16, y), f"  {n:6s} {lay['error'] * 1000:7.1f} mm",
                        font=F_MONO, fill=INK if ok else DIM)
                df.text((260, y), "placed" if ok else "not placed", font=F_MONO,
                        fill=(140, 230, 150) if ok else (230, 140, 140))
                y += 22
            df.text((W - 230, H - 122), f"{placed}/4 placed", font=F_BIG, fill=ACCENT)
            written[0] += write(proc, final, seconds=FINAL_S)
    finally:
        mujoco.mj_step = real_step
        big.close()
        small.close()
        proc.stdin.close()
        proc.wait()
    return out, written[0]


def run_summary(args) -> tuple[pathlib.Path, int]:
    out = args.out / "seg-99-summary.mp4"
    args.out.mkdir(parents=True, exist_ok=True)
    proc = ffmpeg(out)
    img = card(
        [("Scripted expert, 10 randomized seeds", F_TITLE, INK),
         (f"{args.placed}/{args.total} props placed"
          f"   ({args.placed / args.total * 100:.0f}%)", F_TITLE, ACCENT),
         (f"median placement error {args.median_mm:.1f} mm", F_BIG, INK)],
        sub=["tolerance 20 mm; a prop counts only if it also rests on the table",
             "4 props x 10 seeds, one process per seed",
             "hand-off via the table where no single arm can reach both ends"])
    n = write(proc, img, seconds=SUMMARY_S)
    proc.stdin.close()
    proc.wait()
    return out, n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    from envs.scene import SCENE

    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--scene", type=pathlib.Path, default=SCENE)
    parser.add_argument("--out", type=pathlib.Path, default=_ROOT / ".cache" / "video")
    parser.add_argument("--placed", type=int, default=24)
    parser.add_argument("--total", type=int, default=40)
    parser.add_argument("--median-mm", type=float, default=1.6)
    args = parser.parse_args()

    t0 = time.time()
    if args.summary:
        path, n = run_summary(args)
    elif args.seed is not None:
        path, n = run_seed(args)
    else:
        raise SystemExit("give --seed N or --summary")
    print(f"{path}  {n} frames  {n / FPS:.1f} s of video  "
          f"({time.time() - t0:.0f} s wall)")


if __name__ == "__main__":
    main()
