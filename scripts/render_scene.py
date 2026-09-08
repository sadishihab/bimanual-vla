#!/usr/bin/env python3
"""Render one frame of the bimanual table scene from each named camera."""

import argparse
import pathlib

import mujoco

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SCENE = REPO_ROOT / "scenes" / "bimanual_table.xml"
CAMERAS = ("overhead", "front")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=pathlib.Path, default=SCENE)
    parser.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("/tmp"))
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument(
        "--keyframe",
        default="home",
        help="keyframe to pose the model with, or '' for the model's default pose",
    )
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(args.scene))
    data = mujoco.MjData(model)

    if args.keyframe:
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, args.keyframe)
        if key_id < 0:
            raise SystemExit(f"no keyframe named {args.keyframe!r} in {args.scene}")
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)

    print(f"loaded {args.scene}: {model.nu} actuators, {model.njnt} joints, {model.nbody} bodies")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with mujoco.Renderer(model, height=args.height, width=args.width) as renderer:
        for camera in CAMERAS:
            renderer.update_scene(data, camera=camera)
            pixels = renderer.render()
            path = args.out_dir / f"scene_{camera}.png"
            _save_png(path, pixels)
            print(f"wrote {path} ({pixels.shape[1]}x{pixels.shape[0]})")


def _save_png(path: pathlib.Path, pixels) -> None:
    try:
        from PIL import Image
    except ImportError:
        import imageio.v3 as iio

        iio.imwrite(path, pixels)
    else:
        Image.fromarray(pixels).save(path)


if __name__ == "__main__":
    main()
