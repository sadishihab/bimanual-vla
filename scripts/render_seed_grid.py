#!/usr/bin/env python3
"""Render a contact sheet of domain-randomized scenes, one tile per seed."""

import argparse
import pathlib
import sys

import mujoco
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from envs.randomize import randomize  # noqa: E402

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SCENE = REPO_ROOT / "scenes" / "bimanual_table.xml"

PAD = 8
LABEL_H = 22
BG = (24, 26, 30)
FG = (235, 235, 235)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=pathlib.Path, default=SCENE)
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("/tmp/seed_grid.png"))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--camera", default="overhead")
    parser.add_argument("--cols", type=int, default=5)
    parser.add_argument("--tile-width", type=int, default=440)
    parser.add_argument("--tile-height", type=int, default=330)
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(args.scene))
    data = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")

    tiles = []
    with mujoco.Renderer(model, height=args.tile_height, width=args.tile_width) as renderer:
        for seed in args.seeds:
            # randomize() overwrites every field it touches, so reusing one model
            # across seeds gives the same result as loading a fresh one each time.
            if key_id >= 0:
                mujoco.mj_resetDataKeyframe(model, data, key_id)
            else:
                mujoco.mj_resetData(model, data)
            randomize(model, data, seed)
            renderer.update_scene(data, camera=args.camera)
            tiles.append(renderer.render().copy())
            print(f"rendered seed {seed}")

    cols = min(args.cols, len(tiles))
    rows = -(-len(tiles) // cols)
    tile_w, tile_h = args.tile_width, args.tile_height + LABEL_H
    sheet = Image.new("RGB", (cols * tile_w + PAD * (cols + 1),
                             rows * tile_h + PAD * (rows + 1)), BG)
    draw = ImageDraw.Draw(sheet)
    for i, (seed, tile) in enumerate(zip(args.seeds, tiles)):
        x = PAD + (i % cols) * (tile_w + PAD)
        y = PAD + (i // cols) * (tile_h + PAD)
        sheet.paste(Image.fromarray(tile), (x, y))
        draw.text((x + 4, y + args.tile_height + 5), f"seed {seed}", fill=FG)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out)
    print(f"wrote {args.out} ({sheet.width}x{sheet.height}, {len(tiles)} tiles)")

    digests = [hash(t.tobytes()) for t in tiles]
    print(f"distinct tiles: {len(set(digests))}/{len(tiles)}")
    if len(set(digests)) != len(tiles):
        raise SystemExit("seeds produced duplicate images")


if __name__ == "__main__":
    main()
