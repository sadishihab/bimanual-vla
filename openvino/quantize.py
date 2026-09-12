#!/usr/bin/env python3
"""INT8 post-training quantization of the ACT IR with NNCF.

    openvino/quantize.py --ir openvino/ir/act_fp32.xml --out openvino/ir

Calibration uses real frames from the recorded dataset, which is the point of
post-training quantization: the activation ranges are measured on the
distribution the policy will actually see, not on noise.  The exported graph
normalizes internally (see openvino/act_io.py), so calibration feeds it raw
observations exactly as deployment will -- images in [0, 1] and joint positions
in radians.  Feeding pre-normalized data here would set every activation range
for the wrong input scale.

``model_type=TRANSFORMER`` is not cosmetic.  ACT is a transformer with two ResNet
backbones in front, and that preset keeps the parts of attention that are
sensitive to quantization -- softmax inputs, the residual adds -- out of INT8,
which is the difference between a working INT8 policy and a broken one.

Reports the divergence from FP32 on frames held out of the calibration set.  With
randomly initialized weights that number says nothing about behaviour, but it
does say whether quantization is numerically sane, which is what it is for here.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np
import openvino as ov

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from act_io import DEFAULT_CAMERAS, STATE_KEY  # noqa: E402


def dataset_frames(root: pathlib.Path, repo_id: str, cameras, needed: int, batch: int):
    """Raw observations from the recorded dataset, as the IR takes them."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id, root=root)
    missing = [c for c in cameras if c not in ds.meta.camera_keys]
    if missing:
        raise SystemExit(f"dataset has no camera(s) {missing}; it has {ds.meta.camera_keys}")
    # Spread the picks across the whole dataset rather than taking the first N
    # frames, which would all come from one episode of one seed.
    idx = np.linspace(0, len(ds) - 1, num=min(needed, len(ds)), dtype=int)
    out = []
    for i in idx:
        s = ds[int(i)]
        item = {STATE_KEY: s[STATE_KEY].numpy()[None].astype(np.float32).repeat(batch, 0)}
        for cam in cameras:
            item[cam] = s[cam].numpy()[None].astype(np.float32).repeat(batch, 0)
        out.append(item)
    return out, len(ds)


def synthetic_frames(specs, needed: int):
    """Fallback when no dataset is present: uniform noise in the right ranges."""
    rng = np.random.default_rng(0)
    out = []
    for _ in range(needed):
        item = {}
        for name, shape in specs:
            item[name] = (rng.random(shape, dtype=np.float32) * 2 - 1
                          if name == STATE_KEY else rng.random(shape, dtype=np.float32))
        out.append(item)
    return out


def main() -> None:
    here = pathlib.Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ir", type=pathlib.Path, default=here / "ir" / "act_fp32.xml")
    parser.add_argument("--out", type=pathlib.Path, default=here / "ir")
    parser.add_argument("--dataset", type=pathlib.Path,
                        default=here.parent / ".cache" / "lerobot" / "bimanual_table_setting")
    parser.add_argument("--repo-id", default="local/bimanual_table_setting")
    parser.add_argument("--subset-size", type=int, default=300,
                        help="calibration frames; NNCF's default is 300")
    parser.add_argument("--eval-frames", type=int, default=16,
                        help="frames held out to measure the INT8 vs FP32 divergence")
    parser.add_argument("--preset", choices=("performance", "mixed"), default="mixed",
                        help="'mixed' keeps activations asymmetric, which suits the "
                             "signed action and torque channels")
    args = parser.parse_args()

    import nncf

    if not args.ir.exists():
        raise SystemExit(f"{args.ir} not found -- run openvino/convert.py first")

    core = ov.Core()
    model = core.read_model(args.ir)
    specs = []
    for port in model.inputs:
        shape = port.get_partial_shape()
        if not shape.is_static:
            raise SystemExit(
                f"{args.ir} has a dynamic input shape ({shape}); NNCF cannot size its "
                "calibration tensors against it. Reconvert with openvino/convert.py, "
                "which pins every dimension.")
        names = port.get_tensor().get_names()
        specs.append((next(iter(names)) if names else port.get_node().get_friendly_name(),
                      tuple(shape.to_shape())))
    batch = specs[0][1][0]
    cameras = [n for n, _ in specs if n != STATE_KEY] or list(DEFAULT_CAMERAS)

    print("INT8 post-training quantization")
    print(f"  source IR      : {args.ir}")
    for name, shape in specs:
        print(f"  input          : {name:32s} {shape}")

    total = args.subset_size + args.eval_frames
    have_dataset = (args.dataset / "meta" / "info.json").exists()
    if have_dataset:
        frames, n_total = dataset_frames(args.dataset, args.repo_id, cameras, total, batch)
        print(f"  calibration    : {len(frames)} frames sampled across {n_total} "
              f"in {args.dataset.name}")
    else:
        frames = synthetic_frames(specs, total)
        print(f"  calibration    : {len(frames)} SYNTHETIC frames -- {args.dataset} not found.")
        print("                   Activation ranges will be measured on noise; rerun with")
        print("                   --dataset pointing at the recorded dataset for real ones.")

    calib, held_out = frames[:args.subset_size], frames[args.subset_size:]
    print(f"  preset         : {args.preset}, model_type=TRANSFORMER")

    t0 = time.time()
    quantized = nncf.quantize(
        model,
        nncf.Dataset(calib, lambda item: item),
        subset_size=len(calib),
        preset=(nncf.QuantizationPreset.MIXED if args.preset == "mixed"
                else nncf.QuantizationPreset.PERFORMANCE),
        model_type=nncf.ModelType.TRANSFORMER,
    )
    print(f"  quantized in   : {time.time() - t0:.0f} s")

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "act_int8.xml"
    ov.save_model(quantized, path)
    size = (path.stat().st_size + path.with_suffix(".bin").stat().st_size) / 1024**2
    src = (args.ir.stat().st_size + args.ir.with_suffix(".bin").stat().st_size) / 1024**2
    print(f"  wrote          : {path.name} + .bin  {size:.1f} MiB"
          f"  ({src / size:.2f}x smaller than {args.ir.stem})")

    if held_out:
        fp32 = core.compile_model(model, "CPU")
        int8 = core.compile_model(quantized, "CPU")
        diffs, rels = [], []
        for item in held_out:
            a = fp32(item)[fp32.output(0)]
            b = int8(item)[int8.output(0)]
            diffs.append(float(np.abs(a - b).max()))
            rels.append(float(np.abs(a - b).max() / (np.abs(a).max() + 1e-12)))
        print(f"\ndivergence from FP32 over {len(held_out)} held-out frames:")
        print(f"  max abs  : {max(diffs):.4f}   mean abs : {float(np.mean(diffs)):.4f}")
        print(f"  max rel  : {max(rels) * 100:.2f} %  mean rel : {float(np.mean(rels)) * 100:.2f} %")
        if not have_dataset:
            print("  (calibrated on noise -- not indicative of the trained policy)")


if __name__ == "__main__":
    main()
