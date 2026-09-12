#!/usr/bin/env python3
"""Convert an ACT checkpoint to OpenVINO IR, in FP32 and FP16.

    openvino/convert.py --checkpoint /path/to/step_0100000 --out openvino/ir
    openvino/convert.py --out openvino/ir          # random weights, for testing

Writes act_fp32.xml/.bin and act_fp16.xml/.bin.  FP16 is the same graph saved
with weights compressed to half precision -- it is not a different topology, and
on most OpenVINO devices it is what you want by default: half the file, and the
GPU and NPU execute in fp16 internally regardless.

The graph is self-contained: normalization in, unnormalization out, static
shapes, fixed batch.  See openvino/act_io.py for why.

Conversion goes through OpenVINO's PyTorch frontend, falling back to ONNX if that
frontend trips over an operator.  Both produce the same IR; the fallback exists
because the frontend's operator coverage moves with each release and this has to
keep working on a machine that is not this one.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np
import openvino as ov
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from act_io import (ENV_KEY, add_model_args, describe_normalization,  # noqa: E402
                    divergence_report, example_inputs, input_specs,
                    load_for_export, real_frames, task_embedder)


def convert_via_torch(module, cfg, cameras, batch):
    example = example_inputs(cfg, cameras, batch)
    return ov.convert_model(module, example_input=example)


def convert_via_onnx(module, cfg, cameras, batch, workdir: pathlib.Path):
    example = example_inputs(cfg, cameras, batch)
    names = [n for n, _ in input_specs(cfg, cameras, batch)]
    onnx_path = workdir / "act.onnx"
    workdir.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        module, example, str(onnx_path),
        input_names=names, output_names=["action"],
        opset_version=17, dynamo=False,
    )
    return ov.convert_model(onnx_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(parser)
    parser.add_argument("--out", type=pathlib.Path,
                        default=pathlib.Path(__file__).resolve().parent / "ir")
    parser.add_argument("--via", choices=("auto", "torch", "onnx"), default="auto")
    parser.add_argument("--check-tolerance", type=float, default=2e-3,
                        help="max allowed abs difference between torch and IR outputs")
    parser.add_argument("--check-frames", type=int, default=16,
                        help="real dataset frames to check parity on")
    args = parser.parse_args()

    module, cfg, cameras, source = load_for_export(args)
    print(f"ACT -> OpenVINO IR")
    print(f"  weights        : {args.checkpoint or 'randomly initialized (no checkpoint)'}")
    print(f"  statistics     : {source}")
    print(f"  cameras        : {', '.join(cameras)}")
    if cfg.env_state_feature:
        print(f"  language       : conditioned, {cfg.env_state_feature.shape[0]}-d task"
              f" embedding as an extra encoder token")
    else:
        print(f"  language       : none (unconditioned policy)")
    print(f"  chunk / batch  : {cfg.chunk_size} / {args.batch}")
    describe_normalization(module, cfg)
    for name, shape in input_specs(cfg, cameras, args.batch):
        print(f"  input          : {name:32s} {shape}")

    example = example_inputs(cfg, cameras, args.batch)
    with torch.inference_mode():
        probe = module(*example).cpu().numpy()
    print(f"  output         : action {tuple(probe.shape)}")

    t0 = time.time()
    ov_model, how = None, None
    if args.via in ("auto", "torch"):
        try:
            ov_model, how = convert_via_torch(module, cfg, cameras, args.batch), "pytorch frontend"
        except Exception as exc:                                   # noqa: BLE001
            if args.via == "torch":
                raise
            print(f"  pytorch frontend failed ({type(exc).__name__}: {exc}); trying ONNX")
    if ov_model is None:
        ov_model, how = convert_via_onnx(module, cfg, cameras, args.batch, args.out), "onnx"
    print(f"  converted via  : {how} in {time.time() - t0:.1f} s")

    # Name the inputs and outputs so a consumer does not have to rely on order.
    specs = input_specs(cfg, cameras, args.batch)
    for port, (name, _) in zip(ov_model.inputs, specs):
        port.get_node().set_friendly_name(name)
        port.get_tensor().set_names({name})
    ov_model.outputs[0].get_tensor().set_names({"action"})

    # Pin every dimension.  The frontend leaves batch (and sometimes more) dynamic,
    # and dynamic shapes are not merely untidy here: the NPU plugin refuses them
    # outright, NNCF cannot size its calibration tensors against them, and the
    # traced graph is only valid for the shapes it was traced at anyway -- ACT
    # iterates over its camera list and its position embeddings in Python, which
    # is what the TracerWarnings above are about.
    ov_model.reshape({i: ov.PartialShape(list(shape)) for i, (_, shape) in enumerate(specs)})
    static = all(port.get_partial_shape().is_static for port in ov_model.inputs)
    print(f"  shapes         : {'static' if static else 'STILL DYNAMIC'}"
          f" {[tuple(p.get_partial_shape().to_shape()) for p in ov_model.inputs]}")
    if not static:
        raise SystemExit("inputs stayed dynamic; the NPU plugin will reject this IR")

    args.out.mkdir(parents=True, exist_ok=True)
    written = []
    for precision, compress in (("fp32", False), ("fp16", True)):
        path = args.out / f"act_{precision}.xml"
        ov.save_model(ov_model, path, compress_to_fp16=compress)
        size = (path.stat().st_size + path.with_suffix(".bin").stat().st_size) / 1024**2
        written.append((precision, path, size))
        print(f"  wrote          : {path.name} + .bin  {size:.1f} MiB")

    # Parity against the torch module that produced it, on CPU.
    #
    # Checked on REAL dataset frames.  Checking on synthetic noise is worse than
    # useless: uniform images are far outside the training distribution, the network
    # answers them from a flat region of its response, and a precision that is badly
    # wrong on real observations can still look exact there.  Measured on the
    # 44k-step checkpoint, the fp16 IR came out at 1.2e-03 against synthetic input
    # and 1.9e+00 against real frames -- three orders of magnitude apart.
    root = args.dataset if args.dataset and (args.dataset / "meta" / "info.json").exists() else None
    env_key = ENV_KEY if cfg.env_state_feature else None
    embed = label = None
    if env_key is not None:
        embed, label = task_embedder(args.checkpoint, cfg.env_state_feature.shape[0])
    if root is not None:
        frames = real_frames(root, args.repo_id, cameras, args.check_frames, args.batch,
                             env_key=env_key, embed=embed)
        print(f"\nparity against torch, on {len(frames)} real frames from {root.name}:")
        if label:
            print(f"  task embeddings from: {label}")
    else:
        names = [n for n, _ in input_specs(cfg, cameras, args.batch)]
        frames = [dict(zip(names, [e.numpy() for e in example]))]
        print(f"\nparity against torch, on 1 SYNTHETIC frame ({args.dataset} not found).")
        print("  This is a weak check -- point --dataset at the recorded dataset for a real one.")

    with torch.inference_mode():
        order = [n for n, _ in input_specs(cfg, cameras, args.batch)]
        reference = np.concatenate(
            [module(*[torch.from_numpy(f[n]) for n in order]).cpu().numpy() for f in frames])

    core = ov.Core()
    failures = []
    for precision, path, _ in written:
        compiled = core.compile_model(core.read_model(path), "CPU")
        got = np.concatenate([compiled(f)[compiled.output(0)] for f in frames])
        worst = divergence_report(reference, got, precision)
        if precision == "fp32" and worst > args.check_tolerance:
            failures.append((precision, worst))
        elif precision != "fp32" and worst > max(args.check_tolerance, 5e-2):
            print(f"      WARNING: {precision} diverges by {worst:.3f} from the torch model."
                  f" That is a precision trade-off, not a conversion error, but do not")
            print(f"      deploy {precision} on this model without checking it closed-loop.")

    if failures:
        raise SystemExit(
            "fp32 IR does not match the torch model: "
            + ", ".join(f"{p} off by {d:.2e}" for p, d in failures)
            + " -- conversion is wrong, not merely imprecise.")
    print("\nIR written to", args.out)


if __name__ == "__main__":
    main()
