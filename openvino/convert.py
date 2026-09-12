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
from act_io import (add_model_args, describe_normalization,  # noqa: E402
                    example_inputs, input_specs, load_for_export)


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
    args = parser.parse_args()

    module, cfg, cameras, source = load_for_export(args)
    print(f"ACT -> OpenVINO IR")
    print(f"  weights        : {args.checkpoint or 'randomly initialized (no checkpoint)'}")
    print(f"  statistics     : {source}")
    print(f"  cameras        : {', '.join(cameras)}")
    print(f"  chunk / batch  : {cfg.chunk_size} / {args.batch}")
    describe_normalization(module, cfg)
    for name, shape in input_specs(cfg, cameras, args.batch):
        print(f"  input          : {name:32s} {shape}")

    example = example_inputs(cfg, cameras, args.batch)
    with torch.inference_mode():
        reference = module(*example).cpu().numpy()
    print(f"  output         : action {tuple(reference.shape)}")

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

    # Numerical check against the torch module that produced it, on CPU.
    core = ov.Core()
    print("\nparity against torch (CPU):")
    for precision, path, _ in written:
        compiled = core.compile_model(core.read_model(path), "CPU")
        got = compiled([e.numpy() for e in example])[compiled.output(0)]
        diff = float(np.abs(got - reference).max())
        rel = diff / float(np.abs(reference).max() + 1e-12)
        tol = args.check_tolerance if precision == "fp32" else max(args.check_tolerance, 5e-2)
        status = "ok" if diff <= tol else "OUT OF TOLERANCE"
        print(f"  {precision}: max abs diff {diff:.2e}  (rel {rel:.2e}, tol {tol:.0e})  {status}")
        if diff > tol:
            raise SystemExit(f"{precision} IR does not match the torch model")
    print("\nIR written to", args.out)


if __name__ == "__main__":
    main()
