#!/usr/bin/env python3
"""Benchmark the ACT IR on every OpenVINO device this machine exposes.

    openvino/benchmark.py                       # every device, every IR found
    openvino/benchmark.py --devices CPU NPU     # only these, skipping absent ones
    openvino/benchmark.py --iters 200

Discovers devices from the runtime rather than assuming any, so the same file runs
unchanged on a CPU-only box and on an Intel Core Ultra Series 2/3 part that
exposes CPU, GPU and NPU.  A device that is absent is reported absent; a device
that is present but cannot compile this graph is reported with the reason and the
run continues.  Nothing here is conditional on a device existing.

Two numbers, because they answer different questions and OpenVINO tunes for them
differently:

  latency     one inference at a time, PERFORMANCE_HINT=LATENCY.  What a control
              loop waits for.  Reported as mean, p50 and p95 -- the tail is the
              part that breaks a 25 Hz loop, and a mean alone hides it.
  throughput  many in flight, PERFORMANCE_HINT=THROUGHPUT, using the plugin's own
              OPTIMAL_NUMBER_OF_INFER_REQUESTS.  What a batch evaluation gets.

Compile time is reported too: on the NPU it is not incidental, since the plugin
compiles a blob for the device on first load.
"""

from __future__ import annotations

import argparse
import pathlib
import statistics
import time
from typing import Dict, List, Optional

import numpy as np
import openvino as ov

# Virtual plugins dispatch onto the real ones; benchmarking them measures the
# dispatcher's policy, not a device, so they are excluded unless asked for.
VIRTUAL = ("AUTO", "MULTI", "HETERO", "BATCH")
PRECISIONS = ("fp32", "fp16", "int8")


def device_label(core: ov.Core, device: str) -> str:
    try:
        return str(core.get_property(device, "FULL_DEVICE_NAME"))
    except Exception:                                              # noqa: BLE001
        return "<name unavailable>"


def make_inputs(model, seed: int = 0) -> Dict[str, np.ndarray]:
    """One static batch matching the IR's inputs.  Images in [0, 1], state signed."""
    rng = np.random.default_rng(seed)
    out = {}
    for port in model.inputs:
        names = port.get_tensor().get_names()
        name = next(iter(names)) if names else port.get_node().get_friendly_name()
        shape = tuple(port.get_partial_shape().to_shape())
        x = rng.random(shape, dtype=np.float32)
        out[name] = x * 2.0 - 1.0 if len(shape) == 2 else x
    return out


def measure_latency(core, model, device, feed, iters, warmup) -> dict:
    t0 = time.perf_counter()
    compiled = core.compile_model(model, device, {"PERFORMANCE_HINT": "LATENCY"})
    compile_s = time.perf_counter() - t0
    request = compiled.create_infer_request()
    for _ in range(warmup):
        request.infer(feed)
    samples = []
    for _ in range(iters):
        t = time.perf_counter()
        request.infer(feed)
        samples.append((time.perf_counter() - t) * 1000.0)
    samples.sort()
    return {
        "compile_s": compile_s,
        "mean_ms": statistics.fmean(samples),
        "p50_ms": float(np.percentile(samples, 50)),
        "p95_ms": float(np.percentile(samples, 95)),
        "min_ms": samples[0],
        "iters": iters,
    }


def measure_throughput(core, model, device, feed, iters) -> dict:
    compiled = core.compile_model(model, device, {"PERFORMANCE_HINT": "THROUGHPUT"})
    try:
        nireq = int(compiled.get_property("OPTIMAL_NUMBER_OF_INFER_REQUESTS"))
    except Exception:                                              # noqa: BLE001
        nireq = 1
    nireq = max(1, nireq)
    queue = ov.AsyncInferQueue(compiled, nireq)
    batch = tuple(model.inputs[0].get_partial_shape().to_shape())[0]
    for _ in range(nireq):                                          # warm every request
        queue.start_async(feed)
    queue.wait_all()
    t = time.perf_counter()
    for _ in range(iters):
        queue.start_async(feed)
    queue.wait_all()
    elapsed = time.perf_counter() - t
    return {"fps": iters * batch / elapsed, "requests": nireq, "elapsed_s": elapsed}


def main() -> None:
    here = pathlib.Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ir-dir", type=pathlib.Path, default=here / "ir")
    parser.add_argument("--precisions", nargs="+", default=list(PRECISIONS),
                        choices=list(PRECISIONS))
    parser.add_argument("--devices", nargs="+", default=None,
                        help="restrict to these; absent ones are reported and skipped")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--throughput-iters", type=int, default=None,
                        help="defaults to --iters")
    parser.add_argument("--include-virtual", action="store_true",
                        help="also benchmark AUTO/MULTI/HETERO/BATCH")
    args = parser.parse_args()

    core = ov.Core()
    present = list(core.available_devices)
    print(f"openvino {ov.__version__}")
    print(f"devices exposed by this machine: {', '.join(present) or '(none)'}")
    for d in present:
        print(f"  {d:8s} {device_label(core, d)}")

    if args.devices:
        wanted = [d.upper() for d in args.devices]
        # Match GPU against GPU.0 etc., so --devices GPU works on a multi-GPU box.
        selected, absent = [], []
        for w in wanted:
            hits = [d for d in present if d == w or d.startswith(w + ".")]
            selected.extend(hits) if hits else absent.append(w)
        for d in absent:
            print(f"  {d:8s} not present on this machine -- skipped")
    else:
        selected = [d for d in present
                    if args.include_virtual or not d.split(".")[0] in VIRTUAL]
        absent = [d for d in ("CPU", "GPU", "NPU") if
                  not any(p == d or p.startswith(d + ".") for p in present)]
        for d in absent:
            print(f"  {d:8s} not present on this machine -- skipped")
    if not selected:
        raise SystemExit("no usable device selected")

    irs = []
    for precision in args.precisions:
        path = args.ir_dir / f"act_{precision}.xml"
        if path.exists():
            mib = (path.stat().st_size + path.with_suffix(".bin").stat().st_size) / 1024**2
            irs.append((precision, path, mib))
        else:
            print(f"  {precision:8s} IR not found at {path.name} -- skipped")
    if not irs:
        raise SystemExit(f"no IR in {args.ir_dir}; run openvino/convert.py first")

    tp_iters = args.throughput_iters or args.iters
    print(f"\n{args.iters} timed iterations after {args.warmup} warmup, "
          f"{tp_iters} for throughput\n")
    header = (f"{'device':10s} {'precision':9s} {'size MiB':>8s} {'compile s':>9s} "
              f"{'mean ms':>9s} {'p50 ms':>9s} {'p95 ms':>9s} {'fps':>8s} {'nireq':>5s}")
    print(header)
    print("-" * len(header))

    rows, skipped = [], []
    for device in selected:
        for precision, path, mib in irs:
            model = core.read_model(path)
            feed = make_inputs(model)
            try:
                lat = measure_latency(core, model, device, feed, args.iters, args.warmup)
                thr = measure_throughput(core, model, device, feed, tp_iters)
            except Exception as exc:                               # noqa: BLE001
                reason = str(exc).strip().splitlines()[-1][:110] if str(exc).strip() else type(exc).__name__
                skipped.append((device, precision, f"{type(exc).__name__}: {reason}"))
                print(f"{device:10s} {precision:9s} {mib:8.1f} "
                      f"{'--':>9s} {'--':>9s} {'--':>9s} {'--':>9s} {'--':>8s} {'--':>5s}"
                      f"   skipped")
                continue
            rows.append((device, precision, mib, lat, thr))
            print(f"{device:10s} {precision:9s} {mib:8.1f} {lat['compile_s']:9.2f} "
                  f"{lat['mean_ms']:9.2f} {lat['p50_ms']:9.2f} {lat['p95_ms']:9.2f} "
                  f"{thr['fps']:8.2f} {thr['requests']:5d}")

    if skipped:
        print("\nskipped combinations:")
        for device, precision, reason in skipped:
            print(f"  {device} / {precision}: {reason}")

    if rows:
        best = min(rows, key=lambda r: r[3]["p95_ms"])
        print(f"\nlowest p95 latency: {best[0]} {best[1]} at {best[3]['p95_ms']:.2f} ms"
              f"  ({1000.0 / best[3]['p95_ms']:.1f} Hz)")
        fastest = max(rows, key=lambda r: r[4]["fps"])
        print(f"highest throughput: {fastest[0]} {fastest[1]} at {fastest[4]['fps']:.2f} fps")
        # The expert was recorded at 25 Hz, so that is the bar a closed loop has to clear.
        print("\nagainst the 25 Hz the demonstrations were recorded at (40 ms budget):")
        for device, precision, _, lat, _ in rows:
            head = "meets" if lat["p95_ms"] <= 40.0 else "misses"
            print(f"  {device:8s} {precision:5s} p95 {lat['p95_ms']:8.2f} ms  {head}")


if __name__ == "__main__":
    main()
