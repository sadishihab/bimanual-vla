# ACT on OpenVINO

Takes a trained ACT checkpoint to OpenVINO IR in FP32, FP16 and INT8, and
benchmarks it on whatever devices the machine exposes.

| file | needs | what it does |
|---|---|---|
| `act_io.py` | lerobot, torch | shared: builds the policy, resolves statistics, wraps ACT for export |
| `convert.py` | lerobot, torch, openvino | checkpoint → `act_fp32.xml` + `act_fp16.xml` |
| `quantize.py` | openvino, nncf, lerobot | `act_fp32.xml` → `act_int8.xml`, calibrated on dataset frames |
| `benchmark.py` | **openvino, numpy only** | latency and throughput on every device present |

`benchmark.py` deliberately imports nothing from lerobot or torch, so it runs on a
deployment machine that has only the OpenVINO runtime and the IR.

## Quick start

```sh
PY=~/.cache/lerobot-verify/venv/bin/python      # an env with lerobot 0.4.4 + openvino + nncf

$PY openvino/convert.py  --checkpoint /path/to/step_0100000 --out openvino/ir
$PY openvino/quantize.py --ir openvino/ir/act_fp32.xml --out openvino/ir
$PY openvino/benchmark.py
```

Omit `--checkpoint` and the architecture is built with **randomly initialized
weights** — useful for developing the pipeline before a checkpoint exists, since
random weights give meaningless behaviour and exactly as meaningful latency.

All numbers below are from the real checkpoint: **`checkpoints/step_0044000`,
44,000 steps, final loss 0.09 (L1 0.09)**.

## Two decisions worth knowing

**The IR is self-contained.** Normalization of the inputs and unnormalization of
the action are compiled into the graph, so it takes raw observations — images in
[0, 1], joint positions in radians — and returns an action in real units. A
deployment target like the NPU should not need a Python preprocessing pipeline
beside it, and this dataset's action mixes units: ten arm channels in radians and
two gripper channels (5 and 11) in newton-metres. The per-channel statistics are
the last thing you want reimplemented by hand at the edge. Folded into the graph
they cannot drift, and `convert.py` prints the two groups' scales so you can see
they were kept apart:

```
action scales: 10 arm channels (rad) std 0.4058-1.0211,
                2 gripper channels (N.m) std 0.5971-0.6085
```

**Every shape is static.** The PyTorch frontend leaves the batch dimension (and
sometimes more) dynamic, and `convert.py` pins all of it. This is not tidiness:
the NPU plugin rejects dynamic shapes outright, NNCF cannot size calibration
tensors against them, and the traced graph was only ever valid at the shapes it
was traced at — ACT iterates over its camera list and position embeddings in
Python, which is what the tracer warnings during conversion are about. Conversion
fails loudly if anything stayed dynamic.

`--batch` changes the compiled batch size; the default is 1, which is what a
control loop uses.

## Quantization

`quantize.py` calibrates on real frames sampled evenly across the whole recorded
dataset — `np.linspace` over all 47,092 frames, not the first N, which would all
come from one episode of one seed. Because the graph normalizes internally,
calibration feeds it raw observations exactly as deployment will; pre-normalized
calibration data would set every activation range against the wrong input scale.

`model_type=TRANSFORMER` is passed deliberately. ACT is a transformer behind two
ResNet18 backbones, and that preset keeps the quantization-sensitive parts of
attention out of INT8. The default `--preset mixed` keeps activations asymmetric,
which suits a signed action vector and the signed torque channels.

It reports divergence from FP32 on frames held out of the calibration set, split
by unit group, because the arm tolerates absolute error far better than a torque
channel whose whole span is under 2 N·m.

## Accuracy of the trained checkpoint

Parity and divergence against the PyTorch module, on **real dataset frames**.

| precision | size | mean abs err | arm (rad) | gripper (N·m) | gripper sign flips |
|---|---|---|---|---|---|
| FP32 | 131.3 MiB | 0.00000 | max 0.00000 | max 0.00001 | 0 / 4800 = 0.00% |
| FP16 | 66.1 MiB | **0.14223** | mean 0.126, p99 1.13, max 2.35 | mean 0.223, max 1.88 | **717 / 4800 = 14.94%** |
| INT8 | 38.0 MiB | 0.0161 | mean 0.015, p99 0.137, max 0.58 | mean 0.020, max 1.22 | **118 / 12800 = 0.92%** |

A *sign flip* on a gripper channel turns close into open. It is the metric that
matters for behaviour, and a mean does not show it.

**Ship FP32.** INT8 is defensible with a closed-loop check: the arm is good to
~0.9° mean and 7.8° at p99, but just under 1% of gripper commands invert.
**FP16 should not be deployed on this model at all** — 15% of gripper commands
invert and the arm's p99 error is 1.1 rad.

### The FP16 result is not a rounding cost

Rounding to fp16 is nearly free for this model; OpenVINO's `compress_to_fp16` is
not. Rounding the network weights through fp16 *in PyTorch* costs a max of
**0.0056**, and rounding the folded normalization constants **0.0054** — but the
compressed IR is off by **2.35**, three hundred times more. Ruled out along the
way:

- **Not execution precision.** The CPU plugin reports `f32` for all three IRs, and
  forcing `INFERENCE_PRECISION_HINT=f32` changes nothing.
- **Not overflow or underflow.** No constant became non-finite, and none that was
  non-zero was zeroed.
- **Not a bug in this converter.** Compressing a freshly read FP32 IR gives the
  identical error, and re-saving FP32 from the same handle is still exact, so
  `save_model` is not mutating a shared model between the two calls.

What remains is inside the compression pass — most likely constants baked into the
traced graph that are neither torch parameters nor the normalization buffers, such
as the position-embedding tables. That was not chased further; the finding that
matters is that FP16 costs this model two orders of magnitude more than fp16
arithmetic implies, and a deployment must measure rather than assume.

### Parity has to be checked on real frames

This was nearly missed. The original check ran on one synthetic uniform-noise
frame, where the FP16 IR looked fine at **1.2e-03**. On real frames the same IR is
off by **1.9e+00** — three orders of magnitude worse. Uniform images are far
outside the training distribution and the network answers them from a flat part of
its response, so a badly wrong precision can look exact there. `convert.py` now
checks against dataset frames and only falls back to synthetic input, loudly, when
no dataset is given.

### Keeping the action head in FP does not help

The obvious first move when INT8 hurts accuracy is to exclude the output head,
which is the smallest projection in the model and lands directly on the action.
Measured, it buys almost nothing: worst gripper error **1.2184** with the head in
FP against **1.2190** with it quantized. So the error originates upstream, in the
backbone and transformer, and the head merely passes it through. Quantizing
everything is the default; `--keep-head-fp32` reproduces the comparison.

## Latency measured on this machine

Intel Core i7-5500U (Broadwell, 2015), openvino 2026.3.1, batch 1, 2×256×256
cameras, chunk 100, 51.6 M parameters, 50 timed iterations after 5 warmup.

```
device     precision size MiB compile s   mean ms    p50 ms    p95 ms      fps nireq
CPU        fp32         131.3      0.76    156.38    150.47    182.69     6.93     4
CPU        fp16          66.1      0.74    163.67    158.32    182.51     6.70     4
CPU        int8          38.0      1.23    172.75    166.72    199.26     6.59     4
```

**INT8 is not faster here, only smaller** (3.47×). That is the expected result on
this part, not a bug: Broadwell has no VNNI, so INT8 dot products get no
hardware path and the extra quantize/dequantize nodes cost about what the narrower
arithmetic saves. INT8 speedups need VNNI (Cascade Lake and later), AMX, or the
NPU — all of which a Core Ultra has and this machine does not.

## On an Intel Core Ultra Series 2/3

`benchmark.py` runs unmodified. It reads `Core().available_devices` rather than
assuming anything, so a Core Ultra reports CPU, GPU and NPU and all three are
benchmarked; on this machine the same command reports GPU and NPU absent and
skips them. A device that is present but cannot compile the graph is reported
with its reason and the run continues.

What to expect there, stated as expectation and not measurement — none of it was
measured, because this machine has no such device:

- **NPU** should be the interesting target for a control loop, and the one that
  most needs INT8: it is built for low-precision inference at low power. Its
  compile time is also the one worth watching, since the plugin compiles a blob on
  first load — the `compile s` column exists for that.
- **GPU (Arc integrated)** executes in fp16 internally regardless of the IR's
  precision, so `act_fp16.xml` is the natural input and FP32 buys nothing but file
  size.
- **CPU** on those parts has VNNI, so unlike here INT8 should actually be faster
  than FP32, not merely smaller.

The benchmark closes by checking p95 against a 40 ms budget, which is the 25 Hz
the demonstrations were recorded at — the bar a closed loop has to clear. This
Broadwell misses it by more than 4x at 181 ms p95.

## Caveat on `select_action`

The exported graph is ACT's forward pass: one observation in, a full
`chunk_size`-long action chunk out. It is not `ACTPolicy.select_action`, which
additionally maintains an action queue across calls and can apply temporal
ensembling — that is stateful Python and does not belong in a traced graph. If you
deploy with temporal ensembling, the queue and the ensembling stay on the host and
the IR supplies the chunks.
