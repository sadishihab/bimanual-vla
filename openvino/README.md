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
weights**. That is how the pipeline was developed and tested, since no trained
checkpoint exists yet: random weights give meaningless *behaviour* and exactly as
meaningful *latency*, which is what the benchmark measures. The INT8 divergence
figure is the one number that needs a real checkpoint to mean anything.

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

It reports divergence from FP32 on frames held out of the calibration set. Re-run
it against the trained checkpoint before trusting INT8 in the loop: a percentage
measured on random weights says only that the arithmetic is sane.

## Measured on this machine

Intel Core i7-5500U (Broadwell, 2015), openvino 2026.3.1, batch 1, 2×256×256
cameras, chunk 100, 51.6 M parameters.

```
device     precision size MiB compile s   mean ms    p50 ms    p95 ms      fps nireq
CPU        fp32         131.2      0.70    159.07    152.63    181.94     7.04     4
CPU        fp16          66.0      0.63    164.81    158.35    184.97     6.81     4
CPU        int8          37.8      1.11    169.79    166.69    180.55     6.53     4
```

Parity against the torch module that produced it: FP32 max abs diff 1.0e-06,
FP16 1.0e-03.

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
