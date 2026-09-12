# Training ACT on the bimanual demonstrations, on Kaggle

`train_act_kaggle.ipynb` trains an [ACT](https://tonyzhaozh.github.io/aloha/)
policy on the scripted-expert demonstrations recorded by
`scripts/record_demos.py`. It is written for **one 16 GB T4**, and for Kaggle's
session limits in particular: it checkpoints to `/kaggle/working` and resumes
from the newest checkpoint, so a restart costs you one checkpoint interval
rather than the whole run.

## What you are uploading

`~/bimanual-demos.tar.gz` — **205 MB** (195.6 MiB) — a gzipped
`LeRobotDataset` v3.0:

| | |
|---|---|
| episodes | 134 (one prop's pick/place, handovers included) |
| frames | 47,092 at 25 Hz (31.4 min) |
| cameras | `overhead`, `front`, 256x256 |
| state / action | 12 joints in, 12 actuator commands out |
| tasks | 7 natural-language strings |

Only successful manipulations are in it — the prop finished within 20 mm of its
slot and resting on the table. It is not a dataset of recoveries.

## Steps

**1. Upload the archive as a Kaggle Dataset.**
[kaggle.com/datasets](https://www.kaggle.com/datasets) → **New Dataset** → drag
in `bimanual-demos.tar.gz` → name it something like `bimanual-demos` → **Create**.
Upload the archive *as the file*; do not let Kaggle unzip it for you, and do not
extract it yourself first — the notebook untars it, and a Dataset of ~20,000
loose frames would be far slower to attach.

**2. Create the notebook.** [kaggle.com/code](https://www.kaggle.com/code) → **New
Notebook** → **File → Import Notebook** → upload `train_act_kaggle.ipynb`.

**3. Attach the data.** In the right-hand panel, **+ Add Input → Datasets →** your
`bimanual-demos`. It mounts under `/kaggle/input/<slug>/`. The notebook finds the
archive by name anywhere under `/kaggle/input`, so the slug does not matter.

**4. Pick the accelerator.** **Settings → Accelerator → GPU T4 x2**.

Choose T4 x2 even though the notebook uses a single GPU: it is the option that
reliably gives a 16 GB T4, and the second card simply sits idle. **P100** also
works (16 GB, somewhat faster for this shape). **TPU** will not work — the code
is CUDA-only. Also set **Settings → Persistence → Variables and Files** if you
want `/kaggle/working` to survive between sessions without re-downloading.

**5. Turn on internet.** **Settings → Internet → On**. The first cell pip-installs
`lerobot==0.4.4` and torchvision fetches ImageNet ResNet18 weights.

**6. Run all cells.** Sections 1–6 take a few minutes and are all checks; section
7 is the training loop.

## Expected runtime

| stage | time |
|---|---|
| `pip install lerobot==0.4.4` | 4–10 min (it pulls its own torch build, ~2.5 GB) |
| extracting the archive | under 1 min |
| loading + the verification cells | 1–2 min |
| training | see below |

The loop prints `it/s` and a live ETA every 100 steps — trust that over any
estimate here. Expect roughly **1.5–4 it/s** at batch 16, so the built-in 8.5 h
budget buys about **45k–120k steps**. `TOTAL_STEPS` defaults to 100,000, which is
a full ACT schedule and will likely take **two sessions**. Kaggle allows 12 h per
session and 30 GPU-hours per week.

You do not have to go the full distance. 134 episodes is a small dataset and the
L1 term usually looks reasonable within the first 10–20k steps; checkpoints land
every 2,000, so you can stop whenever the loss curve flattens.

If `it/s` is low **and** GPU utilisation is low, the bottleneck is video decoding,
not the GPU: the frames are AV1, and Kaggle gives only 4 vCPUs. Raising
`NUM_WORKERS` to 4 is the first thing to try; lowering `BATCH_SIZE` will not help.

## Sizing, and why these numbers

- `BATCH_SIZE = 16` — two 256x256 streams through two ResNet18 backbones. Measured
  at ~0.58 GiB per checkpoint and comfortably inside 16 GB. **On OOM, drop to 8**;
  that is the only change needed.
- `NUM_WORKERS = 3` — 4 vCPUs, one left for the main process.
- `CHUNK_SIZE = 100` — ACT's default, and 4.0 s of action horizon at 25 Hz against
  episodes that run 11–27 s.
- `USE_AMP = True` — fp16 with a `GradScaler`. ACT's VAE loss needs the scaler;
  do not switch to plain fp16 without it.
- `KEEP_CKPTS = 2` — each checkpoint is ~0.58 GiB and `/kaggle/working` is capped
  at 20 GB.

## The mixed-unit action, and normalization

The action vector is **not one quantity**:

| channels | meaning | unit |
|---|---|---|
| 0–4, 6–10 | arm joint position targets | rad |
| 5, 11 | gripper torque, negative closes | N·m |

The gripper is force-controlled, because a commanded jaw angle relaxes its grip
as the servo settles and drops the prop. `observation.state` is homogeneous — all
12 are joint positions in radians — so only the action needs care.

LeRobot normalizes with **per-element** statistics: `stats['action']['mean']` is a
12-vector and `NormalizerProcessorStep` holds it as a tensor of that shape, so
each channel is standardized against its own mean and spread and the two units
are never pooled. Sections 4 and 6 of the notebook assert this rather than
assuming it — section 4 on the stored statistics, section 6 empirically on a real
batch, checking that the radian channels *and* the torque channels both come out
standardized rather than one of them being squashed.

That matters because a single shared scale would be actively harmful here: the
radian channels span about ±2.8 while the torques span only [−0.8, 1.0], so a
pooled scale would flatten the gripper channel to near-constant and the policy
would never learn to open or close. If you change `normalization_mapping`, keep it
per-feature-type and keep the statistics per-element; the section 6 assertion will
catch you if you do not.

One honest caveat the checks do not fix: the torque channels are nearly
*discrete* — the expert commands about three levels (open, hold, close) plus the
ramps between them — so a policy regressing them with an L1 loss is fitting a
near-categorical target. It trains, but if the learned grasp is unreliable at
evaluation time, that is the first thing to look at.

## What you get out

Each `/kaggle/working/act_bimanual/step_*` directory is a self-contained lerobot
checkpoint — policy weights, config, and the pre/post-processors with their
normalization buffers baked in:

```python
from lerobot.policies.act.modeling_act import ACTPolicy
policy = ACTPolicy.from_pretrained("/kaggle/working/act_bimanual/step_0100000")
```

Download it from the notebook's **Output** tab. There is no evaluation in this
notebook: closing the loop needs the MuJoCo scene from this repo, which is not
part of the dataset.
