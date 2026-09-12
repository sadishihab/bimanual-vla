# Bimanual table setting: SO-101 in MuJoCo, ACT, and OpenVINO

Two SO-101 arms set a table in MuJoCo. A scripted expert picks four props out of a
randomized layout and places them into a place setting, handing a prop from one arm
to the other when no single arm can both reach it and reach its slot. Successful
manipulations are recorded as a LeRobot v3.0 dataset, which an ACT policy trains
on, which is then converted to OpenVINO IR and quantized for Intel inference
hardware.

Everything in this repo is driven by measurement. Where a primitive fails, the
mechanism and the number are recorded rather than the failure being tuned away, and
several of the design decisions below exist only because a measurement forced them.

## What was measured, and what was not

| | status |
|---|---|
| Gripper geometry, jaw pads, grasp envelope | **measured** in MuJoCo |
| Reach and static-torque envelope | **measured** |
| Pick success, 4 props x 10 seeds | **measured**: 23/40 → 37/40 |
| In-air handover feasibility | **measured** — and ruled out |
| Full-task success, 4 props x 10 seeds | **measured**: 24/40 props, 0/10 complete settings |
| Demonstration dataset | **built**: 134 episodes, 47,092 frames |
| ACT training | **run**: 44,000 steps, final loss 0.09 (L1 0.09) |
| OpenVINO FP32/FP16/INT8 conversion | **measured** on the trained checkpoint, against real frames |
| INT8 / FP16 accuracy | **measured**: INT8 usable with care, **FP16 is not** |
| Closed-loop policy success | **measured**: 3/40 props (2 earned) against the expert's 24/40 |
| OpenVINO CPU latency | **measured** on a Broadwell i7-5500U |
| OpenVINO GPU / NPU latency | **not measured.** No such device was available — see [Intel hardware](#intel-hardware-mapping) |

The caveat that remains: latency on Intel GPU and NPU is unmeasured, for the reason
in [Intel hardware](#intel-hardware-mapping). The policy *is* now evaluated in
closed loop, and it does far worse than the expert — see
[closed-loop evaluation](#closed-loop-evaluation).

## Architecture

```
scenes/bimanual_table.xml     dual SO-101, table, overhead + front cameras
envs/scene.py                 loads the scene and converts the gripper to force control
envs/randomize.py             seeded domain randomization; the reach map
envs/task.py                  the place-setting goal, and which props cross the midline
control/ik.py                 damped-least-squares IK for one arm
control/gripper.py            gripper force control and the measured jaw pads
control/primitives.py         pick, place, handover, and the planners behind them
scripts/                      per-seed runners, sweeps, and the demo recorder
notebooks/train_act_kaggle.ipynb   ACT training on Kaggle
openvino/                     IR conversion, INT8 quantization, device benchmark
```

The data path:

```
MuJoCo scene ─► scripted expert ─► record_demos.py ─► pack_lerobot.py ─► LeRobotDataset v3.0
                                                                              │
                                                   ACT (lerobot 0.4.4) ◄──────┘
                                                         │
                            convert.py ─► FP32 / FP16 IR ─► quantize.py ─► INT8 IR
                                                         │
                                                  benchmark.py ─► CPU / GPU / NPU
```

Two rules run through the whole repo, both from hard experience: **one seed per
process** (sweep loops live in shell scripts, not Python, so each case is a fresh
interpreter with a fresh model), and **no renderer unless the output genuinely needs
pixels** — memory has been the recurring failure here. The demo recorder is the only
thing that builds a renderer, and it frees it before exit.

## The SO-101 gripper, and what it forced

Almost every prop dimension and every primitive detail in this repo traces back to
four findings about the actual gripper.

**1. The collision hull is not the finger.** MuJoCo collides a mesh geom as its
convex hull, and both SO-101 jaws are concave parts. On the fixed finger the hull
bridges from the wrist mount to the fingertip as a single **1428 mm² facet tilted
21.7°** off the opening axis. Of its 606 facets facing the aperture, **not one is
within 5° of parallel**. Grasping against that ramp touches only the object's top
edge and drives it downward — measured at **2.07 N onto the table against a 0.30 N
fork** — and it does so at *any* grasp height, because the ramp is fixed to the tool
frame and moves with it. No choice of grasp height fixes a wedge.

The fix was not to invent geometry. The real inner faces under the hull are flat and
parallel: fitting planes through the mesh vertices gives **0.23° and 0.24°** off the
opening axis with a **0.33 mm residual**, 11.5 mm wide, ~10 mm deep, **15.80 mm
apart**. Each jaw gets an explicit pad laid on the face that is already there, placed
from that measurement rather than typed in.

**2. The tool site is not the middle of the aperture.** It sits on the *fixed*
finger, so the aperture spans `[site, site + opening]` along the site's +z axis.
Centring the site on an object drops the fixed finger straight onto it. Grasps are
therefore offset back along the opening axis by half the object's width plus a 4 mm
clearance, so the jaws straddle the object instead. This offset is 10 mm for the
flatware and **22 mm for the mug** — two cells of the reach map's 10 mm grid — and it
comes back as a bug three more times in this repo (see [reach](#the-reach-and-torque-envelope)).

**3. Position control cannot hold a grasp.** A commanded jaw angle relaxes its grip
as the servo settles, and the prop falls out during the lift. The gripper is driven
in **torque** instead: `OPEN_TORQUE = +1.0 N·m` runs the jaw back to its stop,
`GRIP_TORQUE = -0.8 N·m` is the closing sweep, and `HOLD_TORQUE = -0.20 N·m` is what
is left on afterwards.

Those are three different numbers because they do three different jobs, and getting
that wrong destroys the grasp in both directions. The closing torque has to sweep the
jaw in from its stop against 0.6 N·m·s/rad of joint damping; left on, it crushes
through whatever is soft enough. How soft depends on the object, because MuJoCo's
time-constant `solref` makes contact stiffness scale with effective mass at the
contact: the 136 mm spoon stopped **2.5 mm into a 12 mm bar**, while the 65 mm one,
with a fifth of the rotational inertia, went **straight through to the hard stop and
ejected the piece**.

Too *little* holding torque drops the prop; too much ejects it, because the jaws
close past parallel on anything narrower than their 15.80 mm reference gap and the
resulting wedge flicks a light piece out. Measured on the 65 mm spoon at 11 g,
**0.35 N·m loses it after one lift sub-step of twelve; 0.20 N·m carries it the whole
way.** 0.20 N·m is about 2.7 N a jaw, so 5.4 N of friction at μ=1 against the 1.5 N
of the heaviest prop and 0.4 N of the lightest.

**4. The props had to be redesigned to the gripper, not the other way round.** The
graspable band is **12–88 mm above the table**: below that the fingers hit the
surface before they are beside the object, above that the arm runs out of top-down
reach, whose ceiling is 93 mm. The original props satisfied none of it — a 150 mm
plate and a 76 mm mug both exceeded the 46 mm jaws, and flat cutlery lay where the
moving jaw's arc cannot reach its side.

| prop | became | why |
|---|---|---|
| plate | 34 mm footed dish, raised rim | a full-size plate cannot be grasped by a 46 mm gripper at all |
| mug | 36 mm barrel, 64 mm tall | 76 mm exceeded the jaws |
| fork / spoon | **65 mm**, handle on a riser at 19–29 mm, 12 mm bar | see below |

The flatware length is the sharpest case. At its original 136 mm the grasp bar sat
98 mm from the far end, and lifting the handle only pivoted the piece about that end:
measured at **-39° to -42° with the bowl or all four tines still on the table while
the handle was 42–46 mm up**, which is the entire lift the arm has. Clearing a piece
held near one end takes a lift of order its own length, and the ceiling is 93 mm — so
the piece had to come down to the lift rather than the lift up to the piece. At 65 mm
the grasp bar is 45 mm from the far end. The bar keeps its section exactly (12 mm
across the jaws, 10 mm tall, 19–29 mm up) so the seating in the pads is unchanged and
only the overhang shrinks.

Each body declares exactly one geom named `<prop>_grasp`, which is the feature the
planner aims at. Without it, a fork's merged bounding box is dominated by its head
and the grasp lands on the wrong part at the wrong width.

## The reach and torque envelope

Reach is a **per-arm, orientation-aware occupancy grid** on a 10 mm cell, and a cell
counts as reachable only if the arm can put its gripper there *with the approach axis
pointing down* **and hold that pose statically** — inverse dynamics at zero velocity
and acceleration, with every joint torque inside its actuator's force range. Position
reach alone is far too generous: folding the wrist under to keep the approach
vertical costs several centimetres, and a pose the arm can touch but not hold is not
a pose you can grasp from.

IK is damped least squares over the five pose-controlling joints. Five joints cannot
generally hit a full six-DOF target, so orientation rows are down-weighted (0.6) and
the residual is reported rather than assumed away; callers clamp any waypoint that
does not converge.

The envelope this produces is awkward in specific ways:

- **Top-down ceiling: 93 mm** above the table.
- The two mount keepouts cut a notch out of the near half: for x below about
  **+0.02 m nothing outside |y| < 0.04 is legal at all**, and past **x = +0.12 the
  middle drops out** too — both arms at full stretch straight ahead — leaving only
  **|y| > 0.14**. The place setting's offsets are wider than table etiquette would
  have them, and that is this notch talking.
- The two arms' reach maps overlap in a band **100–120 mm wide across y**, **169–193
  cells** depending on grasp height.
- Lift height depends strongly on direction. A vertical lift clamps at **27 mm** over
  one fork's grasp point; allowing the lift **40 mm of draw-back toward the arm's own
  base** reaches the full 50 mm, because the top-down ceiling rises steeply on the way
  in. So the lift goes straight up only as far as it takes to clear the table, then
  draws back.

**The jaw offset belongs in the reach test, and originally was not.** The map is
indexed by where the *prop* sits, but what must be reachable is where the *tool site*
goes — 11 mm away for the fork, 13 for the spoon, 21 for the plate, **22 for the
mug**. Ignoring it made the goal layout promise slots the arms cannot release at: on
seed 0 it handed the mug a slot whose release pose its receiving arm misses by
**7.5 mm**, and called the prop movable.

The fix is a *translation*, not a safety margin, and which translation is known: a
top-down grasp preserves the prop's yaw from pick through release, so the prop
arrives at its slot lying the way it lies now. A round grasp feature grasps the same
at every yaw, so for the plate and mug any direction will do — which makes their legal
set *larger*, since the arm must reach the grasp site and not the prop's centre.
Eroding by a disk instead, which is what "stay clear of the boundary" suggests, left
**all ten seeds with no legal setting at all**.

## Pick success: 57.5% → 92.5%

Four props × ten seeds, one process per case. A pass needs >30 mm of rise, no table
contact, and the prop still held.

| | plate | fork | spoon | mug | overall |
|---|---|---|---|---|---|
| before | 10/10 | 3/10 | 2/10 | 8/10 | **23/40 = 57.5%** |
| after | 10/10 | 10/10 | 9/10 | 8/10 | **37/40 = 92.5%** |

Three fixes account for it, and each came from a measurement rather than a guess:

1. **Clamp every waypoint into the reach envelope, and make the approach Cartesian.**
   Commanding a waypoint outside the envelope does not merely miss it — the servos
   saturate, the tool lags the command, and the straight path the caller asked for is
   not the path flown. Measured on the spoon, the lift's sub-step spacing collapsed
   from 4.2 mm to 3.0 mm as the arm ran out of reach, and the resulting off-axis
   motion sheared the spoon out of the jaws.
2. **Route the transit over the props and subdivide it.** The transit is the longest
   leg by far — 185 mm against the descent's 50 — and the tool's bow off the commanded
   line is second order in the joint step. At twelve sub-steps it still dipped onto
   the prop it had come for; 36 fixed it. Interpolating joint targets between two
   tool poses bows the tool off the line: demanding the grasp yaw at the first
   sub-step swung the tool **39 mm below the line, through the fork it was about to
   pick**.
3. **Shorten the flatware to 65 mm and ease the squeeze once the jaws load** (above).

The three remaining failures are honest ones: two mug transit knocks and one dropped
spoon.

## The task, and why the handover goes via the table

The place setting is defined relative to the plate as an anchor. The anchor is not
written down — it is searched for over the same reach masks the randomizer spawns
into, scored first by how many props one arm could both pick up and put down and only
then by how far inside their own masks the four targets sit. A goal position the arms
cannot get to is not a goal.

The two arms split the table at y = 0. A prop lying on the far side of its own slot
cannot be carried there by either arm alone, so it must be handed over. **The handover
is mediated by the table: the giving arm sets the prop down where both arms can
reach, and the receiving arm picks it up again.** That is not a shortcut. An in-air
handover was searched for and does not exist in this scene, and the search is kept in
the code as a reported measurement rather than deleted.

Two top-down grippers separate only along their shared lateral axis — along the
approach they are stacked, and along the opening axis the body runs -27 to +68 mm of
the tool site. Across the lateral axis the gripper's collision hull spans **52.0 mm**
(-27.8 to +24.2 mm of the site), so two grippers at the same yaw must stand at least
that far apart to clear each other. Turning one end-for-end buys almost nothing: the
hull is nearly symmetric there, so opposed yaws still need **48.4 mm**.

The longest grasp feature in this scene is the mug's **36 mm**. With *both* grasps
free to sit at opposite ends of it:

| prop | grasp feature | interpenetration |
|---|---|---|
| mug | 36 mm | **14.2 mm** across 36 contacts |
| plate | 34 mm | **16.1 mm** |
| fork | 32 mm | **18.1 mm** |

Searching wider does not find a pose, only a shallower floor. Over both arms' yaws
independently — opposed and perpendicular included — and with the two grasps stacked
up the mug's 64 mm barrel, the least interpenetration is **14.8 mm** for the fork and
spoon, **27.5 mm** for the mug, **30.0 mm** for the plate. The one clear pose that
turns up is not a grasp at all: it sits both jaw lines tangent to the mug's barrel,
where the chord between them is **0.0 mm** and there is nothing to close on.

So there is no pose to implement an in-air handover against, and none is implemented.
`handover()` reports the shallowest clash it found on every call — on seed 0's fork,
18.2 mm over 43 contacts across 193 shared cells.

## Full-task sweep

Four props × ten seeds. A prop passes only if it ends within 20 mm of its slot **and**
resting on the table — a prop still in the jaws is not placed, wherever its xy reads.

| | props placed | complete settings | handovers |
|---|---|---|---|
| without the jaw-offset fix | 25/40 (62.5%) | 0/10 | 4/11 |
| with it | 24/40 (60.0%) | 0/10 | 2/8 |

The jaw-offset fix is close to neutral on the rate. The one prop is seed 5's spoon
crossing the tolerance line (19.6 → 21.8 mm) plus seed 0 reshuffling which props it
completes; the goal layouts differ between the two runs, so the seeds are not paired
and that difference is noise. What it does do is what it was for: honest reach tests
need **fewer** handovers (11 → 8), because a setting one arm can do end-to-end is now
scored as one, and `release unreachable` failures halve.

**Placement accuracy is not what fails.** A prop that gets placed lands a median
**1.6 mm** from its slot, worst **18.8 mm**, against a 20 mm tolerance. The 16
failures are all upstream:

| failure | n | mechanism |
|---|---|---|
| set down short | 5 | the giving arm cannot put the prop down at the transfer spot |
| off target | 4 | flatware shed mid-carry, grip faded for 29–48 of the carry's sub-steps |
| pick failed | 4 | never lifts; one is shoved 8.8 mm before the grasp |
| release unreachable | 2 | release pose outside the envelope — the arm refuses rather than dropping it |
| not received | 1 | set down fine, receiving arm cannot pick it up again |

`complete settings` is 0/10 because it needs all four props and no seed gets all four.
One seed (26) did, in a later run.

### The tolerance budget

The fork originally finished its handover **17.9 mm** off a 20 mm tolerance — error
compounding across the extra pick/place cycle. Four accumulators, cut to **4.1 mm**:

| accumulator | measurement |
|---|---|
| grasp taken off the balance point | fork held **7.23 mm** off its centre of mass (spoon 4.18). The prop rotates in the jaws for the whole carry: hang below the tool grew **43.4 → 59.9 mm** across one transit with grip force never dropping below 2.24 N, then pivoted out entirely |
| stale tool→prop offset | the release was computed from an offset measured *before* the carry. Re-measuring over the target: **7.65 → 2.32 mm** (fork), **4.71 → 0.93 mm** (mug) |
| release height off the body origin | held, the fork hangs 39.8 mm below the tool at its origin but **43.4 mm at its lowest tine**, so "origin 4 mm up" left the tine 0.4 mm off the table; the descent drove it in and popped the fork out |
| `geom_rbound` for the load's hang | a bounding *sphere* understates the lowest point by 4.7 mm (mug) to **9.2 mm** (plate), inflating every transit's required height |

And one intervention that was actively harmful. The mid-carry re-grip fired when grip
force faded — but it re-squeezed at the 0.8 N·m closing sweep that `gripper.py` already
measures as ejecting light flatware, and re-seating gently instead still stops the arm
0.15 s a time, spending exactly the time-under-load that makes a wedged prop creep.
Over the 28 sub-steps where the spoon's grip had faded:

| re-grip strategy | spoon put down |
|---|---|
| re-squeeze at closing torque | **84.3 mm** off |
| re-seat at holding torque | **96.0 mm** off |
| do nothing | **6.0 mm** off |

It is now a counter, not an intervention. What the fade was really reporting was a
grasp taken off the centre of mass, which is fixed at the source.

### Known unresolved

**The flatware is shed on long carries.** The grip fades for most of the carry (29–48
sub-steps) and the piece leaves the jaws. This is the largest single remaining failure
mechanism and it has no fix here: `HOLD_TORQUE` cannot go up (0.35 N·m ejects the 11 g
spoon) and re-gripping makes it worse. It is a genuine limit of a parallel jaw that
closes past parallel on a 12 mm bar, recorded rather than tuned around.

## Closed-loop evaluation

`scripts/eval_policy.py` runs the trained policy in the scene with its twelve
outputs driving `data.ctrl` directly, in place of the scripted primitives. Scored
by the same test as the expert sweep — within 20 mm of the slot and resting on the
table — so the numbers are comparable.

| | props placed | complete settings |
|---|---|---|
| scripted expert | **24/40 (60.0%)** | 0/10 |
| ACT, 44k steps | **3/40 (7.5%)**, of which **2 earned** | 0/10 |

One of the three is not a placement: on seed 3 the spoon's slot happened to land
9.8 mm from where the spoon already lay, and the policy never touched it. Exactly
1 of the 40 prop-slot pairs starts inside tolerance, so that artifact affects one
cell — but it is reported separately rather than counted as a success, and the
script now marks it.

### What it actually does

```
                    plate      fork      spoon      mug
placed (earned)       2/10      0/10       0/10     0/10
median error        58.3 mm  135.4 mm   139.6 mm  144.8 mm
best error          13.1 mm   27.0 mm      9.8 mm  24.4 mm
```

Over the 40 prop-outcomes: the jaws contacted a prop **11** times, lifted one clear
of the table **5** times, moved one more than 10 mm **11** times — and **29** props
were never moved at all. One was knocked off the table.

It is not incoherent. The arms move purposefully (3–16 rad of joint travel per
episode), reach, close, and on five occasions lift a prop clear and carry it
part-way. But **every one of those interactions is with the plate**, bar a single
spoon touch. The fork and the mug were never contacted in any of the ten seeds.

The reason is visible in how the dataset was built, and it is not a bug in the
evaluation:

- **The policy is not task-conditioned.** ACT's inputs here are the two images and
  the twelve joint positions. The dataset's seven task strings were recorded but ACT
  never consumes them, so the policy cannot be told which prop to move.
- **Every episode starts from the parked pose, and the expert always does the plate
  first** (`PLACE_ORDER` begins with it). The plate is also the most common episode,
  47 of 134. So from the opening observation, the demonstrated action is nearly
  always "go for the plate."
- **The VAE has collapsed.** Final loss 0.09 with L1 0.09 means the KL term is
  ~0, so the latent carries nothing and ACT regresses the conditional *mean* of the
  demonstrations rather than sampling a mode. Faced with a multi-modal target — four
  props, two arms, handover or not — the mean is the plate behaviour.
- **The horizon is 100 actions.** `n_action_steps` is 100, four seconds at 25 Hz, so
  the policy commits to a four-second plan per inference. Once the plate has been
  moved or fumbled, the state is off-distribution and nothing recovers.

So the honest summary is that the policy learned the opening move of the
demonstrations and little else. That is the expected outcome of behaviour cloning on
134 success-only episodes with no task conditioning, and it is a data and
conditioning problem well before it is a training-length problem.

## The dataset

| | |
|---|---|
| episodes | **134** |
| frames | **47,092** at 25 Hz (31.4 min) |
| seeds | 49 of 50 contributed (seed 47: all four props failed) |
| with a handover | 20 (direct 114) |
| per prop | plate 47, fork 28, spoon 27, mug 32 |
| episode length | min 275, median 296, max 680 frames |
| placement error | median 1.6 mm, worst 18.8 mm |
| on disk | 228 MB (205 MB as `.tar.gz`) |
| format | LeRobotDataset **v3.0**, verified to load with no conversion |

Observations are the two scene cameras (`overhead`, `front`) at 256×256 plus 12 joint
positions; the action is the 12 actuator commands. Recording is at 25 Hz — every 20th
step of the 500 Hz physics — with state and action sampled *before* the step, so each
pair is the command and the state it was applied to.

**An episode is one prop, not one seed.** It runs from the start of a prop's pick to
the end of its final place, handover included, and is kept only if that prop finished
in tolerance and resting on the table. Per-seed episodes would have recorded nothing:
no seed gets all four props. Failed attempts are discarded, so this is **not** a
dataset of recoveries.

The expert was not modified to record it. Capture is a hook on `mujoco.mj_step` — the
one call every leg of every primitive already steps through — so these are the same
trajectories the sweep measures.

### The action vector mixes units

| channels | meaning | unit |
|---|---|---|
| 0–4, 6–10 | arm joint position targets | rad |
| 5, 11 | gripper torque, negative closes | **N·m** |

This is a direct consequence of finding 3 above: the gripper is force-controlled
because a commanded jaw angle drops the prop. `observation.state` is homogeneous (all
12 are joint positions in radians), so only the action needs care. Nothing in the
LeRobot schema records the distinction, so the dataset card does.

## ACT training setup

Trained to **44,000 steps, final loss 0.09 (L1 0.09)** — the KL term has collapsed to
near zero, which is the usual end state for ACT's VAE on a dataset this size. The
checkpoint is `checkpoints/step_0044000` (gitignored: 207 MB of weights plus 413 MB of
optimizer state). Trained on Kaggle with `notebooks/train_act_kaggle.ipynb`.

| | |
|---|---|
| policy | ACT, lerobot 0.4.4, **51.6 M** trainable parameters |
| backbone | ResNet18 ×2 (one per camera), ImageNet-initialized |
| transformer | d_model 512, 8 heads, 4 encoder + 1 decoder layers, VAE latent 32, KL weight 10 |
| action chunk | 100 = **4.0 s** at 25 Hz, against episodes of 11–27 s |
| target | one 16 GB T4 (Kaggle), batch 16, 3 dataloader workers |
| precision | fp16 with `GradScaler` — ACT's VAE loss needs it |
| optimizer | AdamW, lr 1e-5, weight decay 1e-4, grad clip 10.0 |
| checkpoints | 0.58 GiB each, every 2,000 steps, last 2 kept |
| expected rate | 1.5–4 it/s → the 8.5 h budget buys ~45k–120k steps |

Written for Kaggle's session limits rather than against them: checkpoints are written
to a temp directory and atomically renamed so a killed session cannot leave a
half-written one, the training cell resumes from the newest checkpoint it finds, and
the loop stops itself at a wall-clock budget and checkpoints rather than being cut off
mid-step.

**Normalization keeps the two units apart.** LeRobot normalizes with *per-element*
statistics — `stats['action']['mean']` is a 12-vector and the normalizer holds it as a
tensor of that shape — so each channel is standardized against its own mean and spread
and the units are never pooled. The notebook asserts this rather than assuming it:
once on the stored statistics (12 values per statistic, no channel degenerate enough
for the normalizer's epsilon to take over — smallest action std is 0.41), and once
empirically on a real batch, requiring the radian channels *and* the torque channels
both to come out standardized. A shared scale would be actively harmful: radians span
about ±2.8 while torques span only [-0.8, 1.0], so pooling would flatten the gripper
channel to near-constant and the policy would never learn to open or close.

One caveat the checks do not fix: the torque channels are nearly *discrete* — the
expert commands about three levels plus ramps — so an L1 regression is fitting a
near-categorical target.

## OpenVINO conversion and quantization

Measured on the trained checkpoint, against **real dataset frames**.

| precision | size | mean abs err | arm (rad) | gripper (N·m) | gripper sign flips |
|---|---|---|---|---|---|
| FP32 | 131.3 MiB | 0.00000 | max 0.00000 | max 0.00001 | 0.00% |
| FP16 | 66.1 MiB | **0.14223** | p99 1.13, max 2.35 | max 1.88 | **14.94%** |
| INT8 | **38.0 MiB** (3.46× smaller) | 0.0161 | p99 0.137, max 0.58 | max 1.22 | **0.92%** |

A *sign flip* on a gripper channel turns close into open, which is the metric that
decides whether behaviour survives; a mean hides it entirely.

**Ship FP32. INT8 is defensible with a closed-loop check** — the arm is good to
~0.9° mean and 7.8° at p99, but just under 1% of gripper commands invert, and a
grasp that inverts once mid-carry drops the prop. **FP16 must not be deployed on
this model**: 15% of gripper commands invert.

Three findings behind that table, each of which cost a measurement to establish:

**FP16's error is not a rounding cost.** Rounding the network weights through fp16
*in PyTorch* costs a max of 0.0056, and the folded normalization constants 0.0054 —
but the compressed IR is off by 2.35, three hundred times more. It is not execution
precision (the CPU plugin reports f32 for all three, and forcing it changes
nothing), not overflow (no constant became non-finite or was zeroed), and not a bug
in this converter (compressing a freshly read FP32 IR gives the identical error).
What remains is inside OpenVINO's compression pass, and that was not chased further.

**Parity must be checked on real frames.** The original check used one synthetic
uniform-noise frame, where the FP16 IR looked fine at 1.2e-03. On real frames the
same IR is off by 1.9e+00. Uniform images sit far outside the training distribution
and the network answers them from a flat part of its response, so a badly wrong
precision can look exact there. This nearly shipped.

**Keeping the action head in FP does not help.** The obvious fix when INT8 hurts
accuracy is to exclude the output head. Measured, it buys 1.2184 against 1.2190 —
nothing. The error originates upstream in the backbone and transformer; the head
just passes it through.

Two decisions shape the IR:

**It is self-contained.** Input normalization and action unnormalization are compiled
into the graph, so it takes raw observations — images in [0, 1], joints in radians —
and returns an action in real units. An NPU should not need a Python preprocessing
pipeline beside it, and given the mixed-unit action, the per-channel statistics are the
last thing worth reimplementing by hand at the edge. Folded in, they cannot drift;
`convert.py` prints both groups' scales so the separation is visible (10 arm channels
std 0.41–1.02 rad, 2 gripper channels std 0.60–0.61 N·m).

**Every shape is static.** The PyTorch frontend leaves the batch dimension dynamic and
conversion pins all of it, failing loudly if anything survives. This is not tidiness:
the NPU plugin rejects dynamic shapes outright, NNCF cannot size calibration tensors
against them, and the traced graph was only ever valid at its traced shapes — ACT
iterates over its camera list and position embeddings in Python.

INT8 is calibrated on 300 real frames sampled evenly across all 47,092 recorded
frames (`np.linspace`, not the first N, which would come from one episode of one
seed), fed raw because the graph normalizes internally — pre-normalized calibration
data would set every activation range against the wrong input scale.
`model_type=TRANSFORMER` is passed deliberately: it keeps the quantization-sensitive
parts of attention out of INT8. Divergence is reported on 32 frames held out of the
calibration set.

### Latency

Intel Core i7-5500U (Broadwell, 2015), openvino 2026.3.1, batch 1, 2×256×256 cameras,
chunk 100, 50 timed iterations after 5 warmup.

| device | precision | size MiB | compile s | mean ms | p50 ms | p95 ms | fps | nireq |
|---|---|---|---|---|---|---|---|---|
| CPU | fp32 | 131.3 | 0.76 | 156.38 | 150.47 | **182.69** | 6.93 | 4 |
| CPU | fp16 | 66.1 | 0.74 | 163.67 | 158.32 | **182.51** | 6.70 | 4 |
| CPU | int8 | 38.0 | 1.23 | 172.75 | 166.72 | **199.26** | 6.59 | 4 |
| GPU | — | — | — | — | — | — | — | not present |
| NPU | — | — | — | — | — | — | — | not present |

Latency is reported as mean, p50 **and p95** because the tail is what breaks a control
loop and a mean hides it. Throughput is measured separately with the plugin's own
optimal in-flight request count.

**INT8 is 3.46× smaller and slightly *slower* on this CPU.** That is the correct
result on this part, not a bug: Broadwell has **no VNNI**, so INT8 dot products get
no hardware path and the quantize/dequantize nodes are pure overhead.

All three precisions miss the **40 ms** a 25 Hz control loop allows — the rate the
demonstrations were recorded at — by more than 4×. `benchmark.py` checks that
explicitly.

## Intel hardware mapping

**Core Ultra Series 2/3 was not available for this work.** Intel Cloud Services
rejects personal accounts, and no Core Ultra hardware was otherwise accessible. All
benchmarks above ran on an **Intel Core i7-5500U (Broadwell, 2015)**, which exposes
**CPU only** — no VNNI, no AMX, and no NPU. The integrated HD Graphics 5500 is Gen8,
below OpenVINO's Gen9 floor, so OpenVINO does not expose it as a GPU device either.

```
openvino 2026.3.1
devices exposed by this machine: CPU
  CPU      Intel(R) Core(TM) i7-5500U CPU @ 2.40GHz
  GPU      not present on this machine -- skipped
  NPU      not present on this machine -- skipped
```

**`openvino/benchmark.py` runs unmodified on a Core Ultra Series 2/3 machine and will
populate the GPU and NPU rows.** It reads `Core().available_devices` rather than
assuming any device, benchmarks each one it finds, reports absent ones as absent, and
reports a device that is present but cannot compile the graph with its reason while
continuing. That skip path was tested explicitly here by requesting
`--devices NPU GPU CPU` on this CPU-only box. The script imports only `openvino` and
`numpy`, so it runs on a deployment machine with no lerobot and no torch:

```sh
python openvino/benchmark.py                    # every device present
python openvino/benchmark.py --devices NPU      # just the NPU
```

What to expect there — stated as expectation, because none of it was measured:

| device | expectation |
|---|---|
| **NPU** | the interesting target for a control loop and the one that most needs INT8: built for low-precision inference at low power. Its compile time is worth watching, since the plugin compiles a device blob on first load — which is why `compile s` is a column |
| **GPU** (Arc integrated) | executes in fp16 internally regardless of IR precision, so `act_fp16.xml` is the natural input and FP32 buys nothing but file size |
| **CPU** | has VNNI on those parts, so unlike here INT8 should actually be *faster* than FP32, not merely smaller |

Whether the policy closes a 25 Hz loop on a Core Ultra is therefore an open question
this repo sets up but does not answer. The 181 ms p95 above is a statement about a
2015 mobile CPU, not about the architecture's deployability.

## Reproducing

```sh
# One seed of the whole task
.venv/bin/python scripts/run_task.py --seed 0

# Sweeps (one process per case, loops in the shell)
scripts/sweep.sh                       # pick primitive, 4 props x 10 seeds
.venv/bin/python scripts/sweep_report.py
JOBS=4 scripts/sweep_task.sh           # whole task, 10 seeds
.venv/bin/python scripts/sweep_task_report.py

# Record and pack demonstrations
JOBS=4 scripts/record_demos.sh                     # seeds 0-49
<lerobot-env>/bin/python scripts/pack_lerobot.py

# OpenVINO
<lerobot-env>/bin/python openvino/convert.py  --checkpoint <ckpt> --out openvino/ir
<lerobot-env>/bin/python openvino/quantize.py --ir openvino/ir/act_fp32.xml
python openvino/benchmark.py
```

`convert.py` and `quantize.py` need lerobot; `benchmark.py` needs only the OpenVINO
runtime. Omitting `--checkpoint` builds ACT with random weights, which is enough to
convert, quantize and benchmark. See `notebooks/README.md` for training and
`openvino/README.md` for the inference pipeline in detail.

## Limitations

- **No closed-loop evaluation.** Nothing here runs the trained policy back in the
  scene, so whether it actually sets the table is unknown. Numerical fidelity through
  quantization is measured; task success is not.
- **FP16 is unusable on this model** and the cause inside OpenVINO's compression pass
  was not identified.
- **0/10 complete settings.** The expert places 60% of props; no seed in the sweep
  completes all four. Individual placements are accurate (1.6 mm median), so the gap
  is reliability, not precision.
- **The flatware shedding is unfixed**, and is the dominant failure.
- **GPU and NPU are unmeasured**, for the reason stated above.
- **The demonstrations are successes only.** A policy trained on them has never seen a
  recovery, and the closed-loop evaluation shows it: nothing recovers once the state
  leaves the demonstrated distribution.
- **The policy is not task-conditioned**, so it cannot be directed at a prop. Adding
  the task string as an input is the first thing to change before training again.
