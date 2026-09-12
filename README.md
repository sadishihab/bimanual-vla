# Bimanual table setting: SO-101 in MuJoCo, ACT, and OpenVINO

Two SO-101 arms set a table in MuJoCo. A scripted expert picks four props out of a
randomized layout and places them into a place setting, handing a prop from one arm
to the other when no single arm can both reach it and reach its slot. Successful
manipulations are recorded as a LeRobot v3.0 dataset, an ACT policy trains on it,
the policy is evaluated back in the scene, and the checkpoint is converted to
OpenVINO IR and quantized for Intel inference hardware.

Everything here is driven by measurement. Where a primitive fails, the mechanism and
the number are recorded rather than the failure being tuned away, and most of the
design decisions below exist only because a measurement forced them.

## What was measured, and what was not

| | status |
|---|---|
| Gripper geometry, jaw pads, grasp envelope | **measured** in MuJoCo |
| Reach and static-torque envelope | **measured** |
| Pick success, 4 props × 10 seeds | **measured**: 23/40 → 37/40 |
| In-air handover feasibility | **measured** — and ruled out |
| Full-task success, 4 props × 10 seeds | **measured**: 24/40 props, 0/10 complete settings |
| Demonstration dataset | **built**: 134 episodes, 47,092 frames |
| ACT training (unconditioned) | **run**: 44,000 steps, final loss 0.09 (L1 0.09) |
| Closed-loop policy success | **measured**: 3/40 props, of which 2 earned |
| Language conditioning | **trained and evaluated**: 3/40, unchanged, and it still only touches the plate |
| OpenVINO FP32 / FP16 / INT8 | **measured** on the trained checkpoint, against real frames |
| OpenVINO CPU latency | **measured** on a Broadwell i7-5500U |
| OpenVINO GPU / NPU latency | **not measured** — no such device available, see [Intel hardware](#intel-hardware-mapping) |

One thing is claimed nowhere in this document: that anything runs well on an Intel
NPU. That has not been measured, and is marked as expectation wherever it appears.
The language-conditioned variant *has* now been trained and evaluated, and it did
not work — see below.

## Architecture

```
scenes/bimanual_table.xml     dual SO-101, table, overhead + front cameras
envs/scene.py                 loads the scene and converts the gripper to force control
envs/randomize.py             seeded domain randomization; the reach map
envs/task.py                  the place-setting goal, and which props cross the midline
control/ik.py                 damped-least-squares IK for one arm
control/gripper.py            gripper force control and the measured jaw pads
control/primitives.py         pick, place, handover, and the planners behind them
control/language.py           task-string conditioning as an extra ACT encoder token
scripts/                      per-seed runners, sweeps, the recorder, the evaluator
notebooks/train_act_kaggle.ipynb   ACT training on Kaggle
openvino/                     IR conversion, INT8 quantization, device benchmark
```

```
MuJoCo scene ─► scripted expert ─► record_demos.py ─► pack_lerobot.py ─► LeRobotDataset v3.0
                      │                                                        │
                      │                             ACT (lerobot 0.4.4) ◄──────┘
                      │                                    │
                      └──────────► eval_policy.py ◄────────┤  closed loop, same scene
                                                           │
                        convert.py ─► FP32 / FP16 IR ─► quantize.py ─► INT8 IR
                                                           │
                                                    benchmark.py ─► CPU / GPU / NPU
```

Two rules run through the repo, both from experience: **one seed per process** —
sweep loops live in shell scripts so each case is a fresh interpreter with a fresh
model — and **no renderer unless the output genuinely needs pixels**, since memory
has been the recurring failure. Only the recorder and the closed-loop evaluator
build one, and both free it.

## The SO-101 gripper, and what it forced

Almost every prop dimension and every primitive detail traces back to four findings
about the actual gripper.

### 1. The collision hull is not the finger

MuJoCo collides a mesh geom as its convex hull, and both SO-101 jaws are concave
parts. On the fixed finger the hull bridges from the wrist mount to the fingertip as
a single **1428 mm² facet tilted 21.7°** off the opening axis. Of its 606 facets
facing the aperture, **not one is within 5° of parallel**.

Grasping against that ramp touches only the object's top edge and drives it
downward — measured at **2.07 N onto the table against a 0.30 N fork** — and it does
so at *any* grasp height, because the ramp is fixed to the tool frame and moves with
it. No choice of grasp height fixes a wedge.

The fix was not to invent geometry. The real inner faces under the hull are flat and
parallel: fitting planes through the mesh vertices gives **0.23° and 0.24°** off the
opening axis with a **0.33 mm residual**, 11.5 mm wide, ~10 mm deep, **15.80 mm
apart**. Each jaw gets an explicit pad laid on the face that is already there,
placed from that measurement rather than typed in.

### 2. The tool site is not the middle of the aperture

It sits on the *fixed* finger, so the aperture spans `[site, site + opening]` along
the site's +z axis. Centring the site on an object drops the fixed finger straight
onto it. Grasps are offset back along the opening axis by half the object's width
plus 4 mm of clearance, so the jaws straddle the object. That offset is 10 mm for
the flatware and **22 mm for the mug** — two cells of the reach map's 10 mm grid —
and it returns as a bug three more times.

### 3. Position control cannot hold a grasp

A commanded jaw angle relaxes its grip as the servo settles and the prop falls out
during the lift. The gripper is driven in **torque**: `OPEN = +1.0 N·m` runs the jaw
to its stop, `GRIP = -0.8 N·m` is the closing sweep, `HOLD = -0.20 N·m` is what
stays on.

Three numbers because three different jobs, and getting it wrong destroys the grasp
in both directions. The closing torque must sweep the jaw in from its stop against
0.6 N·m·s/rad of damping; left on, it crushes through whatever is soft enough. How
soft depends on the object, because MuJoCo's time-constant `solref` makes contact
stiffness scale with effective mass at the contact: the 136 mm spoon stopped **2.5 mm
into a 12 mm bar**, while the 65 mm one, with a fifth of the rotational inertia,
went **straight through to the hard stop and ejected the piece**.

Too little holding torque drops the prop; too much ejects it, because the jaws close
past parallel on anything narrower than their 15.80 mm reference gap and the wedge
flicks a light piece out. Measured on the 65 mm spoon at 11 g: **0.35 N·m loses it
after one lift sub-step of twelve; 0.20 N·m carries it the whole way.** 0.20 N·m is
about 2.7 N a jaw, so 5.4 N of friction at μ=1 against the 1.5 N of the heaviest
prop and 0.4 N of the lightest.

### 4. The props had to be redesigned to the gripper

The graspable band is **12–88 mm above the table**: below that the fingers hit the
surface before they are beside the object, above that the arm runs out of top-down
reach, whose ceiling is 93 mm. The original props satisfied none of it — a 150 mm
plate and a 76 mm mug both exceeded the 46 mm jaws, and flat cutlery lay where the
moving jaw's arc cannot reach its side.

| prop | became | why |
|---|---|---|
| plate | 34 mm footed dish, raised rim | a full-size plate cannot be grasped by a 46 mm gripper at all |
| mug | 36 mm barrel, 64 mm tall | 76 mm exceeded the jaws |
| fork / spoon | **65 mm**, handle on a riser at 19–29 mm, 12 mm bar | see below |

The flatware is the sharpest case. At its original 136 mm the grasp bar sat 98 mm
from the far end, and lifting the handle only pivoted the piece about that end:
measured at **−39° to −42° with the bowl or all four tines still on the table while
the handle was 42–46 mm up**, which is the entire lift the arm has. Clearing a piece
held near one end takes a lift of order its own length, and the ceiling is 93 mm — so
the piece came down to the lift rather than the lift up to the piece. At 65 mm the
grasp bar is 45 mm from the far end, and the bar keeps its section exactly (12 mm
across the jaws, 10 mm tall, 19–29 mm up) so the seating in the pads is unchanged.

Each body declares one geom named `<prop>_grasp`, the feature the planner aims at.
Without it a fork's merged bounding box is dominated by its head and the grasp lands
on the wrong part at the wrong width.

## The reach and torque envelope

Reach is a **per-arm, orientation-aware occupancy grid** on a 10 mm cell. A cell
counts as reachable only if the arm can put its gripper there *with the approach
axis pointing down* **and hold that pose statically** — inverse dynamics at zero
velocity and acceleration, every joint torque inside its actuator's force range.
Position reach alone is far too generous: folding the wrist under to keep the
approach vertical costs several centimetres, and a pose the arm can touch but not
hold is not a pose you can grasp from.

IK is damped least squares over the five pose-controlling joints. Five joints cannot
generally hit a six-DOF target, so orientation rows are down-weighted and the
residual is reported rather than assumed away; callers clamp any waypoint that does
not converge.

The envelope is awkward in specific, measured ways:

- **Top-down ceiling: 93 mm** above the table.
- The mount keepouts cut a notch out of the near half: for x below about **+0.02 m
  nothing outside |y| < 0.04 is legal**, and past **x = +0.12 the middle drops out**
  — both arms at full stretch straight ahead — leaving only **|y| > 0.14**. The place
  setting's offsets are wider than etiquette would have them, and that is this notch
  talking.
- The two arms' maps overlap in a band **100–120 mm wide across y**, **169–193
  cells** depending on grasp height.
- Lift height depends strongly on direction: a vertical lift clamps at **27 mm** over
  one fork's grasp point, and allowing **40 mm of draw-back toward the arm's base**
  reaches the full 50 mm, because the ceiling rises steeply on the way in. So the
  lift goes straight up only far enough to clear the table, then draws back.

**The jaw offset belongs in the reach test, and originally was not.** The map is
indexed by where the *prop* sits, but what must be reachable is where the *tool site*
goes — 11 mm away for the fork, 13 for the spoon, 21 for the plate, **22 for the
mug**. Ignoring it made the goal layout promise slots the arms cannot release at: on
seed 0 it handed the mug a slot whose release pose its receiving arm misses by
**7.5 mm**, then called the prop movable.

The fix is a *translation*, not a margin, and which translation is known: a top-down
grasp preserves the prop's yaw from pick to release. A round grasp feature grasps the
same at every yaw, so for the plate and mug any direction will do — which makes their
legal set *larger*, since the arm must reach the grasp site, not the prop's centre.
Eroding by a disk instead, which is what "stay clear of the boundary" suggests, left
**all ten seeds with no legal setting at all**.

## Pick success: 57.5% → 92.5%

Four props × ten seeds, one process per case. A pass needs >30 mm of rise, no table
contact, and the prop still held.

| | plate | fork | spoon | mug | overall |
|---|---|---|---|---|---|
| before | 10/10 | 3/10 | 2/10 | 8/10 | **23/40 = 57.5%** |
| after | 10/10 | 10/10 | 9/10 | 8/10 | **37/40 = 92.5%** |

Three fixes, each from a measurement:

1. **Clamp every waypoint into the reach envelope; make the approach Cartesian.**
   Commanding a waypoint outside the envelope does not merely miss it — the servos
   saturate and the straight path asked for is not the path flown. Measured on the
   spoon, the lift's sub-step spacing collapsed from 4.2 mm to 3.0 mm as the arm ran
   out of reach, and the off-axis motion sheared the spoon out of the jaws.
2. **Route the transit over the props and subdivide it.** The transit is 185 mm
   against the descent's 50, and the tool's bow off the commanded line is second
   order in the joint step: at twelve sub-steps it still dipped onto the prop it had
   come for, and 36 fixed it. Demanding the grasp yaw at the first sub-step swung the
   tool **39 mm below the line, through the fork it was about to pick**.
3. **Shorten the flatware to 65 mm and ease the squeeze once the jaws load.**

The three remaining failures are honest: two mug transit knocks and one dropped spoon.

## The task, and why the handover goes via the table

The setting is defined relative to the plate as anchor. The anchor is not written
down — it is searched over the same reach masks the randomizer spawns into, scored
first by how many props one arm could both pick up and put down, then by how far
inside their own masks the four targets sit. A goal the arms cannot reach is not a
goal.

The arms split the table at y = 0, so a prop lying on the far side of its own slot
must be handed over. **The handover is mediated by the table**: the giving arm sets
the prop down where both arms can reach, and the receiving arm picks it up. That is
not a shortcut. An in-air handover was searched for and does not exist in this scene,
and the search is kept in the code as a reported measurement rather than deleted.

Two top-down grippers separate only along their shared lateral axis — along the
approach they are stacked, and along the opening axis the body runs −27 to +68 mm of
the tool site. Across the lateral axis the gripper's collision hull spans **52.0 mm**
(−27.8 to +24.2 mm of the site), so two grippers at the same yaw must stand at least
that far apart to clear each other. Turning one end-for-end buys almost nothing: the
hull is nearly symmetric, so opposed yaws still need **48.4 mm**.

The longest grasp feature in the scene is the mug's **36 mm**. With *both* grasps
free to sit at opposite ends of it:

| prop | grasp feature | interpenetration |
|---|---|---|
| mug | 36 mm | **14.2 mm** across 36 contacts |
| plate | 34 mm | **16.1 mm** |
| fork | 32 mm | **18.1 mm** |

**52 mm of hull against 36 mm of grasp feature**: the arms cannot be far enough apart
to clear while both holding the same prop. Searching wider finds no pose, only a
shallower floor — over both arms' yaws independently, and with the grasps stacked up
the mug's 64 mm barrel, the least interpenetration is **14.8 mm** (fork and spoon),
**27.5 mm** (mug), **30.0 mm** (plate). The one clear pose that turns up is not a
grasp: it sits both jaw lines tangent to the mug's barrel, where the chord between
them is **0.0 mm** and there is nothing to close on.

So there is no pose to implement an in-air handover against, and none is implemented.
`handover()` reports the shallowest clash it found on every call — on seed 0's fork,
18.2 mm over 43 contacts across 193 shared cells.

## Full-task sweep: 24/40

Four props × ten seeds. A prop passes only if it ends within 20 mm of its slot **and**
resting on the table — a prop still in the jaws is not placed, wherever its xy reads.

| | props placed | complete settings | handovers |
|---|---|---|---|
| without the jaw-offset fix | 25/40 (62.5%) | 0/10 | 4/11 |
| with it | **24/40 (60.0%)** | 0/10 | 2/8 |

The jaw-offset fix is near-neutral on the rate. The one prop is seed 5's spoon
crossing the line (19.6 → 21.8 mm) plus seed 0 reshuffling which props it completes;
the goal layouts differ between runs, so the seeds are not paired and that difference
is noise. What it does do is what it was for: honest reach tests need **fewer**
handovers (11 → 8), because a setting one arm can do end-to-end is now scored as one,
and `release unreachable` failures halve.

**Placement accuracy is not what fails.** A placed prop lands a median **1.6 mm** from
its slot (plate 1.7, fork 1.6, spoon 1.9, mug 0.5), worst 5.8 mm, against a 20 mm
tolerance. The 16 failures are all upstream:

| failure | n | mechanism |
|---|---|---|
| set down short | 5 | the giving arm cannot put the prop down at the transfer spot |
| off target | 4 | flatware shed mid-carry, grip faded for 29–48 of the carry's sub-steps |
| pick failed | 4 | never lifts; one is shoved 8.8 mm before the grasp |
| release unreachable | 2 | release pose outside the envelope — the arm refuses rather than dropping it |
| not received | 1 | set down fine, receiving arm cannot pick it up again |

`complete settings` is 0/10 because it needs all four props and no seed gets all four
(one seed later did, in the 50-seed recording run).

### The tolerance budget

The fork originally finished its handover **17.9 mm** off a 20 mm tolerance — error
compounding across the extra pick/place cycle. Four accumulators, cut to **4.1 mm**:

| accumulator | measurement |
|---|---|
| grasp taken off the balance point | fork held **7.23 mm** off its centre of mass (spoon 4.18). The prop rotates in the jaws for the whole carry: hang below the tool grew **43.4 → 59.9 mm** across one transit with grip force never below 2.24 N, then pivoted out |
| stale tool→prop offset | the release was computed from an offset measured *before* the carry. Re-measuring over the target: **7.65 → 2.32 mm** (fork), **4.71 → 0.93 mm** (mug) |
| release height off the body origin | held, the fork hangs 39.8 mm below the tool at its origin but **43.4 mm at its lowest tine**, so "origin 4 mm up" left the tine 0.4 mm off the table; the descent drove it in and popped the fork out |
| `geom_rbound` for the load's hang | a bounding *sphere* understates the lowest point by 4.7 mm (mug) to **9.2 mm** (plate), inflating every transit's required height |

And one intervention that was actively harmful. The mid-carry re-grip fired when grip
force faded — but it re-squeezed at the 0.8 N·m sweep that `gripper.py` already
measures as ejecting light flatware, and re-seating gently still stops the arm 0.15 s
a time, spending the time-under-load that makes a wedged prop creep. Over the 28
sub-steps where the spoon's grip had faded:

| re-grip strategy | spoon put down |
|---|---|
| re-squeeze at closing torque | **84.3 mm** off |
| re-seat at holding torque | **96.0 mm** off |
| do nothing | **6.0 mm** off |

It is now a counter, not an intervention.

### Known unresolved

**The flatware is shed on long carries.** The grip fades for most of the carry (29–48
sub-steps) and the piece leaves the jaws. It is the largest remaining failure and it
has no fix here: `HOLD_TORQUE` cannot go up (0.35 N·m ejects the 11 g spoon) and
re-gripping makes it worse. A genuine limit of a parallel jaw that closes past
parallel on a 12 mm bar, recorded rather than tuned around.

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
in tolerance and resting on the table. Per-seed episodes would have recorded nothing.
Failed attempts are discarded, so this is **not** a dataset of recoveries — a fact the
closed-loop result below cashes out.

The expert was not modified to record it: capture is a hook on `mujoco.mj_step`, the
one call every leg of every primitive already steps through.

### The action vector mixes units

| channels | meaning | unit |
|---|---|---|
| 0–4, 6–10 | arm joint position targets | rad |
| 5, 11 | gripper torque, negative closes | **N·m** |

A direct consequence of finding 3: the gripper is force-controlled because a
commanded jaw angle drops the prop. `observation.state` is homogeneous (all 12 are
joint positions in radians), so only the action needs care. LeRobot normalizes with
per-element statistics, so each channel is standardized on its own scale and the units
are never pooled; the training notebook asserts that rather than assuming it, once on
the stored statistics and once empirically on a real batch.

## ACT training

| | |
|---|---|
| policy | ACT, lerobot 0.4.4, **51.6 M** trainable parameters |
| backbone | ResNet18 ×2 (one per camera), ImageNet-initialized |
| transformer | d_model 512, 8 heads, 4 encoder + 1 decoder layers, VAE latent 32, KL weight 10 |
| action chunk | 100 = **4.0 s** at 25 Hz |
| hardware | one 16 GB T4 (Kaggle), batch 16, 3 workers, fp16 with `GradScaler` |
| optimizer | AdamW, lr 1e-5, weight decay 1e-4, grad clip 10.0 |
| result | **44,000 steps, final loss 0.09 (L1 0.09)** |

Loss 0.09 with L1 0.09 means the KL term has collapsed to ~0. That matters below.

The notebook is written for Kaggle's session limits rather than against them:
checkpoints written to a temp directory and atomically renamed so a killed session
cannot leave a half-written one, resume from the newest checkpoint, and a wall-clock
budget that stops and checkpoints rather than being cut off mid-step.

## Closed-loop evaluation: 3/40

`scripts/eval_policy.py` runs the policy in the scene with its twelve outputs driving
`data.ctrl` directly, in place of the scripted primitives. Scored by the same test as
the expert sweep.

| | props placed | complete settings |
|---|---|---|
| scripted expert | **24/40 (60.0%)** | 0/10 |
| ACT, 44k steps | **3/40 (7.5%)**, of which **2 earned** | 0/10 |

One of the three is not a placement: on seed 3 the spoon's slot landed 9.8 mm from
where the spoon already lay and the policy never touched it. Exactly 1 of the 40
prop-slot pairs starts inside tolerance, so the artifact is one cell — but it is
reported separately rather than counted, because an untouched pass is not evidence a
policy can place. The two genuine ones are the plate on seeds 4 and 6: moved 65 and
84 mm, lifted clear, jaws loaded for 271 and 144 control steps.

```
                    plate      fork     spoon       mug
placed (earned)      2/10      0/10      0/10      0/10
median error       58.3 mm  135.4 mm  139.6 mm  144.8 mm
best error         13.1 mm   27.0 mm     9.8 mm   24.4 mm
```

### What it actually does

Not incoherent. Over 40 prop-outcomes the jaws contacted a prop **11** times, lifted
one clear of the table **5** times, moved one more than 10 mm **11** times — and **29
props were never moved at all**. One was knocked off the table. The arms move
purposefully, 3–16 rad of joint travel per episode.

But **every one of those interactions is with the plate**, bar a single spoon touch.
The fork and the mug were never contacted in any of the ten seeds.

### Diagnosis

Three causes, all visible in how the data was built:

- **No task conditioning.** ACT's inputs here are the two images and the twelve joint
  positions. The dataset's seven task strings were recorded but ACT never consumes
  them, so the policy cannot be told which prop to move.
- **VAE collapse.** Final loss 0.09 with L1 0.09 means KL ≈ 0, so the latent carries
  nothing and ACT regresses the conditional *mean* of the demonstrations rather than
  sampling a mode. Faced with a multi-modal target — four props, two arms, handover or
  not — the mean is one behaviour.
- **Plate class imbalance.** Every episode starts from the parked pose, the expert
  always takes the plate first, and the plate is **47 of 134** episodes. So from the
  opening observation the demonstrated action is nearly always "go for the plate", and
  that is the mean the policy regresses to.

Add the 100-action (4 s) open-loop horizon and nothing recovers once the plate has
been moved or fumbled. The policy learned the opening move and little else — a data
and conditioning problem well before a training-length one.

## The language-conditioned variant

**Status: trained to 44,000 steps on the same schedule, evaluated, and it did not
change the behaviour.**

lerobot 0.4.4's ACT has **no language path** — the only matches for "language" in
`modeling_act.py` and `configuration_act.py` are the Apache licence header, and
nothing in `lerobot/policies/` consumes `FeatureType.LANGUAGE` except `sarm`.

It did not need a fork. Alongside the latent and the robot-state token, ACT already
accepts one more 1-D observation vector — `observation.environment_state` — gives it
its own `nn.Linear` into `dim_model` and its own learned positional embedding, and
stacks it with the image tokens. A sentence embedding is exactly that shape, so the
language token goes in through that slot and **lerobot is untouched**. Verified:
declaring the feature grows the 1-D positional embedding from two rows to three,
builds `Linear(384, 512)`, and the projection receives gradient.

- Encoder: **MiniLM-L6, 22.7 M parameters, frozen**, mean-pooled and L2-normalized to
  384 dims. Only the **384→512 projection is learned: 197,120 parameters** of 51.8 M.
- Seven task strings, embedded once and looked up — training speed unchanged, and the
  **dataset did not need repacking**, because the embedding is injected into the batch
  rather than stored as a column. `ENV → IDENTITY` normalization is what makes that
  possible: no statistics are needed for a key the dataset has never seen.
- Measured structure: each prop's direct and handover phrasings are mutual nearest
  neighbours (fork/hand-fork **0.872**, spoon/hand-spoon **0.904**, mug/hand-mug
  **0.869**), so prop identity dominates the embedding.
- The result is an ordinary ACT checkpoint that `ACTPolicy.from_pretrained` loads,
  which a forked `ACT.forward` would have broken — and with it the OpenVINO pipeline.
- The conversion path takes it: a conditioned IR has a **fourth input**,
  `observation.environment_state (1, 384)`, appended after the cameras so the
  unconditioned graph is unchanged. Verified with random weights; `convert.py` detects
  the token from a conditioned checkpoint's own config.

Two caveats were stated before training, and the second one is what happened. The
token conditions the encoder feeding the decoder, **not the VAE posterior**. And
conditioning addresses only the first of the three diagnosed causes — so if the
latent stays collapsed, it need not help.

### Result: it did not help

Because the policy is task-conditioned, the evaluation can be **directed**: one
episode per prop, each from a fresh reset of the same layout, each given that prop's
own task string. Identical observation, different sentence — so any change is the
conditioning and nothing else.

| | props placed | target contacted | lifted |
|---|---|---|---|
| unconditioned, undirected | 3/40 (2 earned) | plate 10, fork 0, spoon 1, mug 0 = **11/40** | 5/40 |
| conditioned, directed | 3/40 (2 earned) | plate 10, fork 0, spoon 0, mug 0 = **10/40** | 5/40 |

**The same 3/40, and it still only touches the plate.** In all 30 episodes where the
instruction named the fork, the spoon or the mug, the policy went for the **plate**
instead — 30 non-target plate contacts. The fork and the mug were never contacted in
any of the 40 directed episodes, exactly as before. The one-contact difference is the
unconditioned policy's accidental spoon touch, which is noise.

The language path is genuinely wired — the projection takes gradient, and the same
observation with two different task strings produces two different action chunks.
The trained policy simply ignores it.

### A hypothesis, weakly supported

In the demonstrations, image and task are almost perfectly correlated: the expert
always works in `PLACE_ORDER`, so "the plate has already been moved" *implies* "the
fork is next". A policy can read the table state from the image and never need the
sentence. The directed evaluation breaks that correlation by always starting
pristine, and the image then says "plate not yet placed", which is what the policy
acts on.

Tested directly — give the fork instruction from a state where the plate episode has
already run: on **1 of 4 seeds** the fork was then contacted (7 control steps, moved
15.5 mm) against **0 of 4** from pristine, and that one seed is the one where the
plate episode actually succeeded. Consistent with the shortcut, but one hit out of
four is not evidence enough to call it established.

What is established is narrower and still useful: conditioning on its own does not
fix this, and the next thing to change is the data rather than the architecture —
episodes that start from varied table states, so the image cannot stand in for the
instruction.

`transformers` is pinned to `>=4.57.1,<5`, which lerobot 0.4.4 declares. Not
cosmetic: installing 5.17 here broke `import lerobot.policies` outright with a
dataclass error in its GR00T module.

## OpenVINO conversion and quantization

Measured on the trained checkpoint, against **real dataset frames**.

| precision | size | mean abs err | arm (rad) | gripper (N·m) | gripper sign flips |
|---|---|---|---|---|---|
| FP32 | 131.3 MiB | 0.00000 | max 0.00000 | max 0.00001 | **0.00%** |
| FP16 | 66.1 MiB | **0.14223** | p99 1.13, max 2.35 | max 1.88 | **14.94%** |
| INT8 | **38.0 MiB** (3.46× smaller) | 0.0161 | p99 0.137, max 0.58 | max 1.22 | **0.92%** |

A *sign flip* on a gripper channel turns close into open. It is the metric that
decides whether behaviour survives, and a mean hides it entirely.

**Ship FP32. INT8 is defensible with a closed-loop check** — the arm is good to ~0.9°
mean and 7.8° at p99, but just under 1% of gripper commands invert, and a grasp that
inverts once mid-carry drops the prop. **FP16 must not be deployed on this model**:
15% of gripper commands invert.

### The original parity check was wrong

This nearly shipped. The first parity check ran on one **synthetic uniform-noise
frame**, where the FP16 IR looked fine at **1.2e-03**. On real dataset frames the same
IR is off by **1.9e+00** — three orders of magnitude worse. Uniform images sit far
outside the training distribution and the network answers them from a flat part of its
response, so a badly wrong precision can look exact there. `convert.py` now checks
against real frames, reports divergence split by unit group, and warns loudly.

### FP16's error is not a rounding cost

Rounding the network weights through fp16 *in PyTorch* costs a max of **0.0056**, and
the folded normalization constants **0.0054** — but the compressed IR is off by
**2.35**, three hundred times more. Ruled out: execution precision (the CPU plugin
reports f32 for all three IRs and forcing it changes nothing), overflow and underflow
(no constant became non-finite or was zeroed), and a bug in the converter (compressing
a freshly read FP32 IR gives the identical error, and re-saving FP32 from the same
handle is still exact). What remains is inside OpenVINO's compression pass, and that
was not chased further.

Excluding the action head from quantization — the obvious first fix when INT8 hurts —
was tried and buys **1.2184 against 1.2190**: nothing. The error originates upstream in
the backbone and transformer; the head passes it through.

INT8 is calibrated on 300 real frames sampled evenly across all 47,092 (`np.linspace`,
not the first N, which would come from one episode of one seed), fed raw because the
graph normalizes internally. `model_type=TRANSFORMER` keeps the quantization-sensitive
parts of attention out of INT8. Divergence is reported on 32 held-out frames.

The IR is **self-contained** — normalization in, unnormalization out — so it takes raw
observations and returns actions in real units, and the per-channel statistics that
keep radians and newton-metres apart cannot drift at the edge. **Every shape is
static**: the PyTorch frontend leaves the batch dimension dynamic and conversion pins
it, because the NPU plugin rejects dynamic shapes, NNCF cannot size calibration
tensors against them, and the traced graph was only ever valid at its traced shapes.

### Latency

Intel Core i7-5500U (Broadwell, 2015), openvino 2026.3.1, batch 1, 2×256×256 cameras,
chunk 100, 50 timed iterations after 5 warmup.

| device | precision | size MiB | compile s | mean ms | p50 ms | p95 ms | fps | nireq |
|---|---|---|---|---|---|---|---|---|
| CPU | fp32 | 131.3 | 0.76 | 156.38 | 150.47 | **182.69** | 6.93 | 4 |
| CPU | fp16 | 66.1 | 0.74 | 163.67 | 158.32 | **182.51** | 6.70 | 4 |
| CPU | int8 | 38.0 | 1.23 | 172.75 | 166.72 | **199.26** | 6.59 | 4 |
| GPU | — | — | — | — | — | — | — | **not present** |
| NPU | — | — | — | — | — | — | — | **not present** |

Latency is reported as mean, p50 **and p95** because the tail is what breaks a control
loop. Throughput is measured separately with the plugin's own optimal in-flight
request count.

**INT8 is 3.46× smaller and slightly *slower* here.** That is the correct result on
this part, not a bug: Broadwell has **no VNNI**, so INT8 dot products get no hardware
path and the quantize/dequantize nodes are pure overhead.

**All three miss the control budget.** The demonstrations were recorded at 25 Hz, which
allows **40 ms**; the best p95 is **182.51 ms**, over 4× too slow. `benchmark.py`
checks that explicitly. A closed loop on this machine is not possible at the recorded
rate — which is a statement about a 2015 mobile CPU, not about the architecture.

## Intel hardware mapping

**Core Ultra Series 2/3 was not available for this work. Intel Cloud Services rejects
personal accounts**, and no Core Ultra hardware was otherwise accessible.

All benchmarks ran on an **Intel Core i7-5500U (Broadwell, 2015)**, which exposes
**CPU only** — **no VNNI, no AMX, and no NPU**. Its integrated HD Graphics 5500 is
Gen8, below OpenVINO's Gen9 floor, so OpenVINO does not expose it as a GPU device
either.

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
continuing. That skip path was tested here by requesting `--devices NPU GPU CPU` on a
CPU-only box. It imports only `openvino` and `numpy`, so it runs on a deployment
machine with no lerobot and no torch:

```sh
python openvino/benchmark.py                    # every device present
python openvino/benchmark.py --devices NPU      # just the NPU
```

What to expect there — **expectation, not measurement**:

| device | expectation |
|---|---|
| **NPU** | the interesting target for a control loop and the one that most needs INT8: built for low-precision inference at low power. Compile time is worth watching, since the plugin compiles a device blob on first load — which is why `compile s` is a column |
| **GPU** (Arc integrated) | executes in fp16 internally regardless of IR precision, so `act_fp16.xml` would be the natural input — except that FP16 is unusable on this model, so INT8 or FP32 it is |
| **CPU** | has VNNI on those parts, so unlike here INT8 should be *faster* than FP32, not merely smaller |

Whether this policy closes a 25 Hz loop on a Core Ultra is an open question this repo
sets up but does not answer.

## Reproducing

```sh
# One seed of the whole task
.venv/bin/python scripts/run_task.py --seed 0

# Sweeps (one process per case, loops in the shell)
scripts/sweep.sh && .venv/bin/python scripts/sweep_report.py            # pick primitive
JOBS=4 scripts/sweep_task.sh && .venv/bin/python scripts/sweep_task_report.py

# Record and pack demonstrations (seeds 0-49)
JOBS=4 scripts/record_demos.sh
<lerobot-env>/bin/python scripts/pack_lerobot.py

# Closed-loop evaluation of a trained checkpoint
scripts/eval_sweep.sh

# OpenVINO
<lerobot-env>/bin/python openvino/convert.py  --checkpoint checkpoints/step_0044000
<lerobot-env>/bin/python openvino/quantize.py --ir openvino/ir/act_fp32.xml
python openvino/benchmark.py
```

The project venv holds mujoco and numpy only. `convert.py`, `quantize.py`,
`pack_lerobot.py` and `eval_policy.py` need a separate environment with lerobot 0.4.4
(plus mujoco for the evaluator); `benchmark.py` needs only the OpenVINO runtime.
Omitting `--checkpoint` builds ACT with random weights, which is enough to convert,
quantize and benchmark. See `notebooks/README.md` for training and
`openvino/README.md` for the inference pipeline in detail.

## Limitations

- **The evaluated policy places 3 of 40 props**, against the expert's 24. The
  diagnosis above is supported by measurement; the fix is not yet demonstrated.
- **Language conditioning did not work.** Trained and evaluated: the same 3/40, and
  the policy still goes for the plate whatever it is told. The likely cause — image
  and task being correlated in the demonstrations, so the sentence is redundant — is
  supported by only 1 of 4 probe seeds and is not established.
- **The demonstrations are successes only**, so the policy has never seen a recovery —
  and the closed-loop run shows exactly that: nothing recovers once the state leaves
  the demonstrated distribution.
- **The flatware shedding is unfixed** and is the expert's dominant failure.
- **FP16 is unusable on this model** and the cause inside OpenVINO's compression pass
  was not identified.
- **GPU and NPU are unmeasured**, for the reason stated above.
- **The conditioned policy is unmeasured end to end.** Its IR converts and quantizes
  (verified with random weights), but no trained conditioned checkpoint exists yet, so
  its accuracy through FP16/INT8 is unknown — and FP16's failure on the unconditioned
  model is reason to check rather than assume.
