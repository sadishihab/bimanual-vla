"""Scripted manipulation primitives for the bimanual table scene.

The only primitive so far is :func:`pick`: a top-down grasp built from four
waypoints -- above the object, descend, close, lift -- with the arm driven by
its position actuators and the simulation stepped throughout.  Joint targets for
each waypoint come from :mod:`control.ik`.

The gripper is not on position control: it is commanded in torque, so the squeeze
survives the lift instead of decaying as a servo settles.  :mod:`control.gripper`
explains why, and :func:`envs.scene.load_scene` is what applies it -- a model
loaded straight from the XML still has a position-controlled gripper and
``pick`` will read its ctrl as radians.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import mujoco
import numpy as np

import itertools

from control.gripper import (GRIP_TORQUE, GRIP_TRIGGER, HOLD_TORQUE, OPEN_TORQUE)
from control.ik import (ARM_JOINTS, ROT_TOL, IKSolver, reachable_above,
                        reachable_toward, top_down_mat)

# Gripper commands are torques (see control.gripper): OPEN_TORQUE holds the jaw
# back against its upper stop, GRIP_TORQUE is the squeeze.  There is no commanded
# jaw *angle* any more -- the jaw stops where the stop or the object puts it,
# which is the whole point.
WIDE_OPEN = 0.10         # sentinel gap for "jaw is clear of the grasp region"

# Standoff and lift stay inside the arm's top-down envelope, whose ceiling is
# 93 mm above the table.  The old 0.10 m put both waypoints above it, so neither
# converged and the arm entered the descent from a pose centimetres off target.
# Both are *requested* heights.  Each is clamped down to whatever the arm can
# actually reach above that particular grasp point -- see control.ik.reachable_above
# -- because a waypoint the arm cannot reach bends the whole path, not just its end.
APPROACH_HEIGHT = 0.05   # m above the grasp point for the standoff waypoint
LIFT_HEIGHT = 0.05       # m above the grasp point for the final waypoint
MIN_STANDOFF = 0.010     # m; below this the descent is not a descent
# The lift goes straight up only as far as it takes to clear the table, and then
# draws back toward the arm's own base.  Pinning it to fixed xy was costing most
# of its height: the top-down ceiling rises steeply on the way in, so a lift that
# tops out at 27 mm over the grasp point reaches the full 50 mm with 40 mm of
# draw-back.  Going straight up is only needed while the prop may still be
# touching the table, which is what the first leg is for.
LIFT_CLEARANCE = 0.015   # m of purely vertical lift before drawing back
LIFT_DRAWBACK = (0.0, 0.02, 0.04, 0.06)   # m of draw-back to try, nearest first
# The transit across the table is flown over the props, not through them, and is
# allowed a looser wrist than the grasp: it is a move, not a grasp, and demanding
# a strict top-down tool the whole way is what leaves it nothing to reach with
# where the standoff has had to clamp low.
TRANSIT_CLEARANCE = 0.015  # m above the tallest prop for the cross-table leg
TRANSIT_ROT_TOL = 0.50     # rad of tool tilt tolerated while in transit
CARTESIAN_STEPS = 12     # sub-waypoints along a straight-line gripper path
# The transit is the longest leg by far -- 185 mm against the descent's 50 -- and
# the tool's bow off the commanded line is second order in the joint step, so at
# twelve sub-steps it still dipped onto the prop it had come for.  More of them
# is the whole fix; they cost only IK solves.
TRANSIT_STEPS = 36
MIN_FINGER_Z = 0.010     # m above the table top, so the jaws clear the surface
# The gripper site sits on the *fixed* finger, not in the middle of the jaw
# aperture: opening the gripper swings only the moving jaw, so the aperture spans
# [site, site + opening] along the +z site axis.  Centring the site on the object
# therefore drops the fixed finger straight onto it.  The grasp point is offset
# back along the opening axis by half the object's width plus this clearance, so
# the aperture straddles the object instead.
JAW_CLEARANCE = 0.004    # m of daylight between the fixed finger and the object edge
LIFT_THRESHOLD = 0.03    # m of net rise that counts as a successful lift
# A grasp feature that is round in the horizontal plane grasps the same at every
# yaw, so for one of those the yaw is the planner's to choose, not the prop's.
# Taking it from the body's rotation instead throws reach away for nothing: on
# seed 0 the mug had to be received at a transfer spot where its landed yaw of
# 0.653 rad missed the grasp by 2.3 mm and left no lift at all, while 150 of the
# 169 cells both arms share are pickable at *some* yaw.  The pair is still tried
# first -- it is free and it is what a non-round feature is stuck with -- and the
# search widens only when the pair does not produce a grasp that can also lift.
ROUND_TOL = 0.001        # m; horizontal half-extents this close count as round
GRASP_YAW_BINS = 8       # yaws over [0, pi) tried for a round feature

# Placing: how high above the table the object's own origin is let go from, and
# how far the tool retreats afterwards before anything else happens.
PLACE_CLEARANCE = 0.004  # m of drop from release to rest
RELEASE_TIME = 0.5       # s to swing the jaws back open
PARK_TIME = 1.5          # s to send an arm back to its keyframe pose
RETREAT_HEIGHT = 0.05    # m of straight-up retreat after letting go
PLACE_TOL = 0.020        # m; how close to its goal a placed prop has to land
# A carried prop creeps in the jaws, and the release pose is built from a
# tool-to-prop offset measured before the carry starts.  Over the transit that
# offset goes stale: measured on the fork's 148 mm final leg, by 8.3 mm across
# the table and enough downward to drive the prop into the table on the descent,
# which levered it out of the jaws and left it 17.9 mm off a 20 mm tolerance.
# The offset is re-measured with the tool standing over the target and the load
# still held, and the descent aimed from that.  Corrections below this are noise.
REAIM_MIN = 0.001        # m of correction worth flying a lateral step for
# The place legs are timed by how far they go rather than given a fixed duration.
# A fixed one is wrong at both ends: the leg from the transit's via point down to
# the standoff is often a millimetre or two and was being given a second and a
# half of it, and a prop under a constant squeeze creeps out of the jaws the whole
# time it is held -- measured, a fork slid 6.4 mm down the pads over one carry and
# was gone by the end of the next leg.  Time under load is the thing to spend
# sparingly.
TOOL_SPEED = 0.10        # m/s along a carried leg
MIN_LEG_TIME = 0.15      # s, so a very short leg is still ramped rather than stepped
MAX_LEG_TIME = 1.5       # s

# Waypoint durations in seconds; converted to steps with the model timestep.
SETTLE_TIME = 0.5
OPEN_TIME = 0.4          # s to swing the jaws back to their stop before moving
TRANSIT_TIME = 1.2       # s for the cross-table leg, above every prop
APPROACH_TIME = 1.5
DESCEND_TIME = 1.0
# Closing is now a torque held until the jaw arrives, not a servo step, so it has
# to be given the jaw's actual travel time.  The jaw retracts to its stop at
# 1.745 rad and the fork's 12 mm bar is reached near 0 rad; against 0.6 N.m.s/rad
# of joint damping, 0.8 N.m settles at about 1.2 rad/s, so ~1.5 s of sweep.
CLOSE_RAMP_TIME = 0.25   # s to swing the command from open to squeeze
CLOSE_TIME = 2.0         # s of held squeeze, sized to the sweep above
CLOSE_SLICE = 0.02       # s between checks for the jaws having loaded up
EASE_TIME = 0.15         # s to drop from the closing torque to the holding one
LIFT_TIME = 1.5
HOLD_TIME = 0.5


# ---------------------------------------------------------------------------
# Scene queries
# ---------------------------------------------------------------------------


def _body_geoms(model: mujoco.MjModel, bid: int) -> List[int]:
    return [g for g in range(model.ngeom) if model.geom_bodyid[g] == bid]


def _body_bbox(model: mujoco.MjModel, bid: int,
               geoms: Optional[List[int]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Axis-aligned bounding box of a body's geoms, in the body frame.

    Returns ``(centre, half_extent)``.  Each geom's own local AABB corners are
    rotated into the body frame, which handles the offset/rotated primitives the
    props are built from.  ``geoms`` restricts the box to a subset -- the grasp
    feature, rather than the whole prop.
    """
    corners = []
    for gid in (_body_geoms(model, bid) if geoms is None else geoms):
        c, h = model.geom_aabb[gid][:3], model.geom_aabb[gid][3:]
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, model.geom_quat[gid])
        rot = rot.reshape(3, 3)
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    local = c + h * np.array([sx, sy, sz])
                    corners.append(model.geom_pos[gid] + rot @ local)
    corners = np.asarray(corners)
    lo, hi = corners.min(axis=0), corners.max(axis=0)
    return (lo + hi) / 2.0, (hi - lo) / 2.0


def _lowest_z(model: mujoco.MjModel, data: mujoco.MjData, bid: int) -> float:
    """World z of the lowest point of a body's geoms.

    ``geom_rbound`` is a bounding *sphere*, so using it here understates the
    lowest point by the geom's own aspect ratio -- measured on this scene's props,
    by 4.7 mm on the mug and 9.2 mm on the plate.  That error is spent twice over:
    it inflates the height a carry is told to clear by, which pushes the transit's
    via point out of the top-down envelope, and it makes ``transit_clear`` report
    a failure that is not there.  The oriented AABB corners are exact for the
    boxes, cylinders and capsules the props are built from.
    """
    lows = []
    for gid in _body_geoms(model, bid):
        c, h = model.geom_aabb[gid][:3], model.geom_aabb[gid][3:]
        rot = data.geom_xmat[gid].reshape(3, 3)
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    corner = c + h * np.array([sx, sy, sz])
                    lows.append(float((data.geom_xpos[gid] + rot @ corner)[2]))
    return min(lows)


def _table_top_z(model: mujoco.MjModel) -> float:
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table_top")
    bid = model.geom_bodyid[gid]
    return float(model.body_pos[bid][2] + model.geom_pos[gid][2] + model.geom_size[gid][2])


def _grasp_feature(model: mujoco.MjModel, bid: int) -> Dict[str, object]:
    """The grasp feature's box in the body frame, with the grasp slid to balance.

    Everything here is body-frame model geometry, so it does not depend on where
    the prop currently is.  :func:`grasp_pose` rotates it into the world to make a
    grasp; :func:`site_offset` needs only its magnitudes.  Shared so the two
    cannot drift apart.

    Along the feature, the grasp goes to the prop's balance point rather than the
    middle of the handle.  A parallel jaw resists a moment only through friction on
    two small pads, and the flatware's mass is nearly all in the head: gripping the
    centre of the fork's 32 mm bar holds it 7.23 mm off its centre of mass (the
    spoon, 4.18 mm).  Measured, that moment rotates the prop in the jaws for the
    whole of a carry -- the fork's hang below the tool grew 43.4 mm to 59.9 mm
    across one transit with the grip force never dropping below 2.24 N, so no
    amount of watching the force notices it -- and it eventually pivots out
    altogether, dropping the prop 46 mm onto the table.

    Both balance points lie inside their own grasp feature, so this costs nothing:
    the clip is what keeps the jaws on the feature for a prop whose centre of mass
    is off the end of it, and the residual arm is reported rather than assumed away.
    """
    # A prop may declare its grasp feature as a geom named <prop>_grasp.  Without
    # that, the merged bounding box of a fork is dominated by its head and the
    # grasp lands on the wrong part at the wrong width.
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
    feature = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{name}_grasp")
    centre, half = _body_bbox(model, bid, [feature] if feature >= 0 else None)

    # Grip across whichever body axis is narrower in the horizontal plane.
    narrow = 0 if half[0] <= half[1] else 1
    along = 1 - narrow
    com = float(model.body_ipos[bid][along])
    balanced = float(np.clip(com, centre[along] - half[along], centre[along] + half[along]))
    slide = balanced - float(centre[along])
    centre = centre.copy()
    centre[along] = balanced
    return {"centre": centre, "half": half, "narrow": narrow, "along": along,
            "width": 2.0 * float(half[narrow]), "feature": feature,
            "moment_arm": abs(com - balanced), "slide": slide,
            "yaw_free": bool(abs(half[0] - half[1]) <= ROUND_TOL)}


def site_offset(model: mujoco.MjModel, object_name: str) -> Dict[str, float]:
    """How far the tool site stands from a prop's own origin, in the plane.

    The reach map is indexed by where the *prop* goes, but what has to be
    reachable is where the *tool site* goes, and two body-frame offsets separate
    them.  The grasp point is the balance point of the grasp feature, which need
    not sit over the body origin -- the fork's is 4.8 mm off it.  And the aperture
    opens to one side of the site, so the site stands the grasp's half-width plus
    its clearance back from the grasp point: 22 mm for the mug, which is two cells
    of the reach map's 10 mm grid.

    Both offsets rotate with the prop, and their sum is a fixed body-frame vector,
    so the site sits on a circle of this radius about the prop's origin.  A goal
    slot cannot know what yaw the prop will arrive at, so the radius is what a
    reach test has to respect -- measured, ignoring it is what let seed 0's goal
    layout hand the mug a slot whose release pose its receiving arm misses by
    7.5 mm, and what let a transfer spot be chosen where the receiving grasp
    missed by 2.3 mm with no lift left at all.

    Returns the two magnitudes and the radius, in metres.
    """
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_name)
    if bid < 0:
        raise ValueError(f"scene has no body {object_name}")
    f = _grasp_feature(model, bid)
    jaw = f["width"] / 2.0 + JAW_CLEARANCE
    lead = np.asarray(f["centre"][:2], dtype=float)
    # The jaw offset runs along the narrow (opening) axis, either way round; the
    # lead is mostly along the other one.  Take the worse of the two.
    step = np.zeros(2)
    step[f["narrow"]] = jaw
    radius = max(float(np.linalg.norm(lead + s * step)) for s in (1.0, -1.0))
    return {"lead": float(np.linalg.norm(lead)), "jaw": float(jaw), "radius": radius}


def grasp_pose(model: mujoco.MjModel, data: mujoco.MjData,
               object_name: str) -> Tuple[np.ndarray, float, Dict[str, object]]:
    """Where and how to grasp ``object_name`` from above.

    The jaws are aligned with the object's shorter horizontal axis, and the grasp
    point is the object's mid-height, raised if needed so the fingers clear the
    table.  Returns ``(position, opening-axis yaw, geometry info)``.
    """
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_name)
    if bid < 0:
        raise ValueError(f"scene has no body {object_name}")

    f = _grasp_feature(model, bid)
    centre, half, width, feature = f["centre"], f["half"], f["width"], f["feature"]
    rot = data.xmat[bid].reshape(3, 3)
    axis = rot[:, f["narrow"]]
    yaw = float(np.arctan2(axis[1], axis[0]))

    world_centre = data.xpos[bid] + rot @ centre

    # Vertical extent of the object in the world, from the rotated box corners.
    span = float(np.abs(rot @ np.diag(half)).sum(axis=1)[2])
    bottom, top = world_centre[2] - span, world_centre[2] + span
    grasp_z = max(world_centre[2], _table_top_z(model) + MIN_FINGER_Z)
    grasp_z = min(grasp_z, top)

    pos = np.array([world_centre[0], world_centre[1], grasp_z])
    info = {
        "grasp_width": width,
        "yaw_free": f["yaw_free"],
        "moment_arm": f["moment_arm"],
        "grasp_slide": f["slide"],
        "grasp_feature": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, feature) if feature >= 0 else None,
        "object_bottom_z": bottom,
        "object_top_z": top,
        "object_half_extent": half.tolist(),
    }
    return pos, yaw, info


def grasp_candidates(centre: np.ndarray, yaw: float,
                     width: float) -> Tuple[Tuple[np.ndarray, float], ...]:
    """The two equivalent top-down grasps of an object, as ``(site_pos, yaw)`` pairs.

    Jaw symmetry means ``yaw`` and ``yaw + pi`` grasp the same way, but because the
    aperture opens to one side of the site the two need *different* site positions
    -- the offset flips with the yaw.  Position and yaw therefore have to be chosen
    together, which is why this cannot go through :func:`control.ik.solve_top_down`.
    """
    out = []
    for candidate in (yaw, yaw + np.pi):
        offset = (width / 2.0 + JAW_CLEARANCE) * np.array([np.cos(candidate), np.sin(candidate)])
        out.append((np.array([centre[0] - offset[0], centre[1] - offset[1], centre[2]]),
                    float(candidate)))
    return tuple(out)


def _object_contacts(model: mujoco.MjModel, data: mujoco.MjData,
                     bid: int) -> Dict[str, int]:
    """Count the object's contacts against the table, the arms and everything else."""
    geoms = set(_body_geoms(model, bid))
    counts = {"table": 0, "arm": 0, "other": 0}
    for i in range(data.ncon):
        c = data.contact[i]
        pair = {c.geom1, c.geom2}
        hit = pair & geoms
        if not hit or len(pair) < 2:
            continue
        other = (pair - geoms).pop()
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[other]) or ""
        if name.startswith("table") or name == "world":
            counts["table"] += 1
        elif name.startswith(("left_", "right_")):
            counts["arm"] += 1
        else:
            counts["other"] += 1
    return counts


def grip_force(model: mujoco.MjModel, data: mujoco.MjData, bid: int,
               arm: str) -> Dict[str, float]:
    """Normal force each jaw presses on body ``bid`` with, in newtons.

    Broken out per jaw rather than summed, because the sum cannot tell a grasp
    from a shove: a real grasp squeezes, so the fixed finger and the moving jaw
    read the same magnitude, while a one-sided push shows up as force on one jaw
    and nothing on the other.  The contact normal force is the first component of
    the contact-frame wrench.
    """
    obj = set(_body_geoms(model, bid))
    jaws = {}
    for label, body in (("fixed", f"{arm}_gripper"), ("moving", f"{arm}_moving_jaw_so101_v1")):
        jbid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
        if jbid < 0:
            raise ValueError(f"scene has no body {body}")
        jaws[label] = set(_body_geoms(model, jbid))

    out = {"fixed": 0.0, "moving": 0.0}
    wrench = np.zeros(6)
    for i in range(data.ncon):
        pair = {data.contact[i].geom1, data.contact[i].geom2}
        if len(pair) < 2 or not pair & obj:
            continue
        other = (pair - obj).pop()
        for label, geoms in jaws.items():
            if other in geoms:
                mujoco.mj_contactForce(model, data, i, wrench)
                out[label] += abs(float(wrench[0]))
    return out


def jaw_gap(model: mujoco.MjModel, data: mujoco.MjData, arm: str, site: int) -> float:
    """Opening between the jaw faces, in metres, for the pose currently in ``data``.

    Measured off the collision meshes rather than assumed: the moving jaw swings
    on an arc, so the gap is not linear in the joint angle.  Only vertices near
    the grasp point count -- the jaw bodies extend well behind it.  Under torque
    control this is the only readout of where the jaw actually ended up, so it is
    recorded at every stage: a gap that stops at the object's width is a grasp, a
    gap that runs down to the hard stop is an empty hand.
    """
    origin = data.site_xpos[site]
    rot = data.site_xmat[site].reshape(3, 3)
    approach, opening = rot[:, 0], rot[:, 2]

    spans = {}
    for body in (f"{arm}_gripper", f"{arm}_moving_jaw_so101_v1"):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
        pts = []
        for gid in _body_geoms(model, bid):
            mesh = model.geom_dataid[gid]
            if mesh < 0 or model.geom_group[gid] != 3:
                continue
            a, n = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            verts = model.mesh_vert[a:a + n].astype(float)
            world = data.geom_xpos[gid] + verts @ data.geom_xmat[gid].reshape(3, 3).T
            local = world - origin
            pts.append(local[np.abs(local @ approach) < 0.025] @ opening)
        spans[body] = np.concatenate(pts) if pts else np.empty(0)
    moving, fixed = spans[f"{arm}_moving_jaw_so101_v1"], spans[f"{arm}_gripper"]
    if not len(moving) or not len(fixed):
        # Past roughly q = 0.6 the moving jaw has swung clear of the grasp region
        # altogether; there is no meaningful gap left to measure, only "wide open".
        return WIDE_OPEN
    return float(moving.min() - fixed.max())


# ---------------------------------------------------------------------------
# Motion
# ---------------------------------------------------------------------------


def _arm_actuators(model: mujoco.MjModel, arm: str) -> Tuple[np.ndarray, int]:
    ids = []
    for name in ARM_JOINTS:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{arm}_{name}")
        if aid < 0:
            raise ValueError(f"scene has no actuator {arm}_{name}")
        ids.append(aid)
    grip = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{arm}_gripper")
    if grip < 0:
        raise ValueError(f"scene has no actuator {arm}_gripper")
    return np.asarray(ids), grip


def _ramp(model: mujoco.MjModel, data: mujoco.MjData, act: np.ndarray, grip: int,
          q_target: Optional[np.ndarray], grip_target: Optional[float],
          duration: float) -> None:
    """Interpolate the arm's ctrl to the given targets, stepping the sim each tick.

    ``None`` targets hold whatever that channel is currently commanding, so the
    gripper can be driven without disturbing the arm and vice versa.  ``q_target``
    is in radians (the arm is on position control) and ``grip_target`` in
    newton-metres (the gripper is not); both are clipped to their ctrlrange.
    """
    steps = max(1, int(round(duration / model.opt.timestep)))
    q_start = data.ctrl[act].copy()
    g_start = float(data.ctrl[grip])
    q_end = q_start if q_target is None else np.clip(
        q_target, model.actuator_ctrlrange[act, 0], model.actuator_ctrlrange[act, 1])
    g_end = g_start if grip_target is None else float(np.clip(
        grip_target, model.actuator_ctrlrange[grip, 0], model.actuator_ctrlrange[grip, 1]))

    for i in range(1, steps + 1):
        alpha = i / steps
        data.ctrl[act] = q_start + alpha * (q_end - q_start)
        data.ctrl[grip] = g_start + alpha * (g_end - g_start)
        mujoco.mj_step(model, data)


def _slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Shortest-arc interpolation between two 3x3 rotations."""
    qa, qb = np.zeros(4), np.zeros(4)
    mujoco.mju_mat2Quat(qa, np.ascontiguousarray(a).ravel())
    mujoco.mju_mat2Quat(qb, np.ascontiguousarray(b).ravel())
    if qa @ qb < 0.0:
        qb = -qb
    dot = float(np.clip(qa @ qb, -1.0, 1.0))
    if dot > 0.9995:
        q = qa + t * (qb - qa)
    else:
        theta = np.arccos(dot)
        q = (np.sin((1.0 - t) * theta) * qa + np.sin(t * theta) * qb) / np.sin(theta)
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, q / np.linalg.norm(q))
    return mat.reshape(3, 3)


def _move_line(model: mujoco.MjModel, data: mujoco.MjData, act: np.ndarray, grip: int,
               solver: IKSolver, start: np.ndarray, end: np.ndarray, yaw: float,
               duration: float, steps: int = CARTESIAN_STEPS,
               probe: Optional[Callable[[int], None]] = None,
               clamp: bool = False, rot_tol: float = ROT_TOL,
               start_mat: Optional[np.ndarray] = None) -> Dict[str, object]:
    """Drive the gripper site along a straight line, re-solving IK at each step.

    Interpolating joint targets between two waypoints lets the tool frame bow
    away from the straight path between them, which is what drove the gripper
    into the object during the descent: the jaws arrived from the side and
    collided before they were around it.  Re-solving IK along the line keeps the
    approach purely vertical, so the jaws straddle the object and touch nothing
    until they close.
    """
    info: Dict[str, object] = {}
    goal_mat = top_down_mat(yaw)
    worst, every = 0.0, True
    for k in range(1, steps + 1):
        point = start + (end - start) * (k / steps)
        # ``start_mat`` spreads the wrist's turn over the leg.  Between two tool
        # poses the controller interpolates *joints*, so the tool bows off the
        # line, and the bow is second order in the joint step -- small when the
        # steps are small, large when one of them is not.  Demanding the grasp yaw
        # at the first sub-step is exactly such a step: from the parked pose it
        # swung the tool 39 mm below the line, through the fork it was going to
        # pick.  Turning a twelfth at a time keeps every step small.
        mat = goal_mat if start_mat is None else _slerp(start_mat, goal_mat, k / steps)
        # ``clamp`` is for a transit whose straight line leaves the envelope in
        # the middle even though both ends are inside it.  Each sub-step is pulled
        # toward the end of the leg until it converges, which keeps the motion
        # monotone toward the goal and every command a pose the arm can hold.
        if clamp:
            point, _ = reachable_toward(solver, data, point, end, mat, rot_tol=rot_tol)
        q, info = solver.solve(data, point, mat, rot_tol=rot_tol)
        worst = max(worst, float(info["pos_err"]))
        every = every and bool(info["converged"])
        _ramp(model, data, act, grip, q, None, duration / steps)
        if probe is not None:
            probe(k)
    # Clamping the endpoints is not the same as the whole line being flyable, so
    # the worst sub-step is reported too rather than only the one we land on.
    info["worst_pos_err"] = worst
    info["all_converged"] = every
    return info


def held_object(model: mujoco.MjModel, data: mujoco.MjData, arm: str) -> Optional[int]:
    """Which prop the jaws are holding, or ``None``.

    A prop counts as held when both pads are on it, which is what distinguishes a
    grasp from something merely leant against.  Nothing else in the scene can be
    in the jaws, so the search is over the props.
    """
    for bid in prop_bodies(model):
        force = grip_force(model, data, bid, arm)
        if min(force.values()) > 0.0:
            return bid
    return None


def _grip_transform(model: mujoco.MjModel, data: mujoco.MjData, arm: str,
                    bid: int) -> Tuple[np.ndarray, np.ndarray]:
    """Pose of the held body in the gripper body's frame, so it can be carried."""
    gb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{arm}_gripper")
    rot = data.xmat[gb].reshape(3, 3)
    return rot.T @ (data.xpos[bid] - data.xpos[gb]), rot.T @ data.xmat[bid].reshape(3, 3)


def _carry(model: mujoco.MjModel, scratch: mujoco.MjData, arm: str, bid: int,
           rel_pos: np.ndarray, rel_mat: np.ndarray, qadr: int) -> None:
    """Move the held body with the gripper, as a rigid grasp would."""
    gb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{arm}_gripper")
    rot = scratch.xmat[gb].reshape(3, 3)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.ascontiguousarray(rot @ rel_mat).ravel())
    scratch.qpos[qadr:qadr + 3] = scratch.xpos[gb] + rot @ rel_pos
    scratch.qpos[qadr + 3:qadr + 7] = quat


def carry_fouls(model: mujoco.MjModel, data: mujoco.MjData, solver: IKSolver,
                arm: str, start: np.ndarray, end: np.ndarray, yaw: float, bid: int,
                steps: int = CARTESIAN_STEPS, rot_tol: float = ROT_TOL) -> int:
    """Sub-steps where the arm, or the thing it is carrying, hits something else.

    The empty-handed check in :func:`path_fouls` is not the right question once
    there is a prop in the jaws: the assembly that has to fit through the gap is
    the gripper *plus* whatever it holds, and for a mug that is another 34 mm
    hanging below the tool.  So the held body is carried along with the gripper at
    the pose it was grasped at, and what counts as a foul is it touching the table
    or another prop -- its contact with the pads is the grasp, not a collision.
    """
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = data.qpos
    rel_pos, rel_mat = _grip_transform(model, data, arm, bid)
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) + "_free")
    qadr = model.jnt_qposadr[jid]
    others = [b for b in prop_bodies(model) if b != bid]

    fouls = 0
    q, _ = solver.solve(data, start, top_down_mat(yaw), rot_tol=rot_tol)
    for k in range(steps + 1):
        point = start + (end - start) * (k / steps)
        point, _ = reachable_toward(solver, scratch, point, end, top_down_mat(yaw),
                                    rot_tol=rot_tol)
        q, _ = solver.solve(scratch, point, top_down_mat(yaw), seed=q, rot_tol=rot_tol)
        scratch.qpos[solver.qadr] = q
        mujoco.mj_kinematics(model, scratch)
        _carry(model, scratch, arm, bid, rel_pos, rel_mat, qadr)
        mujoco.mj_forward(model, scratch)
        carried = _object_contacts(model, scratch, bid)
        fouls += bool(carried["table"] or carried["other"]
                      or any(_object_contacts(model, scratch, b)["arm"] for b in others))
    return fouls


def prop_bodies(model: mujoco.MjModel) -> List[int]:
    """Every body that is a loose prop, found by its free joint rather than by name."""
    return sorted({int(model.jnt_bodyid[j]) for j in range(model.njnt)
                   if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE})


def path_fouls(model: mujoco.MjModel, data: mujoco.MjData, solver: IKSolver,
               arm: str, start: np.ndarray, end: np.ndarray, yaw: float,
               bodies: Sequence[int], steps: int = CARTESIAN_STEPS,
               rot_tol: float = ROT_TOL) -> int:
    """Sub-steps of a straight-line path where the open gripper already touches the prop.

    A waypoint that converges is not yet a waypoint that can be flown to: the
    descent can still put the jaws through the object on the way in, and then the
    arm stalls against it while IK goes on reporting a millimetre of residual.
    Measured on the plate, whose two arms are within 0.01 mm of each other on
    grasp residual: one descends cleanly, the other arrives 14.42 mm short with a
    21.82 N one-sided shove on the prop.  Nothing kinematic about the grasp pose
    distinguishes them -- only the corridor does.

    Kinematic only.  Each sub-step's IK solution is written into a scratch state
    with the jaws run out to their stop, where they sit under the opening torque,
    and MuJoCo's own collision pass is asked what touches what.  No physics is
    stepped and no renderer is built.

    The seeding has to match :func:`_move_line` exactly or the check answers about
    a different arm.  Five joints against a six-DOF target leaves isolated
    solution branches, and DLS lands in whichever one its seed is nearest: solving
    each sub-step cold from the home pose put this arm on a branch with a
    different shoulder_pan, whose pad clears the plate by 5 mm, while the branch
    the motion actually flies -- warm-started at the standoff and then chained
    down the line -- puts the same pad 13.3 mm from the plate's axis, inside its
    17 mm rim.  So the standoff is solved from ``data`` as the approach does, and
    every sub-step from the one before it.

    ``bodies`` is every prop the path could touch, not only the one being picked:
    a leg that crosses the table can just as well sweep a prop it has no interest
    in, and the sweep is what leaves the descent aiming at a prop that has moved.
    """
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = data.qpos
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{arm}_gripper")
    scratch.qpos[model.jnt_qposadr[jid]] = model.jnt_range[jid, 1]

    fouls = 0
    q, _ = solver.solve(data, start, top_down_mat(yaw), rot_tol=rot_tol)
    for k in range(steps + 1):
        point = start + (end - start) * (k / steps)
        point, _ = reachable_toward(solver, scratch, point, end, top_down_mat(yaw),
                                    rot_tol=rot_tol)
        q, _ = solver.solve(scratch, point, top_down_mat(yaw), seed=q, rot_tol=rot_tol)
        scratch.qpos[solver.qadr] = q
        mujoco.mj_forward(model, scratch)
        fouls += any(_object_contacts(model, scratch, b)["arm"] for b in bodies)
    return fouls


def _plan_lift(model: mujoco.MjModel, data: mujoco.MjData, solver: IKSolver, arm: str,
               target: np.ndarray, yaw: float, seed: np.ndarray):
    """Where the lift can actually get to: straight up to clear, then drawn back.

    The lift used to be a vertical line at the grasp point's own xy, which throws
    away most of the height available.  Top-down reach improves sharply toward the
    arm's base -- measured, a fork whose vertical lift clamps at 27 mm over the
    grasp point makes the full 50 mm once the lift is allowed 40 mm of draw-back,
    and the same holds for the mug and spoon.  Going straight up matters only
    while the prop may still be in contact with the table; after that, drawing
    back is free.

    Returns the drawn-back top of the lift, the vertical clearance point, and how
    far back the winner drew, preferring the least draw-back that reaches highest.
    """
    clear, c_info = reachable_above(solver, data, target, yaw, LIFT_CLEARANCE, seed=seed)
    mount = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{arm}_mount")
    if mount < 0:
        raise ValueError(f"scene has no body {arm}_mount")
    inward = model.body_pos[mount][:2] - target[:2]
    norm = float(np.linalg.norm(inward))
    inward = inward / norm if norm > 1e-9 else np.zeros(2)

    best = None
    for back in LIFT_DRAWBACK:
        base = target.copy()
        base[:2] += inward * back
        pos, info = reachable_above(solver, data, base, yaw, LIFT_HEIGHT, seed=seed)
        if best is None or info["height"] > best[1]["height"] + 1e-4:
            best = (pos, info, back)
    return best[0], best[1], clear, c_info, best[2]


def plan_grasp(model: mujoco.MjModel, data: mujoco.MjData, arm: str,
               object_name: str) -> Dict[str, object]:
    """Choose the grasp pose and both waypoint heights for ``arm`` on ``object_name``.

    Both members of the symmetric yaw pair are costed, and the one that lets the
    arm lift highest wins: a yaw that grasps but cannot then go anywhere is not
    the better grasp.  Pure kinematics, so :func:`best_arm` can use it to compare
    arms without stepping the simulation.
    """
    solver = IKSolver(model, arm)
    props = prop_bodies(model)
    centre, yaw0, geom = grasp_pose(model, data, object_name)

    ceiling = max(float(data.geom_xpos[g][2] + model.geom_rbound[g])
                  for b in props for g in _body_geoms(model, b))
    parked = data.site_xpos[solver.site].copy()

    def consider(pos, candidate, best):
        q_low, low = solver.solve(data, pos, top_down_mat(candidate))
        # Seeded the way each motion is: the approach solves the standoff cold
        # from wherever the arm is, the lift solves warm from the grasp pose.
        stand, s_info = reachable_above(solver, data, pos, candidate, APPROACH_HEIGHT,
                                        floor=MIN_STANDOFF)
        lift, l_info, clear, c_info, back = _plan_lift(model, data, solver, arm, pos,
                                                       candidate, q_low)
        # As high as the arm is already holding the tool, so the cross-table leg is
        # level rather than a long descending diagonal.  A diagonal is low by the
        # time it arrives, which is where it catches the prop it came for.
        via = np.array([stand[0], stand[1],
                        max(stand[2], ceiling + TRANSIT_CLEARANCE, parked[2])])
        via, _ = reachable_toward(solver, data, via, stand, top_down_mat(candidate),
                                  rot_tol=TRANSIT_ROT_TOL)
        # Every leg the arm will fly before it has hold of anything, checked the
        # same way: the transit across the table, the drop onto the standoff, and
        # the descent.  Checking only the descent is what let a candidate through
        # whose transit knocked the prop 10.7 mm before the jaws ever closed.
        fouls = (path_fouls(model, data, solver, arm, parked, via, candidate, props,
                            steps=TRANSIT_STEPS, rot_tol=TRANSIT_ROT_TOL)
                 + path_fouls(model, data, solver, arm, via, stand, candidate, props,
                              rot_tol=TRANSIT_ROT_TOL)
                 + path_fouls(model, data, solver, arm, stand, pos, candidate, props))
        score = (not low["converged"], fouls, -l_info["height"],
                 low["pos_err"] + low["rot_err"])
        if best is None or score < best[0]:
            best = (score, pos, candidate, low, stand, s_info, lift, l_info, fouls,
                    clear, c_info, back, via)
        return best

    # The symmetric pair first; more yaws only if the pair cannot both grasp and
    # lift, and only for a feature round enough that another yaw grasps the same.
    groups = [grasp_candidates(centre, yaw0, geom["grasp_width"])]
    if geom["yaw_free"]:
        groups.append(tuple(c for k in range(1, GRASP_YAW_BINS)
                            for c in grasp_candidates(
                                centre, yaw0 + np.pi * k / GRASP_YAW_BINS,
                                geom["grasp_width"])))
    best, widened = None, False
    for group in groups:
        for pos, candidate in group:
            best = consider(pos, candidate, best)
        # A converged grasp that can also lift is all the pair has to produce.
        if not best[0][0] and -best[0][2] > LIFT_THRESHOLD:
            break
        widened = len(groups) > 1
    geom["yaw_widened"] = bool(widened)

    (_, target, yaw, low, standoff, s_info, lifted_to, l_info, fouls,
     clear_to, c_info, drawback, hover) = best
    geom["chosen_yaw"] = yaw
    geom["jaw_offset"] = float(np.linalg.norm(target[:2] - centre[:2]))
    geom["grasp_ik"] = {"at_grasp": low["pos_err"], "converged": low["converged"]}
    geom["standoff"] = {k: s_info[k] for k in ("height", "clamped", "floor_ok")}
    geom["lift"] = {k: l_info[k] for k in ("height", "clamped", "floor_ok")}
    geom["lift"]["drawback"] = float(drawback)
    geom["lift_clearance"] = float(c_info["height"])
    # A standoff clamped below the object's top starts the descent alongside the
    # prop rather than above it.  With the jaws run back to their stop that is
    # measured clear, but it is worth knowing when it happens.
    geom["standoff_below_object"] = bool(target[2] + s_info["height"] < geom["object_top_z"])
    geom["descent_fouls"] = int(fouls)

    geom["hover_height"] = float(hover[2] - target[2])
    geom["transit_clear"] = bool(hover[2] >= ceiling + TRANSIT_CLEARANCE)
    return {"solver": solver, "centre": centre, "target": target, "yaw": yaw,
            "standoff": standoff, "hover": hover, "clear_to": clear_to,
            "lifted_to": lifted_to, "geometry": geom,
            "grasp_ok": bool(low["converged"]), "lift_height": float(l_info["height"]),
            "fouls": int(fouls)}


def best_arm(model: mujoco.MjModel, data: mujoco.MjData, object_name: str,
             arms: Sequence[str] = ("left", "right")) -> Tuple[str, Dict[str, object]]:
    """Which arm should pick ``object_name``: the one that can grasp it and lift highest.

    Reach is not symmetric -- the props are placed by a per-arm reach map, and a
    prop out at one side is often graspable by one arm only -- so the arm cannot
    be fixed per prop across seeds.  Ordered by whether the grasp converges, then
    by a descent corridor clear of the prop, then by lift height, then by grasp
    residual.  Residual alone is useless as a tie-break here: on the plate the two
    arms differ by 0.01 mm and one of them drives straight into it.

    ``arms`` narrows the choice, which is how a caller that also has to *put the
    prop somewhere* keeps the two halves of the job on one arm.
    """
    best = None
    for arm in arms:
        plan = plan_grasp(model, data, arm, object_name)
        score = (not plan["grasp_ok"], plan["fouls"], -plan["lift_height"],
                 plan["geometry"]["grasp_ik"]["at_grasp"])
        if best is None or score < best[0]:
            best = (score, arm, plan)
    return best[1], best[2]


def _squeeze(model: mujoco.MjModel, data: mujoco.MjData, act: np.ndarray, grip: int,
             arm: str, bid: int, duration: float) -> bool:
    """Close the jaws, then ease the command once they have loaded up.

    Closing and holding want different torques.  The closing one has to sweep the
    jaw all the way in from its stop against the joint's damping; the holding one
    only has to carry the object.  Leaving the closing torque on crushes through
    anything whose contact is soft enough, and how soft that is depends on the
    object -- see :data:`control.gripper.HOLD_TORQUE`.

    Returns whether the jaws ever loaded up, which is the difference between a
    grasp that was eased and one that closed on nothing.
    """
    _ramp(model, data, act, grip, None, GRIP_TORQUE, CLOSE_RAMP_TIME)
    eased = False
    for _ in range(max(1, int(round(duration / CLOSE_SLICE)))):
        _ramp(model, data, act, grip, None, None, CLOSE_SLICE)
        if not eased and min(grip_force(model, data, bid, arm).values()) >= GRIP_TRIGGER:
            _ramp(model, data, act, grip, None, HOLD_TORQUE, EASE_TIME)
            eased = True
    return eased


def pick(model: mujoco.MjModel, data: mujoco.MjData, arm: str,
         object_name: str) -> Dict[str, object]:
    """Run a scripted top-down pick of ``object_name`` with ``arm``.

    Steps the simulation in place through: settle, move above the object with the
    gripper open, descend, close, lift.  Returns a report of what the IK asked
    for, what the arm achieved and whether the object came up.
    """
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_name)
    if bid < 0:
        raise ValueError(f"scene has no body {object_name}")

    act, grip = _arm_actuators(model, arm)

    # Let the props drop the couple of millimetres they spawn above the table.
    _ramp(model, data, act, grip, None, None, SETTLE_TIME)
    rest_z = float(data.xpos[bid][2])
    rest_xy = data.xpos[bid][:2].copy()

    plan = plan_grasp(model, data, arm, object_name)
    solver = plan["solver"]
    centre, target, yaw = plan["centre"], plan["target"], plan["yaw"]
    standoff, hover = plan["standoff"], plan["hover"]
    clear_to, lifted_to, geom = plan["clear_to"], plan["lifted_to"], plan["geometry"]
    stages: List[Dict[str, object]] = []

    def record(name: str, ik: Optional[Dict[str, object]], goal: Optional[np.ndarray]) -> None:
        r: Dict[str, object] = {"stage": name, "site_pos": data.site_xpos[solver.site].tolist()}
        if ik is not None:
            r["ik"] = ik
            r["target_pos"] = np.asarray(goal).tolist()
            r["site_err"] = float(np.linalg.norm(data.site_xpos[solver.site] - goal))
        r["object_z"] = float(data.xpos[bid][2])
        r["contacts"] = _object_contacts(model, data, bid)
        r["jaw_gap"] = jaw_gap(model, data, arm, solver.site)
        r["grip_force"] = grip_force(model, data, bid, arm)
        # Horizontal shove, which z alone hides: a prop knocked sideways before
        # the grasp leaves the descent aimed at where it used to be.
        r["object_shift"] = float(np.linalg.norm(data.xpos[bid][:2] - rest_xy))
        stages.append(r)

    # Approach through free space above the object, opening the jaws on the way.
    # There is no open *angle* to choose any more: the opening torque runs the jaw
    # back to its stop at 1.745 rad, which is further open than the 1.10 rad the
    # position servo used to be given, and further open is the safe direction --
    # the moving jaw's tip retracts from 27 mm to 65 mm behind the grasp point, so
    # it cannot graze the prop on the way down.
    geom["open_torque"], geom["grip_torque"] = OPEN_TORQUE, GRIP_TORQUE

    # Cartesian, not a joint ramp, and in two legs: across the table above every
    # prop, then straight down onto the standoff.  Interpolating joint targets
    # bows the tool off the line, and flying straight at a low standoff rakes
    # whatever is in between; both were measured to shove props before the grasp.
    _ramp(model, data, act, grip, None, OPEN_TORQUE, OPEN_TIME)
    _move_line(model, data, act, grip, solver, data.site_xpos[solver.site].copy(),
               hover, yaw, TRANSIT_TIME, steps=TRANSIT_STEPS, clamp=True,
               rot_tol=TRANSIT_ROT_TOL,
               start_mat=data.site_xmat[solver.site].reshape(3, 3).copy())
    ik = _move_line(model, data, act, grip, solver, hover, standoff, yaw,
                    APPROACH_TIME, clamp=True, rot_tol=TRANSIT_ROT_TOL)
    record("approach", ik, standoff)

    # Straight down onto the grasp point, jaws held open and clear of the object.
    ik = _move_line(model, data, act, grip, solver, standoff, target, yaw, DESCEND_TIME)
    record("descend", ik, target)

    geom["eased"] = _squeeze(model, data, act, grip, arm, bid, CLOSE_TIME)
    record("close", None, None)

    # Straight up first, so a successful grasp is not levered against the table
    # while the prop may still be touching it, then drawn back toward the base
    # where the top-down ceiling is higher.  The squeeze is sampled at every
    # sub-step: whether the torque command holds its force through the motion is
    # the question this primitive originally failed on.
    trace: List[Dict[str, object]] = []

    def sample(k: int) -> None:
        f = grip_force(model, data, bid, arm)
        trace.append({"step": k,
                      "height": float(data.site_xpos[solver.site][2] - target[2]),
                      "object_z": float(data.xpos[bid][2]),
                      "jaw_gap": jaw_gap(model, data, arm, solver.site),
                      "grip_force": f})

    ik = _move_line(model, data, act, grip, solver, target, clear_to, yaw,
                    LIFT_TIME * 0.3, probe=sample)
    ik = _move_line(model, data, act, grip, solver, clear_to, lifted_to, yaw,
                    LIFT_TIME * 0.7, probe=sample)
    record("lift", ik, lifted_to)

    _ramp(model, data, act, grip, None, None, HOLD_TIME)

    final_z = float(data.xpos[bid][2])
    contacts = _object_contacts(model, data, bid)
    rise = final_z - rest_z
    lifted = bool(rise > LIFT_THRESHOLD and contacts["table"] == 0 and contacts["arm"] > 0)

    return {
        "arm": arm,
        "object": object_name,
        "grasp_target": target.tolist(),
        "grasp_yaw": yaw,
        "object_centre": centre.tolist(),
        "geometry": geom,
        "rest_z": rest_z,
        "approach_shift": float(stages[0]["object_shift"]),
        "final_z": final_z,
        "rise": rise,
        "contacts": contacts,
        "lifted": lifted,
        "stages": stages,
        "lift_trace": trace,
    }


# ---------------------------------------------------------------------------
# Placing
# ---------------------------------------------------------------------------


def _leg_time(start: np.ndarray, end: np.ndarray) -> float:
    """How long to spend on a leg, from its length at a fixed tool speed."""
    return float(np.clip(np.linalg.norm(np.asarray(end) - np.asarray(start)) / TOOL_SPEED,
                         MIN_LEG_TIME, MAX_LEG_TIME))


def _carry_drop(model: mujoco.MjModel, data: mujoco.MjData, solver: IKSolver,
                bid: int) -> float:
    """How far below the tool the held prop's lowest point hangs, in metres."""
    return float(data.site_xpos[solver.site][2]) - _lowest_z(model, data, bid)


def _hold_offset(model: mujoco.MjModel, data: mujoco.MjData, solver: IKSolver,
                 bid: int) -> np.ndarray:
    """Where the held prop sits relative to the tool site, right now."""
    return data.xpos[bid] - data.site_xpos[solver.site]


def _release_pose(model: mujoco.MjModel, offset: np.ndarray, drop: float,
                  target_xy: Sequence[float]) -> np.ndarray:
    """Tool pose that puts a prop hanging at ``offset`` down on ``target_xy``.

    The xy comes from the offset, because the goal is written against the prop's
    body origin.  The height does not: it is set from ``drop``, the hang of the
    prop's *lowest* point below the tool, so that ``PLACE_CLEARANCE`` is a real
    gap under the whole prop.

    Measuring the height off the origin instead is what made the descent eject
    the load.  A prop resting on the table has its origin at its lowest point,
    but a prop in the jaws is tilted -- the fork carried here hangs 39.8 mm below
    the tool at its origin and 43.4 mm at its lowest tine.  Releasing at "origin
    4 mm up" therefore left the tine 0.4 mm off the table, the descent drove it
    in, and the reaction popped the fork out of the jaws: the gap snapped from
    10.6 mm to 4.0 mm, the grip force went to zero, and it fell the whole 46 mm.
    """
    return np.array([target_xy[0] - offset[0], target_xy[1] - offset[1],
                     _table_top_z(model) + PLACE_CLEARANCE + drop])


def _reaim(model: mujoco.MjModel, data: mujoco.MjData, solver: IKSolver, bid: int,
           yaw: float, target_xy: Sequence[float],
           plan: Dict[str, object]) -> Optional[Dict[str, object]]:
    """Re-derive the release from where the prop is in the jaws *now*.

    The offset the plan's release was built from was measured before the carry,
    and the prop creeps in the jaws for the whole of it.  Returns the corrected
    standoff, release and retreat, or ``None`` when the correction is beneath
    noise or its poses are not reachable -- in which case the planned ones stand,
    which is no worse than not looking.
    """
    release = _release_pose(model, _hold_offset(model, data, solver, bid),
                            _carry_drop(model, data, solver, bid), target_xy)
    shift = release - np.asarray(plan["release"], dtype=float)
    if float(np.linalg.norm(shift)) < REAIM_MIN:
        return None
    _, info = solver.solve(data, release, top_down_mat(yaw))
    if not info["converged"]:
        return None
    # Directly over the corrected release at the standoff the plan chose, so the
    # whole correction is spent laterally at height and the descent stays
    # vertical.  A slanted descent drags the load into whatever is next to it.
    standoff = float(plan["above"][2] - np.asarray(plan["release"], dtype=float)[2])
    above, a_info = reachable_above(solver, data, release, yaw, standoff,
                                    floor=MIN_STANDOFF)
    retreat, _ = reachable_above(solver, data, release, yaw, RETREAT_HEIGHT,
                                floor=MIN_STANDOFF)
    return {"above": above, "release": release, "retreat": retreat,
            "report": {"above": above.tolist(), "release": release.tolist(),
                       "retreat": retreat.tolist(),
                       "shift": [float(v) for v in shift],
                       "lateral": float(np.linalg.norm(shift[:2])),
                       "sink": float(shift[2])},
            "shift": [float(v) for v in shift],
            "lateral": float(np.linalg.norm(shift[:2])), "sink": float(shift[2]),
            "standoff_ok": bool(a_info["floor_ok"])}


def _grip_faded(model: mujoco.MjModel, data: mujoco.MjData, arm: str, bid: int) -> bool:
    """Whether the jaws have lost their grip on ``bid``.  Observation only.

    This used to re-seat the squeeze when it fired, and that made things worse in
    both of the ways it could.  A re-squeeze at the closing torque applies the
    same 0.8 N.m sweep that :mod:`control.gripper` measures as ejecting light
    flatware outright, and re-seating gently instead still stops the arm for
    0.15 s each time, spending exactly the time under load that makes a wedged
    prop creep.  Measured on the spoon's carry to the transfer spot, over the 28
    sub-steps at which the grip had faded: re-squeezing at the closing torque put
    it down 84.3 mm off, re-seating at the holding torque 96.0 mm off, and doing
    nothing at all 6.0 mm off.

    What the fade was really reporting was a grasp taken off the prop's centre of
    mass, which :func:`grasp_pose` now corrects at the source.  So the fade is
    counted and reported, and nothing is done about it mid-carry.
    """
    return bool(min(grip_force(model, data, bid, arm).values()) < GRIP_TRIGGER / 2.0)


def plan_place(model: mujoco.MjModel, data: mujoco.MjData, arm: str,
               target_xy: Sequence[float]) -> Optional[Dict[str, object]]:
    """Where to fly to let the held prop down at ``target_xy``.

    Returns ``None`` when the jaws are empty, which is the honest answer to
    "place what?".  The tool keeps the orientation it grasped at, so the offset
    from tool to prop is a constant translation and the release pose follows from
    where the prop has to end up rather than from where the tool has to be.
    """
    bid = held_object(model, data, arm)
    if bid is None:
        return None

    solver = IKSolver(model, arm)
    site = data.site_xpos[solver.site].copy()
    rot = data.site_xmat[solver.site].reshape(3, 3)
    yaw = float(np.arctan2(rot[1, 2], rot[0, 2]))      # site +z is the opening axis
    offset = _hold_offset(model, data, solver, bid)
    drop = _carry_drop(model, data, solver, bid)

    release = _release_pose(model, offset, drop, target_xy)
    _, r_info = solver.solve(data, release, top_down_mat(yaw))

    above, a_info = reachable_above(solver, data, release, yaw, APPROACH_HEIGHT,
                                    floor=MIN_STANDOFF)
    # The transit has to clear the other props by the height of whatever is
    # hanging from the jaws, not merely by the tool's own.
    others = [b for b in prop_bodies(model) if b != bid]
    ceiling = max([float(data.geom_xpos[g][2] + model.geom_rbound[g])
                   for b in others for g in _body_geoms(model, b)] or [_table_top_z(model)])
    via = np.array([above[0], above[1],
                    max(above[2], ceiling + TRANSIT_CLEARANCE + drop, site[2])])
    via, _ = reachable_toward(solver, data, via, above, top_down_mat(yaw),
                              rot_tol=TRANSIT_ROT_TOL)

    fouls = (carry_fouls(model, data, solver, arm, site, via, yaw, bid,
                         steps=TRANSIT_STEPS, rot_tol=TRANSIT_ROT_TOL)
             + carry_fouls(model, data, solver, arm, via, above, yaw, bid,
                           rot_tol=TRANSIT_ROT_TOL)
             + carry_fouls(model, data, solver, arm, above, release, yaw, bid))
    retreat, t_info = reachable_above(solver, data, release, yaw, RETREAT_HEIGHT,
                                      floor=MIN_STANDOFF)
    return {
        "solver": solver, "body": bid, "yaw": yaw, "via": via, "above": above,
        "release": release, "retreat": retreat, "carry_drop": drop, "fouls": int(fouls),
        "release_ok": bool(r_info["converged"]), "release_ik": r_info["pos_err"],
        "approach": {k: a_info[k] for k in ("height", "clamped", "floor_ok")},
        "retreat_height": {k: t_info[k] for k in ("height", "clamped", "floor_ok")},
        "transit_clear": bool(via[2] >= ceiling + TRANSIT_CLEARANCE + drop),
    }


def place(model: mujoco.MjModel, data: mujoco.MjData, arm: str,
          target_xy: Sequence[float]) -> Dict[str, object]:
    """Put whatever ``arm`` is holding down at ``target_xy``.

    Transit above everything else on the table, descend onto the spot, open, and
    retreat straight up so the jaws do not drag what they just let go of.  The
    legs are the same two-phase, clamped, corridor-checked motions the pick uses;
    the difference is that the moving assembly is the gripper plus its load, so
    the clearances are measured from the load's lowest point.
    """
    act, grip = _arm_actuators(model, arm)
    plan = plan_place(model, data, arm, target_xy)
    if plan is None:
        return {"arm": arm, "object": None, "placed": False, "reason": "nothing held",
                "target_xy": list(map(float, target_xy)), "final_xy": None,
                "final_z": None, "error": float("inf"), "carry_drop": 0.0,
                "fouls": 0, "release_ok": False, "transit_clear": False,
                "approach": None, "contacts": None, "stages": []}

    solver, bid, yaw = plan["solver"], plan["body"], plan["yaw"]
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
    if not plan["release_ok"]:
        # Flying at a release the arm cannot reach does not put the prop down
        # somewhere slightly wrong, it puts it down somewhere else entirely: the
        # fork missed its goal by 138 mm that way.  Better to hold on and say so.
        return {"arm": arm, "object": name, "placed": False,
                "reason": f"release pose off by {plan['release_ik'] * 1000:.1f} mm",
                "target_xy": list(map(float, target_xy)), "final_xy": None,
                "final_z": None, "error": float("inf"),
                "carry_drop": plan["carry_drop"], "fouls": plan["fouls"],
                "release_ok": False, "transit_clear": plan["transit_clear"],
                "approach": plan["approach"], "contacts": None, "stages": []}
    stages: List[Dict[str, object]] = []

    def record(stage: str, ik: Optional[Dict[str, object]],
               goal: Optional[np.ndarray]) -> None:
        r: Dict[str, object] = {"stage": stage, "object_z": float(data.xpos[bid][2]),
                                "object_xy": data.xpos[bid][:2].tolist(),
                                "site_z": float(data.site_xpos[solver.site][2]),
                                "hang": _carry_drop(model, data, solver, bid),
                                "contacts": _object_contacts(model, data, bid),
                                "jaw_gap": jaw_gap(model, data, arm, solver.site),
                                "grip_force": grip_force(model, data, bid, arm)}
        if ik is not None:
            r["ik"] = ik
            r["site_err"] = float(np.linalg.norm(data.site_xpos[solver.site] - goal))
        stages.append(r)

    fades = [0]

    def hold(_k: int) -> None:
        fades[0] += _grip_faded(model, data, arm, bid)

    here = data.site_xpos[solver.site].copy()
    _move_line(model, data, act, grip, solver, here, plan["via"], yaw,
               _leg_time(here, plan["via"]), steps=TRANSIT_STEPS, clamp=True,
               rot_tol=TRANSIT_ROT_TOL, probe=hold,
               start_mat=data.site_xmat[solver.site].reshape(3, 3).copy())
    ik = _move_line(model, data, act, grip, solver, plan["via"], plan["above"], yaw,
                    _leg_time(plan["via"], plan["above"]), clamp=True,
                    rot_tol=TRANSIT_ROT_TOL, probe=hold)
    record("carry", ik, plan["above"])

    # Everything the carry accumulated is corrected here, in one lateral step at
    # height, so that only the descent's own creep is left to land as error.
    aim = _reaim(model, data, solver, bid, yaw, target_xy, plan)
    above, release, retreat = plan["above"], plan["release"], plan["retreat"]
    if aim is not None:
        above, release, retreat = aim["above"], aim["release"], aim["retreat"]
        ik = _move_line(model, data, act, grip, solver, plan["above"], above, yaw,
                        _leg_time(plan["above"], above), probe=hold)
        record("re-aim", ik, above)

    ik = _move_line(model, data, act, grip, solver, above, release, yaw,
                    _leg_time(above, release))
    record("descend", ik, release)

    _ramp(model, data, act, grip, None, OPEN_TORQUE, RELEASE_TIME)
    record("release", None, None)

    ik = _move_line(model, data, act, grip, solver, release, retreat,
                    yaw, _leg_time(release, retreat))
    record("retreat", ik, retreat)
    _ramp(model, data, act, grip, None, None, HOLD_TIME)

    final = data.xpos[bid][:2].copy()
    error = float(np.linalg.norm(final - np.asarray(target_xy, dtype=float)))
    contacts = _object_contacts(model, data, bid)
    return {
        "arm": arm,
        "object": name,
        "target_xy": list(map(float, target_xy)),
        "final_xy": final.tolist(),
        "final_z": float(data.xpos[bid][2]),
        "error": error,
        "carry_drop": plan["carry_drop"],
        "fouls": plan["fouls"],
        "release_ok": plan["release_ok"],
        "transit_clear": plan["transit_clear"],
        "approach": plan["approach"],
        "reaim": None if aim is None else aim["report"],
        "grip_fades": int(fades[0]),
        "contacts": contacts,
        "placed": bool(error <= PLACE_TOL and contacts["table"] > 0
                       and contacts["arm"] == 0),
        "stages": stages,
    }


# ---------------------------------------------------------------------------
# Handover
# ---------------------------------------------------------------------------

# Both arms have to hold the same prop at once, so the search is over where on it
# the receiving arm grips.  Two top-down grippers separate only along their shared
# lateral axis: along the approach they are stacked, and along the opening axis
# the body runs -27 to +68 mm from the tool site, so there is nothing to be gained
# there either.
MEET_OFFSETS = 13        # receiving grasp points sampled along the prop's long axis
MEET_CELLS = 40          # meeting positions tried, nearest the midline first
TRANSFER_TRIES = 24      # transfer cells tested for pickability, best-scoring first
MEET_ARMS = ("left", "right")


def _arm_geoms(model: mujoco.MjModel, arm: str) -> set:
    return {g for g in range(model.ngeom)
            if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                  model.geom_bodyid[g]) or "").startswith(arm + "_")}


def arm_clash(model: mujoco.MjModel, data: mujoco.MjData) -> Tuple[int, float]:
    """Contacts between the two arms, and the deepest interpenetration in metres."""
    left, right = _arm_geoms(model, "left"), _arm_geoms(model, "right")
    count, deepest = 0, 0.0
    for i in range(data.ncon):
        c = data.contact[i]
        pair = {c.geom1, c.geom2}
        if (pair & left) and (pair & right):
            count += 1
            deepest = max(deepest, -float(c.dist))
    return count, deepest


def meeting_pose(model: mujoco.MjModel, data: mujoco.MjData, giver: str, taker: str,
                 bid: int) -> Dict[str, object]:
    """Search for a pose where both arms can hold ``bid`` at once.

    The giving arm's grasp is already fixed -- it is holding the thing -- so what
    is searched is where the prop is presented and where along it the receiving
    arm takes hold.  A candidate has to clear three bars: both arms' IK converges
    on its own grasp, the two arms are not in contact, and the receiving grasp is
    on the prop's own grasp feature rather than off the end of it.

    Returns the best candidate found, or the closest miss with the numbers that
    made it a miss, because "too thin" is a measurement and not a verdict.
    """
    from envs.randomize import CELL, GRID_X, GRID_Y, reach_map, PROPS

    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
    spec = {s.name: s for s in PROPS}[name]
    rmap = reach_map(model, data)
    both = rmap._arm_mask(giver, spec.grasp_z) & rmap._arm_mask(taker, spec.grasp_z)
    ix, iy = np.nonzero(both)
    if not len(ix):
        return {"found": False, "reason": "the arms' masks do not overlap at this height",
                "shared_cells": 0}
    cells = np.stack([ix * CELL + GRID_X[0], iy * CELL + GRID_Y[0]], axis=1)
    cells = cells[np.argsort(np.abs(cells[:, 1]))][:MEET_CELLS]

    feature = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{name}_grasp")
    _, half = _body_bbox(model, bid, [feature] if feature >= 0 else None)
    long_axis = 0 if half[0] >= half[1] else 1
    reach = float(half[long_axis])
    width = 2.0 * float(half[1 - long_axis])

    solvers = {a: IKSolver(model, a) for a in (giver, taker)}
    scratch = mujoco.MjData(model)
    best_miss = {"found": False, "clash": None, "separation": 0.0}
    for xy in cells:
        centre = np.array([xy[0], xy[1], spec.grasp_z])
        for yaw in (0.0, np.pi / 2):
            along = np.array([np.cos(yaw + np.pi / 2), np.sin(yaw + np.pi / 2), 0.0])
            for gi, ti in itertools.product(np.linspace(-reach, reach, MEET_OFFSETS),
                                            repeat=2):
                # Both grasps are free to sit anywhere along the feature, not just
                # the receiving one, so the separation searched runs to the whole
                # length of the feature rather than half of it.  The pick can be
                # planned to take one end if that is what a handover needs.
                offset = ti - gi
                give = grasp_candidates(centre + along * gi, yaw, width)[0]
                take = grasp_candidates(centre + along * ti, yaw, width)[0]
                qg, ig = solvers[giver].solve(data, give[0], top_down_mat(give[1]))
                qt, it = solvers[taker].solve(data, take[0], top_down_mat(take[1]))
                if not (ig["converged"] and it["converged"]):
                    continue
                scratch.qpos[:] = data.qpos
                scratch.qpos[solvers[giver].qadr] = qg
                scratch.qpos[solvers[taker].qadr] = qt
                mujoco.mj_forward(model, scratch)
                clash, deep = arm_clash(model, scratch)
                if clash == 0:
                    return {"found": True, "centre": centre.tolist(), "yaw": float(yaw),
                            "offset": float(offset), "give": give[0].tolist(),
                            "give_yaw": float(give[1]), "take": take[0].tolist(),
                            "take_yaw": float(take[1]), "shared_cells": int(both.sum()),
                            "feature_length": 2.0 * reach}
                if best_miss["clash"] is None or deep < best_miss["deepest"]:
                    best_miss = {"found": False, "clash": clash, "deepest": deep,
                                 "separation": abs(float(offset)),
                                 "centre": centre.tolist()}
                best_miss["widest"] = max(best_miss.get("widest", 0.0), abs(float(offset)))
    best_miss.update({"shared_cells": int(both.sum()), "feature_length": 2.0 * reach,
                      "reason": "no offset along the prop separates the two grippers"})
    return best_miss


def park(model: mujoco.MjModel, data: mujoco.MjData, arm: str,
         duration: float = PARK_TIME) -> None:
    """Send ``arm`` back to the pose the scene's keyframe parks it in.

    A handover has both arms working the same square of table, one after the
    other, and the corridor checks only ever ask about props -- nothing in them
    looks at the other arm.  So the arm that has just let go is moved out of the
    way rather than left standing over the spot the other one has to reach into.
    """
    act, grip = _arm_actuators(model, arm)
    if not model.nkey:
        return
    _ramp(model, data, act, grip, model.key_ctrl[0, act].copy(), OPEN_TORQUE, duration)


def transfer_spot(model: mujoco.MjModel, data: mujoco.MjData, giver: str, taker: str,
                  bid: int, avoid: Sequence[Sequence[float]] = ()
                  ) -> Tuple[Optional[np.ndarray], bool]:
    """A place on the table both arms can reach, clear of everything else.

    ``avoid`` is extra (x, y, radius) circles to stay out of -- the slots already
    filled, and where the remaining props still lie.

    Returns ``(xy, pickable)``.  Reaching a cell at the grasp height is not the
    same as being able to *pick* from it, and the difference is what a handover
    falls down: on seed 0 the mug's spot came out at x = -0.170, which the reach
    map covers at the mug's 782 mm grasp height, and there the receiving arm's
    grasp IK missed by 2.3 mm and its lift clamped to 0 mm of the 30 mm that
    counts as a lift -- so the jaws closed on the mug, held it at 2.97 N, and
    never got it off the table.

    Two things separate the map from the pick, and the test has to respect both.
    The map is indexed by where the *prop* sits, but what has to be reachable is
    where the *tool site* goes, and the aperture opens to one side: the site
    stands the grasp's half-width plus its clearance away from the grasp point --
    10 mm for the flatware, 22 mm for the mug, which is two grid cells.  And a
    pick has to rise, so the pose is wanted at the lift height as well as at the
    grasp.  So each candidate cell is tested at the site positions the pick will
    actually command, at both heights, for both arms, against the same
    converge-and-hold standard the reach map itself is built from.

    ``pickable`` says whether the returned cell passed that test.  When none of
    the candidates does, the best-scoring one is returned anyway with
    ``pickable`` false: that is no worse than not looking, and the caller can say
    so rather than present a doomed spot as a good one.
    """
    from envs.randomize import (CELL, GRID_X, GRID_Y, PROPS, PROP_SPACING,
                                TABLE_MARGIN, reach_map, _table_bounds)

    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
    spec = {s.name: s for s in PROPS}[name]
    rmap = reach_map(model, data)
    both = rmap._arm_mask(giver, spec.grasp_z) & rmap._arm_mask(taker, spec.grasp_z)
    ix, iy = np.nonzero(both)
    if not len(ix):
        return None, False
    xy = np.stack([ix * CELL + GRID_X[0], iy * CELL + GRID_Y[0]], axis=1)

    centre, half = _table_bounds(model)
    limit = half - spec.radius - TABLE_MARGIN
    keep = np.all(np.abs(xy - centre) <= limit, axis=1)
    blocks = [(a[0], a[1], a[2]) for a in avoid]
    for other in prop_bodies(model):
        if other == bid:
            continue
        oname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, other)
        blocks.append((data.xpos[other][0], data.xpos[other][1],
                       {s.name: s.radius for s in PROPS}[oname]))
    for bx, by, br in blocks:
        keep &= np.linalg.norm(xy - np.array([bx, by]), axis=1) >= spec.radius + br + PROP_SPACING
    if not keep.any():
        return None, False
    # Furthest inside the shared region, not merely nearest the midline.  The
    # midline runs the whole depth of the table and its far end is the edge of
    # what either arm can do: picking y = 0 alone put the spoon down at x = -0.17,
    # where the receiving arm could reach the cell but not make the grasp.
    from envs.randomize import _morph
    depth = np.zeros_like(both, dtype=float)
    eroded = both.copy()
    for step in range(1, 12):
        eroded = _morph(eroded, CELL, False)
        if not eroded.any():
            break
        depth[eroded] = step * CELL
    inside = depth[ix[keep], iy[keep]]
    xy = xy[keep]
    order = np.lexsort((np.abs(xy[:, 1]), -inside))

    # Where the grasp point sits relative to the prop's own origin, taken from the
    # prop as it is held now: it keeps that orientation through the release, and
    # the goal the place aims at is the origin.
    grasp_pt, grasp_yaw, ginfo = grasp_pose(model, data, name)
    lead = grasp_pt[:2] - data.xpos[bid][:2]
    heights = (spec.grasp_z, spec.grasp_z + LIFT_THRESHOLD)
    solvers = {a: IKSolver(model, a) for a in (taker, giver)}

    def holdable_at(arm: str, site: np.ndarray, yaw: float, z: float) -> bool:
        q, info = solvers[arm].solve(data, np.array([site[0], site[1], z]),
                                    top_down_mat(yaw))
        return bool(info["converged"] and solvers[arm].holdable(q))

    yaws = [grasp_yaw]
    if ginfo["yaw_free"]:
        yaws += [grasp_yaw + np.pi * k / GRASP_YAW_BINS
                 for k in range(1, GRASP_YAW_BINS)]

    def pickable_at(cell: np.ndarray) -> bool:
        at = np.array([cell[0] + lead[0], cell[1] + lead[1], spec.grasp_z])
        for y0 in yaws:
            for site, yaw in grasp_candidates(at, y0, ginfo["grasp_width"]):
                if all(holdable_at(arm, site, yaw, z)
                       for arm in (taker, giver) for z in heights):
                    return True
        return False

    for k in order[:TRANSFER_TRIES]:
        if pickable_at(xy[k]):
            return xy[k], True
    return xy[order[0]], False


def handover(model: mujoco.MjModel, data: mujoco.MjData, from_arm: str,
             to_arm: str, avoid: Sequence[Sequence[float]] = ()) -> Dict[str, object]:
    """Move whatever ``from_arm`` holds into ``to_arm``'s hands.

    Not the in-air kind, and :func:`meeting_pose` is what settles that.  The two
    arms' reach maps overlap generously -- 169 to 193 cells, a band 100 to 120 mm
    wide across y -- so reaching a common point is not the problem.  The gripper
    is: its collision hull spans 52.0 mm across the lateral axis (-27.8 to +24.2
    mm of the tool site), so two top-down grippers held at the same yaw must
    stand at least that far apart along that axis to clear each other.  Turning
    one of them end for end buys little -- the hull is nearly symmetric there, so
    opposed yaws still need 48.4 mm -- and the other two axes offer nothing:
    along the approach the grippers are stacked, and along the opening axis the
    body runs -27 to +68 mm of the tool site (at the keyframe's 0.6 rad of jaw;
    the moving jaw swings that to +111 mm at its stop).  The longest grasp
    feature in this scene is the mug's 36 mm, and with *both* grasps free to sit
    at opposite ends of it the arms still interpenetrate 14.2 mm across 36
    contacts.  The fork's 32 mm feature leaves 18.1 mm of overlap, the plate's
    34 mm leaves 16.1.

    Searching wider does not find a pose either, only a shallower floor: over
    both arms' yaws independently -- opposed and perpendicular included, which
    the search below does not try -- and with the two grasps stacked up the mug's
    64 mm barrel, the least interpenetration is 14.8 mm for the fork and spoon,
    27.5 mm for the mug and 30.0 mm for the plate.  The one clear pose that turns
    up is not a grasp: it sits both jaw lines tangent to the mug's barrel, where
    the chord between them is 0.0 mm and there is nothing to close on.  There is
    no pose to implement an in-air handover against, so none is implemented.

    What is left is the table: the giving arm sets the prop down where both arms
    can reach it, and the receiving arm picks it up again.
    """
    bid = held_object(model, data, from_arm)
    if bid is None:
        return {"from": from_arm, "to": to_arm, "object": None, "handed": False,
                "reason": "nothing held"}
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
    meeting = meeting_pose(model, data, from_arm, to_arm, bid)

    spot, pickable = transfer_spot(model, data, from_arm, to_arm, bid, avoid)
    if spot is None:
        return {"from": from_arm, "to": to_arm, "object": name, "handed": False,
                "reason": "no spot on the table both arms can reach is clear",
                "meeting": meeting}

    put = place(model, data, from_arm, spot)
    if put["placed"]:
        park(model, data, from_arm)
    got = pick(model, data, to_arm, name) if put["placed"] else None
    return {
        "from": from_arm, "to": to_arm, "object": name,
        "spot": spot.tolist(),
        "spot_pickable": bool(pickable),
        "in_air": bool(meeting["found"]),
        "meeting": meeting,
        "put_down": put,
        "picked_up": got,
        "handed": bool(put["placed"] and got is not None and got["lifted"]),
        "reason": None if put["placed"] else "could not set it down at the transfer spot",
    }
