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

from control.gripper import GRIP_TORQUE, OPEN_TORQUE
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
# The transit across the table is flown over the props, not through them, and is
# allowed a looser wrist than the grasp: it is a move, not a grasp, and demanding
# a strict top-down tool the whole way is what leaves it nothing to reach with
# where the standoff has had to clamp low.
TRANSIT_CLEARANCE = 0.015  # m above the tallest prop for the cross-table leg
TRANSIT_ROT_TOL = 0.50     # rad of tool tilt tolerated while in transit
CARTESIAN_STEPS = 12     # sub-waypoints along a straight-line gripper path
MIN_FINGER_Z = 0.010     # m above the table top, so the jaws clear the surface
# The gripper site sits on the *fixed* finger, not in the middle of the jaw
# aperture: opening the gripper swings only the moving jaw, so the aperture spans
# [site, site + opening] along the +z site axis.  Centring the site on the object
# therefore drops the fixed finger straight onto it.  The grasp point is offset
# back along the opening axis by half the object's width plus this clearance, so
# the aperture straddles the object instead.
JAW_CLEARANCE = 0.004    # m of daylight between the fixed finger and the object edge
LIFT_THRESHOLD = 0.03    # m of net rise that counts as a successful lift

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


def _table_top_z(model: mujoco.MjModel) -> float:
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table_top")
    bid = model.geom_bodyid[gid]
    return float(model.body_pos[bid][2] + model.geom_pos[gid][2] + model.geom_size[gid][2])


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

    # A prop may declare its grasp feature as a geom named <prop>_grasp.  Without
    # that, the merged bounding box of a fork is dominated by its head and the
    # grasp lands on the wrong part at the wrong width.
    feature = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{object_name}_grasp")
    centre, half = _body_bbox(model, bid, [feature] if feature >= 0 else None)
    rot = data.xmat[bid].reshape(3, 3)
    world_centre = data.xpos[bid] + rot @ centre

    # Grip across whichever body axis is narrower in the horizontal plane.
    narrow = 0 if half[0] <= half[1] else 1
    axis = rot[:, narrow]
    yaw = float(np.arctan2(axis[1], axis[0]))
    width = 2.0 * float(half[narrow])

    # Vertical extent of the object in the world, from the rotated box corners.
    span = float(np.abs(rot @ np.diag(half)).sum(axis=1)[2])
    bottom, top = world_centre[2] - span, world_centre[2] + span
    grasp_z = max(world_centre[2], _table_top_z(model) + MIN_FINGER_Z)
    grasp_z = min(grasp_z, top)

    pos = np.array([world_centre[0], world_centre[1], grasp_z])
    info = {
        "grasp_width": width,
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

    best = None
    for pos, candidate in grasp_candidates(centre, yaw0, geom["grasp_width"]):
        q_low, low = solver.solve(data, pos, top_down_mat(candidate))
        # Seeded the way each motion is: the approach solves the standoff cold
        # from wherever the arm is, the lift solves warm from the grasp pose.
        stand, s_info = reachable_above(solver, data, pos, candidate, APPROACH_HEIGHT,
                                        floor=MIN_STANDOFF)
        lift, l_info = reachable_above(solver, data, pos, candidate, LIFT_HEIGHT,
                                       seed=q_low)
        fouls = path_fouls(model, data, solver, arm, stand, pos, candidate, props)
        score = (not low["converged"], fouls, -l_info["height"],
                 low["pos_err"] + low["rot_err"])
        if best is None or score < best[0]:
            best = (score, pos, candidate, low, stand, s_info, lift, l_info, fouls)

    _, target, yaw, low, standoff, s_info, lifted_to, l_info, fouls = best
    geom["chosen_yaw"] = yaw
    geom["jaw_offset"] = float(np.linalg.norm(target[:2] - centre[:2]))
    geom["grasp_ik"] = {"at_grasp": low["pos_err"], "converged": low["converged"]}
    geom["standoff"] = {k: s_info[k] for k in ("height", "clamped", "floor_ok")}
    geom["lift"] = {k: l_info[k] for k in ("height", "clamped", "floor_ok")}
    # A standoff clamped below the object's top starts the descent alongside the
    # prop rather than above it.  With the jaws run back to their stop that is
    # measured clear, but it is worth knowing when it happens.
    geom["standoff_below_object"] = bool(target[2] + s_info["height"] < geom["object_top_z"])
    geom["descent_fouls"] = int(fouls)

    # Route the cross-table transit over everything loose on it, then come down
    # onto the standoff.  Flying straight at the standoff is what shoved six props
    # off their spot before the jaws ever touched them -- 167 mm on one of them --
    # because a standoff clamped low turns that line nearly flat.  The via point is
    # directly over the standoff, so the last leg is vertical and shares the
    # descent's corridor rather than cutting a new one across the table.
    ceiling = max(float(data.geom_xpos[g][2] + model.geom_rbound[g])
                  for b in props for g in _body_geoms(model, b))
    hover = np.array([standoff[0], standoff[1],
                      max(standoff[2], ceiling + TRANSIT_CLEARANCE)])
    hover, h_info = reachable_toward(solver, data, hover, standoff,
                                     top_down_mat(yaw), rot_tol=TRANSIT_ROT_TOL)
    geom["hover_height"] = float(hover[2] - target[2])
    geom["transit_clear"] = bool(hover[2] >= ceiling + TRANSIT_CLEARANCE)
    return {"solver": solver, "centre": centre, "target": target, "yaw": yaw,
            "standoff": standoff, "hover": hover, "lifted_to": lifted_to, "geometry": geom,
            "grasp_ok": bool(low["converged"]), "lift_height": float(l_info["height"]),
            "fouls": int(fouls)}


def best_arm(model: mujoco.MjModel, data: mujoco.MjData,
             object_name: str) -> Tuple[str, Dict[str, object]]:
    """Which arm should pick ``object_name``: the one that can grasp it and lift highest.

    Reach is not symmetric -- the props are placed by a per-arm reach map, and a
    prop out at one side is often graspable by one arm only -- so the arm cannot
    be fixed per prop across seeds.  Ordered by whether the grasp converges, then
    by a descent corridor clear of the prop, then by lift height, then by grasp
    residual.  Residual alone is useless as a tie-break here: on the plate the two
    arms differ by 0.01 mm and one of them drives straight into it.
    """
    best = None
    for arm in ("left", "right"):
        plan = plan_grasp(model, data, arm, object_name)
        score = (not plan["grasp_ok"], plan["fouls"], -plan["lift_height"],
                 plan["geometry"]["grasp_ik"]["at_grasp"])
        if best is None or score < best[0]:
            best = (score, arm, plan)
    return best[1], best[2]


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
    lifted_to, geom = plan["lifted_to"], plan["geometry"]
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
               hover, yaw, TRANSIT_TIME, clamp=True, rot_tol=TRANSIT_ROT_TOL,
               start_mat=data.site_xmat[solver.site].reshape(3, 3).copy())
    ik = _move_line(model, data, act, grip, solver, hover, standoff, yaw,
                    APPROACH_TIME, clamp=True, rot_tol=TRANSIT_ROT_TOL)
    record("approach", ik, standoff)

    # Straight down onto the grasp point, jaws held open and clear of the object.
    ik = _move_line(model, data, act, grip, solver, standoff, target, yaw, DESCEND_TIME)
    record("descend", ik, target)

    # Swing the command over to the squeeze, then hold it while the jaw sweeps in.
    _ramp(model, data, act, grip, None, GRIP_TORQUE, CLOSE_RAMP_TIME)
    _ramp(model, data, act, grip, None, None, CLOSE_TIME)
    record("close", None, None)

    # Straight back up, so a successful grasp is not levered against the table.
    # The squeeze is sampled at every sub-step: whether the torque command holds
    # its force through the motion is the whole question this primitive failed on.
    trace: List[Dict[str, object]] = []

    def sample(k: int) -> None:
        f = grip_force(model, data, bid, arm)
        trace.append({"step": k,
                      "height": float(data.site_xpos[solver.site][2] - target[2]),
                      "object_z": float(data.xpos[bid][2]),
                      "jaw_gap": jaw_gap(model, data, arm, solver.site),
                      "grip_force": f})

    ik = _move_line(model, data, act, grip, solver, target, lifted_to, yaw, LIFT_TIME,
                    probe=sample)
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
