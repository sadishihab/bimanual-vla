"""Scripted manipulation primitives for the bimanual table scene.

The only primitive so far is :func:`pick`: a top-down grasp built from four
waypoints -- above the object, descend, close, lift -- with the arm driven by
its position actuators and the simulation stepped throughout.  Joint targets for
each waypoint come from :mod:`control.ik`.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import mujoco
import numpy as np

from control.ik import ARM_JOINTS, IKSolver, top_down_mat

# Gripper actuator: the jaw closes at the low end of ctrlrange and opens toward
# the high end (verified against the SO-101 model).
GRIPPER_OPEN = 1.10
GRIPPER_CLOSED = -0.15

APPROACH_HEIGHT = 0.10   # m above the grasp point for the standoff waypoint
LIFT_HEIGHT = 0.10       # m above the grasp point for the final waypoint
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
APPROACH_TIME = 1.5
DESCEND_TIME = 1.0
CLOSE_TIME = 0.8
LIFT_TIME = 1.5
HOLD_TIME = 0.5


# ---------------------------------------------------------------------------
# Scene queries
# ---------------------------------------------------------------------------


def _body_geoms(model: mujoco.MjModel, bid: int) -> List[int]:
    return [g for g in range(model.ngeom) if model.geom_bodyid[g] == bid]


def _body_bbox(model: mujoco.MjModel, bid: int) -> Tuple[np.ndarray, np.ndarray]:
    """Axis-aligned bounding box of a body's geoms, in the body frame.

    Returns ``(centre, half_extent)``.  Each geom's own local AABB corners are
    rotated into the body frame, which handles the offset/rotated primitives the
    props are built from.
    """
    corners = []
    for gid in _body_geoms(model, bid):
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

    centre, half = _body_bbox(model, bid)
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
    gripper can be driven without disturbing the arm and vice versa.
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
    solver = IKSolver(model, arm)

    # Let the props drop the couple of millimetres they spawn above the table.
    _ramp(model, data, act, grip, None, None, SETTLE_TIME)
    rest_z = float(data.xpos[bid][2])

    centre, yaw0, geom = grasp_pose(model, data, object_name)

    # Pick the symmetric variant the arm can actually reach, preferring one that
    # also works at the standoff height so approach and lift use the same grasp.
    best = None
    for pos, candidate in grasp_candidates(centre, yaw0, geom["grasp_width"]):
        _, low = solver.solve(data, pos, top_down_mat(candidate))
        _, high = solver.solve(data, pos + np.array([0.0, 0.0, APPROACH_HEIGHT]),
                               top_down_mat(candidate))
        score = (not low["converged"], not high["converged"],
                 low["pos_err"] + low["rot_err"])
        if best is None or score < best[0]:
            best = (score, pos, candidate, low, high)
    _, target, yaw, low, high = best
    geom["chosen_yaw"] = yaw
    geom["jaw_offset"] = float(np.linalg.norm(target[:2] - centre[:2]))
    geom["grasp_ik"] = {"at_grasp": low["pos_err"], "at_standoff": high["pos_err"]}

    waypoints = (
        ("approach", target + np.array([0.0, 0.0, APPROACH_HEIGHT]), GRIPPER_OPEN, APPROACH_TIME),
        ("descend", target, GRIPPER_OPEN, DESCEND_TIME),
        ("close", None, GRIPPER_CLOSED, CLOSE_TIME),
        ("lift", target + np.array([0.0, 0.0, LIFT_HEIGHT]), GRIPPER_CLOSED, LIFT_TIME),
    )

    stages: List[Dict[str, object]] = []
    for name, pos, grip_target, duration in waypoints:
        record: Dict[str, object] = {"stage": name}
        q_target = None
        if pos is not None:
            q_target, ik_info = solver.solve(data, pos, top_down_mat(yaw))
            record["target_pos"] = np.asarray(pos).tolist()
            record["ik"] = ik_info
        _ramp(model, data, act, grip, q_target, grip_target, duration)
        record["site_pos"] = data.site_xpos[solver.site].tolist()
        if pos is not None:
            record["site_err"] = float(np.linalg.norm(data.site_xpos[solver.site] - pos))
        record["object_z"] = float(data.xpos[bid][2])
        record["contacts"] = _object_contacts(model, data, bid)
        stages.append(record)

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
        "final_z": final_z,
        "rise": rise,
        "contacts": contacts,
        "lifted": lifted,
        "stages": stages,
    }
