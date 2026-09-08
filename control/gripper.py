"""Force control for the SO-101 gripper.

The vendored SO-101 model drives every joint, the gripper included, with a
``<position>`` actuator.  That is right for the arm and wrong for the jaws.  A
position servo produces ``kp * (ctrl - q) - kv * qdot``, so its output goes to
zero exactly when it arrives at the commanded angle -- and a gripper closed on
an object arrives there by pressing the soft contact until the object has been
penetrated by the whole of the position error.  Grip force therefore decays to
nothing as the servo settles, which is what emptied the jaws the moment the arm
started to move: measured 49.96 -> 28.29 -> 21.09 -> 0.00 N over a three-step
lift.  Commanding *past* the object keeps the error non-zero and so keeps the
force up, but only by driving the jaws through it (6 mm into a 12 mm handle).

The fix is to command the thing we actually care about.  A torque actuator
outputs ``gear * ctrl`` regardless of where the joint has ended up, so a closing
command holds its force indefinitely, and the penetration settles wherever the
contact balances that torque instead of wherever the servo's error runs out.

Only the gripper changes.  The five pose-controlling joints of each arm stay on
position control, which is what the IK and the reach map assume.

Commands are torques at the gripper joint, in newton-metres, negative closing:

    OPEN_TORQUE   holds the jaw open against its upper stop
    GRIP_TORQUE   the squeeze held through the close, the lift and the hold
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import mujoco
import numpy as np

# The jaw closes toward the low end of the joint range, so a closing torque is
# negative.  Both are well inside the servo's +/-3.35 N.m limit: the opening
# command only has to beat 0.6 N.m.s/rad of joint damping (it decides how fast
# the jaw retracts, not where it ends up -- that is the joint stop), and the
# squeeze only has to beat the weight of a 30 g fork many times over.
OPEN_TORQUE = 1.0     # N.m
GRIP_TORQUE = -0.8    # N.m

_ARMS = ("left", "right")


def gripper_actuators(model: mujoco.MjModel) -> List[int]:
    """Actuator ids of every arm's gripper, in arm order."""
    ids = []
    for arm in _ARMS:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{arm}_gripper")
        if aid < 0:
            raise ValueError(f"scene has no actuator {arm}_gripper")
        ids.append(aid)
    return ids


def force_control_gripper(model: mujoco.MjModel) -> Dict[str, object]:
    """Convert every gripper actuator from a position servo to a torque source.

    Rewrites the compiled actuator in place rather than the XML, because the arms
    reach the scene through ``<attach>``: the submodel's actuators are copied in
    wholesale and there is no XML spelling for replacing one of them.  The values
    written are exactly what ``<motor gear="1">`` compiles to -- unit fixed gain,
    no bias term -- so the result is a direct torque actuator and not an
    approximation of one.

    ``ctrlrange`` becomes the actuator's ``forcerange``, since ctrl is now a
    torque.  Any keyframe's gripper ctrl is rewritten to :data:`OPEN_TORQUE` for
    the same reason: the stored 0.6 meant radians and would otherwise be read as
    0.6 N.m of squeeze on reset.
    """
    ids = gripper_actuators(model)
    for aid in ids:
        model.actuator_gaintype[aid] = mujoco.mjtGain.mjGAIN_FIXED
        model.actuator_gainprm[aid, :] = 0.0
        model.actuator_gainprm[aid, 0] = 1.0
        model.actuator_biastype[aid] = mujoco.mjtBias.mjBIAS_NONE
        model.actuator_biasprm[aid, :] = 0.0
        model.actuator_ctrlrange[aid] = model.actuator_forcerange[aid]
        model.actuator_ctrllimited[aid] = 1
        model.key_ctrl[:, aid] = OPEN_TORQUE
    return {
        "actuators": [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in ids],
        "ctrlrange": np.asarray(model.actuator_ctrlrange[ids]).tolist(),
        "open_torque": OPEN_TORQUE,
        "grip_torque": GRIP_TORQUE,
    }


# ---------------------------------------------------------------------------
# Jaw pads
# ---------------------------------------------------------------------------

# MuJoCo collides a mesh geom as its convex hull, and both SO-101 jaws are
# concave parts, so the hull is not the finger.  On the fixed finger the hull
# bridges from the wrist mount to the fingertip in one 1428 mm^2 facet tilted
# 21.7 degrees off the opening axis: of its 606 facets that face the aperture,
# not one is within 5 degrees of parallel.  Grasping against that ramp touches
# only the object's top edge and drives it down -- measured 2.07 N onto the
# table against a 0.30 N fork -- at any grasp height, because the ramp is fixed
# to the tool frame and moves with it.
#
# The fix is to give each jaw an explicit pad to grasp against.  The real inner
# faces under the hull are flat and parallel: fitting a plane through the mesh
# vertices at each face gives 0.23 and 0.24 degrees off the opening axis with a
# 0.33 mm residual, 11.5 mm wide and about 10 mm deep, 15.80 mm apart.  So the
# pads are not invented geometry -- they are boxes laid on the faces that are
# already there, and they are placed from that measurement rather than typed in.
PAD_REFERENCE_ANGLE = 0.0   # rad; the jaw angle the two faces are parallel at
PAD_THICKNESS = 0.0015      # m, built backwards so the pad face lies on the mesh face
PAD_FACE_TOL = 0.0005       # m; how close to the extreme a vertex counts as on the face
PAD_DEPTH_WINDOW = 0.015    # m either side of the site, along the approach axis
PAD_LATERAL_WINDOW = 0.025  # m either side of the site

_JAWS = (("gripper", +1), ("moving_jaw_so101_v1", -1))


def _site_frame(data: mujoco.MjData, site: int) -> Tuple[np.ndarray, np.ndarray]:
    rot = data.site_xmat[site].reshape(3, 3)
    return data.site_xpos[site].copy(), rot.copy()


def measure_jaw_face(model: mujoco.MjModel, data: mujoco.MjData, arm: str, body: str,
                     sign: int, site: int) -> Dict[str, object]:
    """The flat inner face of one jaw, in the gripper site frame.

    ``sign`` is +1 for the fixed finger, whose face bounds the aperture from
    below on the opening axis, and -1 for the moving jaw, which bounds it from
    above.  ``data`` must already hold the reference configuration; the caller
    sets the jaw angle, since the moving jaw's face is only parallel to the fixed
    one at :data:`PAD_REFERENCE_ANGLE`.

    Returns the face's position on the opening axis, its depth and lateral
    extents, and how far the fitted plane is off parallel -- the last as a check
    that the face really is flat, not as something the caller has to act on.
    """
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
    if bid < 0:
        raise ValueError(f"scene has no body {body}")
    origin, rot = _site_frame(data, site)
    approach, lateral, opening = rot[:, 0], rot[:, 1], rot[:, 2]

    pts = []
    for gid in range(model.ngeom):
        if model.geom_bodyid[gid] != bid or model.geom_group[gid] != 3:
            continue
        mesh = model.geom_dataid[gid]
        if mesh < 0:
            continue
        a, n = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        verts = model.mesh_vert[a:a + n].astype(float)
        pts.append(data.geom_xpos[gid]
                   + verts @ data.geom_xmat[gid].reshape(3, 3).T - origin)
    if not pts:
        raise ValueError(f"{body} has no collision mesh to measure")

    local = np.vstack(pts)
    depth, lat, opn = local @ approach, local @ lateral, local @ opening
    keep = (np.abs(depth) < PAD_DEPTH_WINDOW) & (np.abs(lat) < PAD_LATERAL_WINDOW)
    local, depth, lat, opn = local[keep], depth[keep], lat[keep], opn[keep]

    face_at = float(opn.max() if sign > 0 else opn.min())
    on_face = np.abs(opn - face_at) < PAD_FACE_TOL
    if on_face.sum() < 3:
        raise RuntimeError(f"{body} has no flat inner face within {PAD_FACE_TOL * 1000:.1f} mm")

    centred = local[on_face] - local[on_face].mean(axis=0)
    normal = np.linalg.svd(centred)[2][-1]
    return {
        "face": face_at,
        "depth": (float(depth[on_face].min()), float(depth[on_face].max())),
        "lateral": (float(lat[on_face].min()), float(lat[on_face].max())),
        "tilt": float(np.degrees(np.arccos(min(1.0, abs(normal @ opening))))),
        "residual": float(np.abs(centred @ normal).max()),
        "vertices": int(on_face.sum()),
    }


def jaw_pad_poses(model: mujoco.MjModel, arm: str) -> Dict[str, Dict[str, object]]:
    """Pad box poses for one arm's two jaws, in each jaw body's own frame.

    Measured, not tabulated: each pad's face is laid on the mesh face found by
    :func:`measure_jaw_face`, with the box built backwards from it so it never
    stands proud of the surface the CAD actually has.  The box axes are the site
    axes -- x approach, y lateral, z opening -- so the pad faces are parallel to
    each other by construction at :data:`PAD_REFERENCE_ANGLE`.
    """
    data = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{arm}_gripper")
    if jid < 0:
        raise ValueError(f"scene has no joint {arm}_gripper")
    data.qpos[model.jnt_qposadr[jid]] = PAD_REFERENCE_ANGLE
    mujoco.mj_kinematics(model, data)

    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{arm}_gripperframe")
    if site < 0:
        raise ValueError(f"scene has no site {arm}_gripperframe")
    origin, rot = _site_frame(data, site)

    out = {}
    for suffix, sign in _JAWS:
        body = f"{arm}_{suffix}"
        face = measure_jaw_face(model, data, arm, body, sign, site)
        (d_lo, d_hi), (l_lo, l_hi) = face["depth"], face["lateral"]
        half = np.array([(d_hi - d_lo) / 2.0, (l_hi - l_lo) / 2.0, PAD_THICKNESS / 2.0])
        # Backwards from the face: away from the aperture, so the pad's own face
        # sits exactly where the mesh's does.
        centre_site = np.array([(d_lo + d_hi) / 2.0, (l_lo + l_hi) / 2.0,
                                face["face"] - sign * PAD_THICKNESS / 2.0])
        world = origin + rot @ centre_site

        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
        body_rot = data.xmat[bid].reshape(3, 3)
        local_mat = body_rot.T @ rot
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, local_mat.ravel())
        out[body] = {
            "pos": (body_rot.T @ (world - data.xpos[bid])).tolist(),
            "quat": quat.tolist(),
            "size": half.tolist(),
            "measured": face,
        }
    return out


def add_jaw_pads(spec: mujoco.MjSpec, probe: mujoco.MjModel) -> Dict[str, object]:
    """Add a flat pad geom to the inner face of every jaw in ``spec``.

    ``probe`` is the same scene already compiled once, which is where the pad
    placement is measured from.  The mesh geoms are left alone so the arms still
    render as themselves; :func:`isolate_grasp_contacts` is what stops those
    meshes deciding a grasp.

    Density is zero and the SO-101 bodies carry explicit ``<inertial>`` elements,
    so the pads add no mass and the arms' dynamics -- and the reach map keyed on
    them -- are unchanged.
    """
    added = {}
    for arm in _ARMS:
        for body, pad in jaw_pad_poses(probe, arm).items():
            geom = spec.body(body).add_geom()
            geom.name = f"{body}_pad"
            geom.type = mujoco.mjtGeom.mjGEOM_BOX
            geom.size = pad["size"]
            geom.pos = pad["pos"]
            geom.quat = pad["quat"]
            geom.group = 3
            geom.density = 0.0
            added[geom.name] = pad
    return added


def isolate_grasp_contacts(model: mujoco.MjModel) -> Dict[str, List[str]]:
    """Make the props collide with the jaw pads but not with the jaw meshes.

    Two contype/conaffinity bits: bit 1 for structural contacts and bit 2 for
    grasp contacts.  Everything that collides at all gets both, so no existing
    pair is lost; the props then drop bit 1 and the jaws' collision meshes drop
    bit 2, which leaves exactly one pair excluded -- prop against jaw mesh.  The
    pads keep both, so they still stop on the table, and the jaw meshes keep bit
    1, so an arm driven into the table still hits it.

    Props are found by their free joint rather than by name, and geoms that were
    already non-colliding (the plate's decorative well) are left alone.
    """
    for gid in range(model.ngeom):
        if model.geom_contype[gid] or model.geom_conaffinity[gid]:
            model.geom_contype[gid] = 3
            model.geom_conaffinity[gid] = 3

    free = {model.jnt_bodyid[j] for j in range(model.njnt)
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE}
    props, meshes = [], []
    for gid in range(model.ngeom):
        if not (model.geom_contype[gid] or model.geom_conaffinity[gid]):
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or f"geom{gid}"
        if model.geom_bodyid[gid] in free:
            model.geom_contype[gid] = model.geom_conaffinity[gid] = 2
            props.append(name)
    for arm in _ARMS:
        for suffix, _ in _JAWS:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{arm}_{suffix}")
            for gid in range(model.ngeom):
                if model.geom_bodyid[gid] == bid and model.geom_dataid[gid] >= 0 \
                        and (model.geom_contype[gid] or model.geom_conaffinity[gid]):
                    model.geom_contype[gid] = model.geom_conaffinity[gid] = 1
                    meshes.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
                                  or f"geom{gid}")
    return {"props": props, "jaw_meshes": meshes}
