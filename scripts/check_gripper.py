#!/usr/bin/env python3
"""Report the collision geometry of one arm's jaws, as MuJoCo actually sees it.

MuJoCo collides a mesh geom as its *convex hull*.  The SO-101 jaws are concave
parts, so the hull is not the shape the CAD draws, and a grasp is decided by the
hull rather than by the visible finger.  This prints, for each jaw, the surfaces
that face the grasp aperture -- hull facets for the mesh geoms, box faces for the
pads -- with how far each is tilted off parallel to the jaw opening axis.

A parallel-jaw pinch needs surfaces near 0 degrees.  A tilted one meeting the
flat side of an object touches only along the object's top edge, and its normal
then has a vertical component that drives the object down rather than squeezing
it, so the grasp is a wedge no matter what height it is made at.

The check passes when each jaw presents a near-parallel surface that the props
can actually reach: a surface on a mesh geom excluded from grasp contacts does
not count, since nothing grasped will ever touch it.

The moving jaw swings on an arc, so the two faces are parallel at one jaw angle
only.  ``--jaw`` picks the angle; the default is the one the pads were laid out
at, and the sweep at the end shows what the arc costs away from it.

No seed and no physics: this is a property of the model, not of a scene.
"""

import argparse
import pathlib
import sys

import mujoco
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from control.gripper import PAD_REFERENCE_ANGLE  # noqa: E402
from envs.scene import SCENE, load_scene  # noqa: E402

# The grasp region: how far from the gripper site a surface must lie to be able
# to touch a grasped object at all.
DEPTH_WINDOW = (-0.015, 0.012)   # m along the approach axis, +ve toward the fingertips
LATERAL_WINDOW = 0.020           # m either side of the site
FACING = 0.30                    # min |normal . opening| to count as facing the aperture
FLAT = 5.0                       # deg off parallel that still counts as a flat pad
GRASP_BIT = 2                    # the contype/conaffinity bit the props collide on

JAWS = (("fixed finger", "{arm}_gripper", +1),
        ("moving jaw", "{arm}_moving_jaw_so101_v1", -1))


def _box_faces(model, gid, data):
    """The six faces of a box geom as (normal, corners) in world coordinates."""
    rot = data.geom_xmat[gid].reshape(3, 3)
    half = model.geom_size[gid][:3]
    for k in range(3):
        u, v = (k + 1) % 3, (k + 2) % 3
        for s in (-1.0, 1.0):
            corners = []
            for su in (-1.0, 1.0):
                for sv in (-1.0, 1.0):
                    offset = np.zeros(3)
                    offset[k], offset[u], offset[v] = s * half[k], su * half[u], sv * half[v]
                    corners.append(data.geom_xpos[gid] + rot @ offset)
            # wind the quad so the area sum below is not self-intersecting
            corners = [corners[0], corners[1], corners[3], corners[2]]
            yield s * rot[:, k], np.asarray(corners)


def _hull_facets(model, gid, data):
    """The convex-hull facets of a mesh geom as (normal, corners) in world coordinates."""
    mesh = model.geom_dataid[gid]
    rot = data.geom_xmat[gid].reshape(3, 3)
    va, nv = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
    verts = data.geom_xpos[gid] + model.mesh_vert[va:va + nv].astype(float) @ rot.T
    pa, pn = model.mesh_polyadr[mesh], model.mesh_polynum[mesh]
    for k in range(pn):
        a0 = model.mesh_polyvertadr[pa + k]
        idx = model.mesh_polyvert[a0:a0 + model.mesh_polyvertnum[pa + k]]
        yield model.mesh_polynormal[pa + k] @ rot.T, verts[idx]


def jaw_surfaces(model, data, body, sign, site):
    """Aperture-facing surfaces of ``body``, as dicts, with those the props can reach."""
    origin = data.site_xpos[site]
    rot = data.site_xmat[site].reshape(3, 3)
    approach, lateral, opening = rot[:, 0], rot[:, 1], rot[:, 2]

    out = []
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
    for gid in range(model.ngeom):
        if model.geom_bodyid[gid] != bid or model.geom_group[gid] != 3:
            continue
        is_mesh = model.geom_dataid[gid] >= 0
        source = _hull_facets(model, gid, data) if is_mesh else _box_faces(model, gid, data)
        graspable = bool(model.geom_contype[gid] & GRASP_BIT
                         or model.geom_conaffinity[gid] & GRASP_BIT)
        for normal, corners in source:
            if sign * (normal @ opening) < FACING:
                continue
            local = corners - origin
            depth = local @ approach
            if depth.max() < DEPTH_WINDOW[0] or depth.min() > DEPTH_WINDOW[1]:
                continue
            if np.abs(local @ lateral).min() > LATERAL_WINDOW:
                continue
            area = sum(0.5 * np.linalg.norm(np.cross(corners[i] - corners[0],
                                                     corners[i + 1] - corners[0]))
                       for i in range(1, len(corners) - 1))
            out.append({
                "geom": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or f"geom{gid}",
                "graspable": graspable,
                "depth": (float(depth.min()), float(depth.max())),
                "tilt": float(np.degrees(np.arccos(min(1.0, abs(normal @ opening))))),
                "area": float(area),
            })
    return out


def set_jaw(model, data, arm, angle):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{arm}_gripper")
    data.qpos[model.jnt_qposadr[jid]] = angle
    mujoco.mj_kinematics(model, data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("left", "right"), default="left")
    parser.add_argument("--jaw", type=float, default=PAD_REFERENCE_ANGLE,
                        help="gripper joint angle to measure at, in radians")
    parser.add_argument("--scene", type=pathlib.Path, default=SCENE)
    args = parser.parse_args()

    model = load_scene(args.scene)
    data = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    set_jaw(model, data, args.arm, args.jaw)
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{args.arm}_gripperframe")

    ok = True
    print(f"{args.arm} gripper at jaw angle {args.jaw:+.3f} rad:"
          f" surfaces facing the grasp aperture")
    for label, template, sign in JAWS:
        surfaces = jaw_surfaces(model, data, template.format(arm=args.arm), sign, site)
        if not surfaces:
            print(f"  {label:13s} nothing faces the aperture near the site")
            ok = False
            continue
        reachable = [s for s in surfaces if s["graspable"]]
        flat = [s for s in reachable if s["tilt"] < FLAT]
        tilts = np.array([s["tilt"] for s in surfaces])
        print(f"  {label:13s} {len(surfaces):4d} surfaces"
              f"   tilt off parallel: min {tilts.min():5.1f} deg,"
              f" median {np.median(tilts):5.1f} deg")
        widest = max(surfaces, key=lambda s: s["area"])
        print(f"  {'':13s} largest {widest['area'] * 1e6:7.1f} mm^2 on {widest['geom']}"
              f" spans depth {widest['depth'][0] * 1000:+7.2f} .."
              f" {widest['depth'][1] * 1000:+7.2f} mm at {widest['tilt']:5.1f} deg"
              f"{'' if widest['graspable'] else '  (excluded from grasp contacts)'}")
        if flat:
            best = max(flat, key=lambda s: s["area"])
            print(f"  {'':13s} graspable and within {FLAT:.0f} deg:"
                  f" {len(flat)}, largest {best['area'] * 1e6:7.1f} mm^2 on"
                  f" {best['geom']} at {best['tilt']:5.2f} deg")
        else:
            print(f"  {'':13s} graspable and within {FLAT:.0f} deg: NONE")
            ok = False

    print(f"\n  the moving jaw swings on an arc, so the faces are parallel at one angle only:")
    print("    jaw angle   pad pair off antiparallel")
    pads = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                              f"{template.format(arm=args.arm)}_pad") for _, template, _ in JAWS]
    for angle in (-0.15, -0.05, 0.0, 0.1, 0.2, 0.3, 0.4):
        set_jaw(model, data, args.arm, angle)
        if min(pads) < 0:
            break
        n0 = data.geom_xmat[pads[0]].reshape(3, 3)[:, 2]
        n1 = data.geom_xmat[pads[1]].reshape(3, 3)[:, 2]
        print(f"    {angle:+7.3f} rad   {np.degrees(np.arccos(min(1.0, abs(n0 @ n1)))):5.2f} deg")

    print(f"\n  {args.arm}: {'both jaws present a graspable flat pad' if ok else 'NO FLAT PAD -- see above'}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
