#!/usr/bin/env python3
"""Report the collision geometry of one arm's jaws, as MuJoCo actually sees it.

MuJoCo collides a mesh geom as its *convex hull*.  The SO-101 jaws are concave
parts, so the hull is not the shape the CAD draws, and a grasp is decided by the
hull rather than by the visible finger.  This prints, for each jaw, the hull
facets that face the grasp aperture: how far they span, and how far each one is
tilted off parallel to the jaw opening axis.

A parallel-jaw pinch needs facets near 0 degrees.  A tilted facet meeting the
flat side of an object touches only along the object's top edge, and its normal
then has a vertical component that drives the object down rather than squeezing
it, so the grasp is a wedge no matter what height it is made at.

No seed and no physics: this is a property of the model, not of a scene.
"""

import argparse
import pathlib
import sys

import mujoco
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from control.primitives import _body_geoms  # noqa: E402
from envs.scene import SCENE, load_scene  # noqa: E402

# The grasp region: how far from the gripper site a facet must lie to be able to
# touch a grasped object at all.
DEPTH_WINDOW = (-0.015, 0.012)   # m along the approach axis, +ve toward the fingertips
LATERAL_WINDOW = 0.020           # m either side of the site
FACING = 0.30                    # min |normal . opening| to count as facing the aperture
FLAT = 5.0                       # deg off parallel that still counts as a flat pad


def jaw_facets(model, data, arm, body, sign, site):
    """Aperture-facing hull facets of ``body``, as (depth_lo, depth_hi, tilt_deg, area)."""
    origin = data.site_xpos[site]
    rot = data.site_xmat[site].reshape(3, 3)
    approach, lateral, opening = rot[:, 0], rot[:, 1], rot[:, 2]

    out = []
    for gid in _body_geoms(model, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)):
        mesh = model.geom_dataid[gid]
        if mesh < 0 or model.geom_group[gid] != 3:
            continue
        gr = data.geom_xmat[gid].reshape(3, 3)
        va, nv = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        verts = data.geom_xpos[gid] + model.mesh_vert[va:va + nv].astype(float) @ gr.T - origin
        pa, pn = model.mesh_polyadr[mesh], model.mesh_polynum[mesh]
        for k in range(pn):
            normal = model.mesh_polynormal[pa + k] @ gr.T
            if sign * (normal @ opening) < FACING:
                continue
            a0 = model.mesh_polyvertadr[pa + k]
            poly = verts[model.mesh_polyvert[a0:a0 + model.mesh_polyvertnum[pa + k]]]
            depth = poly @ approach
            if depth.max() < DEPTH_WINDOW[0] or depth.min() > DEPTH_WINDOW[1]:
                continue
            if np.abs(poly @ lateral).min() > LATERAL_WINDOW:
                continue
            area = 0.0
            for i in range(1, len(poly) - 1):
                area += 0.5 * np.linalg.norm(np.cross(poly[i] - poly[0], poly[i + 1] - poly[0]))
            tilt = float(np.degrees(np.arccos(min(1.0, abs(normal @ opening)))))
            out.append((float(depth.min()), float(depth.max()), tilt, float(area)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("left", "right"), default="left")
    parser.add_argument("--scene", type=pathlib.Path, default=SCENE)
    args = parser.parse_args()

    model = load_scene(args.scene)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{args.arm}_gripperframe")

    ok = True
    print(f"{args.arm} gripper, hull facets facing the grasp aperture")
    for label, body, sign in (("fixed finger", f"{args.arm}_gripper", +1),
                              ("moving jaw", f"{args.arm}_moving_jaw_so101_v1", -1)):
        facets = jaw_facets(model, data, args.arm, body, sign, site)
        if not facets:
            print(f"  {label:13s} no facets face the aperture near the site")
            ok = False
            continue
        tilts = np.array([f[2] for f in facets])
        areas = np.array([f[3] for f in facets])
        widest = facets[int(np.argmax(areas))]
        flat = tilts < FLAT
        print(f"  {label:13s} {len(facets):4d} facets"
              f"   tilt off parallel: min {tilts.min():5.1f} deg,"
              f" median {np.median(tilts):5.1f} deg")
        print(f"  {'':13s} largest facet {widest[3] * 1e6:7.1f} mm^2 spans depth"
              f" {widest[0] * 1000:+7.2f} .. {widest[1] * 1000:+7.2f} mm at {widest[2]:5.1f} deg")
        print(f"  {'':13s} facets within {FLAT:.0f} deg of parallel: {int(flat.sum())}")
        if not flat.any():
            ok = False
    print(f"  {args.arm}: {'both jaws present a flat pad' if ok else 'NO FLAT PAD -- see above'}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
