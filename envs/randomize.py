"""Seeded domain randomization for the bimanual table-setting scene.

The entry point is :func:`randomize`, which takes ``(model, data, seed)`` and
mutates both in place.  A given seed always produces an identical scene.

Prop placement is driven by a reachability map that is *orientation-aware*: a
cell counts as reachable only if the arm can put its gripper there with a
top-down approach, which is the only approach the pick primitive uses.  Position
alone is not enough -- constraining the approach axis to vertical folds the wrist
under and costs several centimetres of reach, so a position-only map happily
places props the arm can touch but cannot grasp.

Building the map is a two-stage filter:

1.  A cheap forward-kinematics sweep (the same one used to size the table)
    rasterizes gripper site positions into an occupancy grid at the height the
    prop is grasped at, then closes it to fill holes left by the finite joint
    sampling.  Top-down reach is a strict subset of position reach, so this is a
    sound prefilter, and it is what keeps stage 2 affordable.
2.  Every surviving on-table cell is checked with the damped-least-squares
    solver in :mod:`control.ik`: the cell survives only if IK converges there
    with the approach axis vertical, for every grasp yaw tested.  Props get a
    uniformly random yaw, so requiring all yaws is the correct semantics -- a
    cell that only works at one yaw is not safe to spawn into.
3.  Each accepted pose must also be *statically holdable*: inverse dynamics at
    zero velocity and acceleration, rejected if any joint's gravity torque
    exceeds its actuator's forcerange.  Reaching a pose and holding it are
    different questions -- the top-down posture throws the arm's mass out
    horizontally, and the sts3215 servos saturate well inside the kinematic
    workspace, sagging centimetres below the commanded pose.

Stage 2 costs a couple of minutes per (arm, height), so masks are cached both in
process and on disk under ``.cache/reach/``, keyed by a hash of the scene
geometry and the map parameters.  The randomization itself does not affect the
map: it changes masses, colours and lights, never link geometry.
"""

from __future__ import annotations

import colorsys
import dataclasses
import hashlib
import pathlib
from typing import Dict, Optional, Sequence, Tuple

import mujoco
import numpy as np

from control.ik import POS_TOL, ROT_TOL, IKSolver, solve_top_down

# ---------------------------------------------------------------------------
# Reachability map
# ---------------------------------------------------------------------------

CELL = 0.01                    # occupancy grid resolution, metres
GRID_X = (-0.60, 0.60)
GRID_Y = (-0.80, 0.80)
CLOSE_RADIUS = 0.03            # fills holes left by finite joint sampling
# Stay this far inside the reach boundary.  This used to be 0.05, sized for the
# old position-only map whose edge was a fuzzy artefact of the finite joint sweep.
# The stage-2 boundary is a real one -- IK convergence at the primitive's own
# tolerance across every yaw bin -- so a margin that large double-counts safety and erodes the
# mug's band at z = 0.835 to nothing.  Two cells of slack covers the grid
# quantization and the Z_TOLERANCE band.
REACH_MARGIN = 0.02
Z_TOLERANCE = 0.04             # half-height of the band sampled around a grasp height

# Joints swept when mapping reach, and the number of samples for each.  wrist_roll
# and the gripper joint do not move the gripper site, so they are left out.
SWEEP = (("shoulder_pan", 25), ("shoulder_lift", 21), ("elbow_flex", 21), ("wrist_flex", 13))

_ARMS = ("left", "right")

# Stage-2 (IK) filter.  Yaw bins span [0, pi) only: the jaws are symmetric, so a
# grasp at yaw and at yaw + pi are the same grasp, and each bin is tried both
# ways round.
#
# The tolerances are the primitive's own, not looser ones.  They used to be 5 mm
# and 0.15 rad, on the reasoning that the controller ramps onto the target anyway.
# That stopped being true once the primitive began clamping its waypoints: it now
# treats a pose it cannot converge on at 1 mm as unreachable, so a cell this map
# passes at 5 mm is one the primitive will refuse.  Two props per ten seeds
# spawned in exactly that gap -- the map called them reachable at 4.6 mm, the
# primitive could not hold the standoff above them at all, and the arm shoved
# them across the table instead of picking them.  Importing the values keeps the
# two from drifting apart again.  Cost is modest: legal cells go from 116/854/415
# to 88/756/354 for plate/mug/cutlery.
IK_YAW_BINS = 3
IK_POS_TOL = POS_TOL           # m
IK_ROT_TOL = ROT_TOL           # rad
IK_MAX_ITERS = 60
CACHE_DIR = pathlib.Path(__file__).resolve().parent.parent / ".cache" / "reach"


def _disk(radius: float) -> Sequence[Tuple[int, int]]:
    n = int(round(radius / CELL))
    return [(i, j) for i in range(-n, n + 1) for j in range(-n, n + 1) if i * i + j * j <= n * n]


def _morph(grid: np.ndarray, radius: float, dilate: bool) -> np.ndarray:
    """Binary dilation/erosion by a disk, without a scipy dependency."""
    out = np.zeros_like(grid) if dilate else np.ones_like(grid)
    for i, j in _disk(radius):
        shifted = np.roll(np.roll(grid, i, axis=0), j, axis=1)
        # np.roll wraps; blank out the wrapped-in rows/columns.
        if i > 0:
            shifted[:i, :] = False
        elif i < 0:
            shifted[i:, :] = False
        if j > 0:
            shifted[:, :j] = False
        elif j < 0:
            shifted[:, j:] = False
        out = (out | shifted) if dilate else (out & shifted)
    return out


class ReachMap:
    """Occupancy grids of where each arm can make a top-down grasp.

    Building one is expensive (minutes, dominated by the IK stage), so callers
    should reuse it; :func:`reach_map` caches in process and each per-arm mask is
    also cached on disk.
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.nx = int(round((GRID_X[1] - GRID_X[0]) / CELL)) + 1
        self.ny = int(round((GRID_Y[1] - GRID_Y[0]) / CELL)) + 1
        self.model = model
        self._data = data
        self._points = self._sweep(model, data)
        self._solvers: Dict[str, IKSolver] = {}
        self._cache: Dict[Tuple[float, str, float], np.ndarray] = {}
        self._arm_cache: Dict[Tuple[str, float], np.ndarray] = {}
        self._geom_hash = _geometry_hash(model)

    @staticmethod
    def _sweep(model: mujoco.MjModel, data: mujoco.MjData) -> Dict[str, np.ndarray]:
        qadr, grids, sites = {}, {}, {}
        for arm in _ARMS:
            sites[arm] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{arm}_gripperframe")
            if sites[arm] < 0:
                raise ValueError(f"scene has no site {arm}_gripperframe")
            for joint, count in SWEEP:
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{arm}_{joint}")
                qadr[arm, joint] = model.jnt_qposadr[jid]
                lo, hi = model.jnt_range[jid]
                # Shrink slightly: the joint limits themselves are not usable poses.
                grids[arm, joint] = np.linspace(lo * 0.98, hi * 0.98, count)

        scratch = mujoco.MjData(model)
        out = {arm: [] for arm in _ARMS}
        (j0, _), (j1, _), (j2, _), (j3, _) = SWEEP
        for v0 in grids[_ARMS[0], j0]:
            for v1 in grids[_ARMS[0], j1]:
                for v2 in grids[_ARMS[0], j2]:
                    for v3 in grids[_ARMS[0], j3]:
                        for arm in _ARMS:
                            scratch.qpos[qadr[arm, j0]] = v0
                            scratch.qpos[qadr[arm, j1]] = v1
                            scratch.qpos[qadr[arm, j2]] = v2
                            scratch.qpos[qadr[arm, j3]] = v3
                        mujoco.mj_kinematics(model, scratch)
                        for arm in _ARMS:
                            out[arm].append(scratch.site_xpos[sites[arm]].copy())
        return {arm: np.asarray(v) for arm, v in out.items()}

    def _position_mask(self, arm: str, height: float) -> np.ndarray:
        """Stage 1: cells the gripper site can occupy at all, in any orientation."""
        pts = self._points[arm]
        band = pts[np.abs(pts[:, 2] - height) <= Z_TOLERANCE]
        grid = np.zeros((self.nx, self.ny), dtype=bool)
        ix = np.round((band[:, 0] - GRID_X[0]) / CELL).astype(int)
        iy = np.round((band[:, 1] - GRID_Y[0]) / CELL).astype(int)
        ok = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
        grid[ix[ok], iy[ok]] = True
        # The sweep is a point cloud, not a filled region, so close it.
        return _morph(_morph(grid, CLOSE_RADIUS, True), CLOSE_RADIUS, False)

    def top_down_ok(self, arm: str, xy: Sequence[float], height: float,
                    *, seed: Optional[np.ndarray] = None) -> Tuple[bool, np.ndarray]:
        """Whether ``arm`` can reach *and hold* ``xy`` at ``height``, approach axis down.

        Requires a solution at every yaw bin that both converges and passes the
        static-torque check.  Returns ``(ok, q)`` where ``q`` is the last solution
        found, usable as a warm start for a neighbouring cell.
        """
        solver = self._solvers.get(arm)
        if solver is None:
            solver = self._solvers[arm] = IKSolver(self.model, arm)
        target = np.array([xy[0], xy[1], height])
        last = seed
        for k in range(IK_YAW_BINS):
            q, info = solve_top_down(
                solver, self._data, target, np.pi * k / IK_YAW_BINS, seed=last,
                accept=solver.holdable,
                max_iters=IK_MAX_ITERS, pos_tol=IK_POS_TOL, rot_tol=IK_ROT_TOL)
            if not info["converged"]:
                return False, last if last is not None else q
            last = q
        return True, last

    def _arm_mask(self, arm: str, height: float) -> np.ndarray:
        """Stage 1 then stage 2, memoized in process and on disk."""
        key = (arm, round(height, 4))
        if key in self._arm_cache:
            return self._arm_cache[key]

        path = CACHE_DIR / self._geom_hash / f"{arm}_{key[1]:.4f}.npy"
        if path.exists():
            mask = np.load(path)
            if mask.shape == (self.nx, self.ny):
                self._arm_cache[key] = mask
                return mask

        coarse = self._position_mask(arm, height)
        # Only cells on the table can ever hold a prop, so do not pay IK for the rest.
        coarse &= self._table_mask()

        mask = np.zeros_like(coarse)
        warm = None
        ix, iy = np.nonzero(coarse)
        for i, j in zip(ix, iy):
            xy = (i * CELL + GRID_X[0], j * CELL + GRID_Y[0])
            ok, warm = self.top_down_ok(arm, xy, height, seed=warm)
            mask[i, j] = ok

        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, mask)
        self._arm_cache[key] = mask
        return mask

    def _table_mask(self) -> np.ndarray:
        gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "table_top")
        bid = self.model.geom_bodyid[gid]
        centre = self.model.body_pos[bid][:2] + self.model.geom_pos[gid][:2]
        half = self.model.geom_size[gid][:2]
        xs = np.arange(self.nx) * CELL + GRID_X[0]
        ys = np.arange(self.ny) * CELL + GRID_Y[0]
        return ((np.abs(xs - centre[0]) <= half[0])[:, None]
                & (np.abs(ys - centre[1]) <= half[1])[None, :])

    def mask(self, height: float, mode: str = "any", margin: float = REACH_MARGIN) -> np.ndarray:
        """Cells reachable at ``height`` by either arm (``any``) or both (``both``)."""
        key = (round(height, 4), mode, round(margin, 4))
        if key not in self._cache:
            left, right = (self._arm_mask(arm, height) for arm in _ARMS)
            combined = (left | right) if mode == "any" else (left & right)
            self._cache[key] = _morph(combined, margin, False) if margin > 0 else combined
        return self._cache[key]

    def cells_to_xy(self, mask: np.ndarray) -> np.ndarray:
        ix, iy = np.nonzero(mask)
        return np.stack([ix * CELL + GRID_X[0], iy * CELL + GRID_Y[0]], axis=1)


def _geometry_hash(model: mujoco.MjModel) -> str:
    """Identify the kinematics the map depends on, so a stale cache is never reused.

    Only fields that move the gripper site matter; the randomizer's masses,
    colours and lights deliberately do not appear here.
    """
    h = hashlib.sha1()
    for arr in (model.body_pos, model.body_quat, model.jnt_pos, model.jnt_axis,
                model.jnt_range, model.jnt_type, model.jnt_bodyid,
                model.site_pos, model.site_quat, model.site_bodyid,
                model.actuator_forcerange, model.actuator_trnid, model.opt.gravity):
        h.update(np.ascontiguousarray(arr, dtype=np.float64).tobytes())
    # Only the arms' own inertial properties matter for the static-hold check --
    # deliberately not the props, whose masses the randomizer changes per seed.
    arm_bodies = [b for b in range(model.nbody)
                  if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "")
                  .startswith(_ARMS)]
    for arr in (model.body_mass[arm_bodies], model.body_inertia[arm_bodies],
                model.body_ipos[arm_bodies]):
        h.update(np.ascontiguousarray(arr, dtype=np.float64).tobytes())
    h.update(repr((CELL, GRID_X, GRID_Y, CLOSE_RADIUS, Z_TOLERANCE, SWEEP,
                   IK_YAW_BINS, IK_POS_TOL, IK_ROT_TOL, IK_MAX_ITERS,
                   "torque-feasible-v1")).encode())
    return h.hexdigest()[:16]


_REACH_CACHE: Dict[int, Tuple[mujoco.MjModel, ReachMap]] = {}


def reach_map(model: mujoco.MjModel, data: mujoco.MjData) -> ReachMap:
    """Reachability map for ``model``, built once and cached per model instance."""
    entry = _REACH_CACHE.get(id(model))
    if entry is None or entry[0] is not model:
        # The model is stored alongside the map so its id() stays valid (and so a
        # stale entry from a freed model of the same id is detected above).
        _REACH_CACHE[id(model)] = (model, ReachMap(model, data))
    return _REACH_CACHE[id(model)][1]


# ---------------------------------------------------------------------------
# Prop definitions
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class PropSpec:
    name: str                       # body name, also used for the free joint
    material: str
    radius: float                   # planar bounding radius, for spacing and table margin
    spawn_z: float                  # body-frame z at spawn (2 mm of clearance over the top)
    grasp_z: float                  # height the gripper must reach to pick it up
    reach: str                      # "any": one arm suffices; "both": must be shared
    mass: Tuple[float, float]       # kg
    slide: Tuple[float, float]      # tangential friction
    torsion: Tuple[float, float]    # torsional friction
    sat: Tuple[float, float]        # HSV saturation range
    val: Tuple[float, float]        # HSV value range


# The plate is the anchor of the place setting, so it is restricted to the region
# both arms share; the others only need one arm each.
# Heights are the world z of each prop's <name>_grasp geom once it has settled on
# the table, all inside the measured 12-88 mm usable band.  Bodies rest with their
# origin on the table top, so spawn_z is uniformly 2 mm of drop clearance.
PROPS: Tuple[PropSpec, ...] = (
    PropSpec("plate", "plate_mat", 0.017, 0.752, 0.764, "both",
             (0.03, 0.09), (0.4, 1.2), (0.005, 0.050), (0.05, 0.45), (0.70, 1.00)),
    PropSpec("mug", "mug_mat", 0.032, 0.752, 0.782, "any",
             (0.05, 0.15), (0.5, 1.3), (0.005, 0.050), (0.45, 0.90), (0.45, 0.90)),
    PropSpec("spoon", "spoon_mat", 0.075, 0.752, 0.774, "any",
             (0.01, 0.04), (0.3, 1.0), (0.002, 0.020), (0.10, 0.60), (0.50, 0.95)),
    PropSpec("fork", "fork_mat", 0.075, 0.752, 0.774, "any",
             (0.01, 0.04), (0.3, 1.0), (0.002, 0.020), (0.10, 0.60), (0.50, 0.95)),
)

PROP_SPACING = 0.02              # extra gap between prop bounding circles
TABLE_MARGIN = 0.015             # keep prop footprints this far inside the table edge
MOUNT_KEEPOUT = (0.13, 0.11)     # half-extents of the no-spawn box around each arm base

TABLE_MAT = "table_top_mat"
TABLE_HUE = (0.055, 0.125)       # wood-ish hues, as an HSV fraction
TABLE_SAT = (0.25, 0.60)
TABLE_VAL = (0.45, 0.80)

LIGHT_TOTAL = (0.60, 1.10)       # summed diffuse intensity, keeps exposure sane
LIGHT_KEY_SHARE = (0.55, 0.80)
LIGHT_POS_X = (-0.30, 0.60)
LIGHT_POS_Y = (-1.20, 1.20)
LIGHT_POS_Z = (1.60, 2.60)
LIGHT_AIM_JITTER = 0.25
LIGHT_WARMTH = 0.06

MIN_DELTA_E = 25.0               # CIE76 separation between every pair of scene colours
MAX_ATTEMPTS = 200


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------


def _srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB (0-1) to CIELAB under D65, for perceptual distance checks."""
    rgb = np.asarray(rgb, dtype=float)
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    m = np.array([[0.4124, 0.3576, 0.1805],
                  [0.2126, 0.7152, 0.0722],
                  [0.0193, 0.1192, 0.9505]])
    xyz = lin @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16.0 / 116.0)
    return np.array([116.0 * f[1] - 16.0, 500.0 * (f[0] - f[1]), 200.0 * (f[1] - f[2])])


def _delta_e(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(_srgb_to_lab(a) - _srgb_to_lab(b)))


def _sample_colors(rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """Table plus one colour per prop, all mutually distinguishable.

    Prop hues are spread around the wheel in a random order at 90 degree spacing
    with bounded jitter, which alone guarantees at least 30 degrees of hue
    separation; the CIE76 check then also rules out pairs that are far apart in
    hue but still look alike because one is washed out or very dark.
    """
    for _ in range(MAX_ATTEMPTS):
        base = rng.uniform(0.0, 1.0)
        order = rng.permutation(len(PROPS))
        colors: Dict[str, np.ndarray] = {}
        for slot, prop in zip(order, PROPS):
            hue = (base + slot / len(PROPS) + rng.uniform(-1.0, 1.0) / 12.0) % 1.0
            sat = rng.uniform(*prop.sat)
            val = rng.uniform(*prop.val)
            colors[prop.name] = np.array(colorsys.hsv_to_rgb(hue, sat, val))
        colors["table"] = np.array(colorsys.hsv_to_rgb(
            rng.uniform(*TABLE_HUE), rng.uniform(*TABLE_SAT), rng.uniform(*TABLE_VAL)))

        names = list(colors)
        if all(_delta_e(colors[a], colors[b]) >= MIN_DELTA_E
               for i, a in enumerate(names) for b in names[i + 1:]):
            return colors
    raise RuntimeError("could not find a well-separated colour set")


# ---------------------------------------------------------------------------
# Placement
# ---------------------------------------------------------------------------


def _table_bounds(model: mujoco.MjModel) -> Tuple[np.ndarray, np.ndarray]:
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table_top")
    bid = model.geom_bodyid[gid]
    centre = model.body_pos[bid][:2] + model.geom_pos[gid][:2]
    return centre, model.geom_size[gid][:2].copy()


def _mount_keepouts(model: mujoco.MjModel) -> list:
    out = []
    for arm in _ARMS:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{arm}_mount")
        if bid >= 0:
            out.append(model.body_pos[bid][:2].copy())
    return out


def _candidates(model: mujoco.MjModel, rmap: ReachMap, prop: PropSpec) -> np.ndarray:
    """Legal centre positions for ``prop``: reachable, on the table, clear of the bases."""
    xy = rmap.cells_to_xy(rmap.mask(prop.grasp_z, prop.reach))
    if not len(xy):
        raise RuntimeError(f"no reachable cells for {prop.name} at z={prop.grasp_z}")

    centre, half = _table_bounds(model)
    limit = half - prop.radius - TABLE_MARGIN
    if np.any(limit <= 0):
        raise RuntimeError(f"{prop.name} does not fit on the table")
    keep = np.all(np.abs(xy - centre) <= limit, axis=1)

    for mount in _mount_keepouts(model):
        box = np.array(MOUNT_KEEPOUT) + prop.radius
        keep &= ~np.all(np.abs(xy - mount) <= box, axis=1)

    xy = xy[keep]
    if not len(xy):
        raise RuntimeError(f"no legal spawn cells for {prop.name}")
    return xy


def _place_one(prop: PropSpec, pool: np.ndarray, placed: Dict[str, dict],
               rng: np.random.Generator) -> bool:
    """Draw a spot for ``prop`` that clears everything already in ``placed``."""
    for _ in range(MAX_ATTEMPTS):
        xy = pool[rng.integers(len(pool))] + rng.uniform(-CELL / 2, CELL / 2, size=2)
        if all(np.linalg.norm(xy - other["pos"][:2]) >= prop.radius + other["radius"] + PROP_SPACING
               for other in placed.values()):
            yaw = float(rng.uniform(-np.pi, np.pi))
            placed[prop.name] = {
                "pos": np.array([xy[0], xy[1], prop.spawn_z]),
                "yaw": yaw,
                "quat": np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]),
                "radius": prop.radius,
            }
            return True
    return False


def _place(model: mujoco.MjModel, data: mujoco.MjData, rmap: ReachMap,
           rng: np.random.Generator) -> Dict[str, dict]:
    """Sample a non-overlapping, reachable, contact-free layout for all props."""
    qadr = {}
    for prop in PROPS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{prop.name}_free")
        if jid < 0:
            raise ValueError(f"scene has no free joint {prop.name}_free")
        qadr[prop.name] = model.jnt_qposadr[jid]
    pools = {prop.name: _candidates(model, rmap, prop) for prop in PROPS}

    for _ in range(MAX_ATTEMPTS):
        placed: Dict[str, dict] = {}
        if not all(_place_one(prop, pools[prop.name], placed, rng) for prop in PROPS):
            continue

        for prop in PROPS:
            a = qadr[prop.name]
            pose = np.concatenate([placed[prop.name]["pos"], placed[prop.name]["quat"]])
            data.qpos[a:a + 7] = pose
            model.qpos0[a:a + 7] = pose
            # Keep any keyframes consistent, so a later reset does not undo this.
            for k in range(model.nkey):
                model.key_qpos[k, a:a + 7] = pose
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        if data.ncon == 0:
            return placed
    raise RuntimeError("could not find a collision-free prop layout")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def randomize(model: mujoco.MjModel, data: mujoco.MjData, seed: int) -> Dict[str, object]:
    """Randomize props, physics, lighting and colours in place for ``seed``.

    Mutates ``model`` (masses, frictions, lights, materials, ``qpos0`` and any
    keyframes) and ``data`` (prop poses).  Call it after resetting to a keyframe:
    prop poses are written into the keyframes too, so a later reset preserves the
    layout, but the arm pose comes from whatever the caller set.

    Returns a dict describing everything that was sampled.
    """
    rng = np.random.default_rng(seed)
    rmap = reach_map(model, data)

    placed = _place(model, data, rmap, rng)

    physics: Dict[str, dict] = {}
    for prop in PROPS:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, prop.name)
        mass = float(rng.uniform(*prop.mass))
        old = float(model.body_mass[bid])
        if old > 0:
            # Shape is unchanged, so inertia scales with mass.
            model.body_inertia[bid] *= mass / old
        model.body_mass[bid] = mass

        slide = float(rng.uniform(*prop.slide))
        torsion = float(rng.uniform(*prop.torsion))
        for gid in range(model.ngeom):
            if model.geom_bodyid[gid] == bid:
                model.geom_friction[gid, 0] = slide
                model.geom_friction[gid, 1] = torsion
        physics[prop.name] = {"mass": mass, "slide_friction": slide, "torsion_friction": torsion}

    lights = _randomize_lights(model, rng)
    colors = _sample_colors(rng)
    for prop in PROPS:
        _set_material(model, prop.material, colors[prop.name])
    _set_material(model, TABLE_MAT, colors["table"])

    mujoco.mj_forward(model, data)
    return {
        "seed": int(seed),
        "props": {k: {"pos": v["pos"].tolist(), "yaw": v["yaw"]} for k, v in placed.items()},
        "physics": physics,
        "lights": lights,
        "colors": {k: v.tolist() for k, v in colors.items()},
    }


def _randomize_lights(model: mujoco.MjModel, rng: np.random.Generator) -> list:
    centre, _ = _table_bounds(model)
    total = rng.uniform(*LIGHT_TOTAL)
    share = rng.uniform(*LIGHT_KEY_SHARE)
    intensities = [total * share, total * (1.0 - share)]
    out = []
    for i in range(model.nlight):
        pos = np.array([rng.uniform(*LIGHT_POS_X), rng.uniform(*LIGHT_POS_Y), rng.uniform(*LIGHT_POS_Z)])
        aim = np.array([centre[0], centre[1], 0.78]) + np.concatenate(
            [rng.uniform(-LIGHT_AIM_JITTER, LIGHT_AIM_JITTER, size=2), [0.0]])
        direction = aim - pos
        direction /= np.linalg.norm(direction)
        intensity = intensities[i] if i < len(intensities) else rng.uniform(0.2, 0.5)
        warmth = rng.uniform(-LIGHT_WARMTH, LIGHT_WARMTH)
        diffuse = np.clip(intensity * np.array([1.0 + warmth, 1.0, 1.0 - warmth]), 0.0, 1.0)
        model.light_pos[i] = pos
        model.light_dir[i] = direction
        model.light_diffuse[i] = diffuse
        out.append({"pos": pos.tolist(), "dir": direction.tolist(), "diffuse": diffuse.tolist()})
    return out


def _set_material(model: mujoco.MjModel, name: str, rgb: np.ndarray) -> None:
    mid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, name)
    if mid < 0:
        raise ValueError(f"scene has no material {name}")
    model.mat_rgba[mid, :3] = rgb
