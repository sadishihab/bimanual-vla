"""The table-setting goal: where each prop is supposed to end up.

A place setting is defined relative to one anchor, the plate, with the diner
seated back along -x and facing out along +x.  From that seat, left is +y, right
is -y, and "above" -- further from the diner, as a place-setting diagram draws it
-- is +x.  So:

    plate   the anchor, centred in front of the diner
    fork    to the diner's left
    spoon   to the diner's right
    mug     above and to the right

The offsets are wider than the props' own bounding radii plus the spacing the
randomizer uses, so a completed setting is not in collision with itself, and
:func:`check_setting` asserts it rather than trusting the arithmetic.

They are also wider than table etiquette would have them, and that is the reach
envelope talking.  The two mounts' keepouts cut a notch out of the near half of
the table: for x below about +0.02 nothing outside |y| < 0.04 is legal at all, so
a setting spread across y has to sit at or beyond that line.  Past x = +0.12 the
middle drops out too, both arms being at full stretch straight ahead, and only
|y| > 0.14 survives.  The mug is placed up and to the right within what is left,
which puts it further out in y and less far up in x than a diagram would.

The anchor is not written down.  It is searched for over the same reach masks the
randomizer spawns into, because a goal position the arms cannot get to is not a
goal -- and the masks are per-prop and per-arm, so whether a whole setting fits
is a question about four positions at once, not about any of them alone.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import mujoco
import numpy as np

from envs.randomize import (MOUNT_KEEPOUT, PROPS, PROP_SPACING, TABLE_MARGIN,
                            _morph, _mount_keepouts, _table_bounds, reach_map)

# Offsets from the plate, in metres, in the table frame described above.
SETTING: Dict[str, Tuple[float, float]] = {
    "plate": (0.00, 0.00),
    "fork": (0.00, 0.10),
    "spoon": (0.00, -0.10),
    "mug": (0.04, -0.19),
}

# Which way the diner faces, as a rotation of the whole setting.  Left and right
# are the diner's own, so rotating the seat leaves the setting correct.  It is
# worth trying all four because the two arms split the table at y = 0 and cannot
# hand a prop over: with the diner along -x or +x the fork and spoon slots
# straddle that line and are necessarily one arm each, while with the diner along
# -y or +y the whole setting lies on one side of it and one arm can own all four.
SEATS = (0.0, np.pi, np.pi / 2, -np.pi / 2)
ANCHOR_STEP = 0.01       # m, the reach map's own cell size
PLACE_ORDER = ("plate", "fork", "spoon", "mug")

# The reach map is indexed by where a *prop* sits, but what has to be reachable is
# where the *tool site* goes, and the site stands off the prop's origin by a fixed
# body-frame vector -- see :func:`control.primitives.site_offset`.  For the flatware
# that is 11 to 13 mm, near enough one cell to hide; for the plate and mug it is 21
# and 22 mm, two cells, and it does not hide.  Ignoring it made these tests promise
# more than the arms can do: on seed 0 the goal layout handed the mug a slot whose
# release pose its receiving arm misses by 7.5 mm, and said all three
# midline-crossing props could be picked up and put down when the mug could not.
#
# The offset is not a safety margin to be eroded away, it is a translation, and
# which translation is known: a top-down grasp preserves the prop's yaw from the
# pick through to the release, so the prop arrives at its slot lying the way it
# lies now.  A round grasp feature grasps the same at every yaw and the planner
# searches them, so for the plate and the mug any direction will do -- which makes
# their legal set larger, not smaller, since the arm has to reach the grasp site
# and not the prop's own centre.  Eroding by a disk instead is what left seeds 0-9
# with no legal setting at all.
_SITE_MASKS: Dict[Tuple, np.ndarray] = {}


def _site_shifts(model: mujoco.MjModel, data: mujoco.MjData,
                 name: str) -> Tuple[Tuple[int, int], ...]:
    """Cell offsets from a prop's origin to the tool sites that could grasp it.

    One entry per grasp the planner would consider, as whole cells of the reach
    map's own grid: the two ends of the symmetric yaw pair for a feature whose yaw
    the prop fixes, and the same pair over every searched yaw for a round one.
    """
    from control.primitives import (GRASP_YAW_BINS, JAW_CLEARANCE,  # noqa: PLC0415
                                    _grasp_feature)
    from envs.randomize import CELL

    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    f = _grasp_feature(model, bid)
    rot = data.xmat[bid].reshape(3, 3)
    lead = (rot @ f["centre"])[:2]
    jaw = f["width"] / 2.0 + JAW_CLEARANCE

    if f["yaw_free"]:
        yaws = [np.pi * k / GRASP_YAW_BINS for k in range(GRASP_YAW_BINS)]
    else:
        axis = rot[:, f["narrow"]][:2]
        yaws = [float(np.arctan2(axis[1], axis[0]))
                if float(np.linalg.norm(axis)) > 1e-9 else 0.0]

    out = []
    for yaw in yaws:
        step = jaw * np.array([np.cos(yaw), np.sin(yaw)])
        for v in (lead - step, lead + step):
            cell = (int(round(v[0] / CELL)), int(round(v[1] / CELL)))
            if cell not in out:
                out.append(cell)
    return tuple(out)


def _reindex(base: np.ndarray, shifts: Sequence[Tuple[int, int]]) -> np.ndarray:
    """Mask of prop cells whose tool site lands on a set cell of ``base``."""
    out = np.zeros_like(base)
    for di, dj in shifts:
        shifted = np.roll(np.roll(base, -di, axis=0), -dj, axis=1)
        # np.roll wraps; blank out the wrapped-in rows and columns.
        if di < 0:
            shifted[:-di, :] = False
        elif di > 0:
            shifted[-di:, :] = False
        if dj < 0:
            shifted[:, :-dj] = False
        elif dj > 0:
            shifted[:, -dj:] = False
        out |= shifted
    return out


def _site_mask(model: mujoco.MjModel, data: mujoco.MjData, rmap, spec,
               arm: Optional[str]) -> np.ndarray:
    """``spec``'s reach mask, re-indexed from tool-site cells to prop cells.

    ``arm`` names one arm, or ``None`` for the whole-setting mask that honours the
    prop's own ``reach`` mode.  Memoized because :func:`goal_layout` asks for these
    once per candidate anchor per seat.
    """
    shifts = _site_shifts(model, data, spec.name)
    key = (id(rmap), spec.name, arm, shifts)
    if key not in _SITE_MASKS:
        base = (rmap.mask(spec.grasp_z, spec.reach) if arm is None
                else rmap._arm_mask(arm, spec.grasp_z))
        _SITE_MASKS[key] = _reindex(base, shifts)
    return _SITE_MASKS[key]


def _legal(model: mujoco.MjModel, data: mujoco.MjData, rmap, spec) -> np.ndarray:
    """Boolean mask of cells this prop may occupy: reachable, on the table, clear."""
    mask = _site_mask(model, data, rmap, spec, None).copy()
    xy = rmap.cells_to_xy(mask)
    centre, half = _table_bounds(model)
    limit = half - spec.radius - TABLE_MARGIN
    keep = np.all(np.abs(xy - centre) <= limit, axis=1)
    for mount in _mount_keepouts(model):
        keep &= ~np.all(np.abs(xy - mount) <= np.array(MOUNT_KEEPOUT) + spec.radius, axis=1)
    out = np.zeros_like(mask)
    ix, iy = np.nonzero(mask)
    out[ix[keep], iy[keep]] = True
    return out


def _cell(rmap, xy: Sequence[float]) -> Tuple[int, int]:
    from envs.randomize import CELL, GRID_X, GRID_Y
    return (int(round((xy[0] - GRID_X[0]) / CELL)), int(round((xy[1] - GRID_Y[0]) / CELL)))


def goal_layout(model: mujoco.MjModel, data: mujoco.MjData,
                anchor: Optional[Sequence[float]] = None) -> Dict[str, np.ndarray]:
    """Target xy for every prop, as a whole setting that all four can be put into.

    With no ``anchor``, searches for one: every legal plate cell is tried, each
    prop's target is required to be legal for that prop, and the winner is the
    setting whose props sit furthest inside their own reach masks -- measured as
    the smallest distance from any of the four to the edge of what it is allowed,
    so the choice is not decided by whichever prop happens to be most cramped.

    A target also has to be reachable by an arm that can reach the prop *where it
    currently is*, because one arm does both halves and nothing here hands a prop
    over.  Ignoring that produced settings where, on seed 0, the left arm was the
    only one that could pick the spoon up and the right the only one that could
    put it down.  So the goal depends on the layout it starts from, which is what
    setting a table from wherever things happen to be actually means.
    """
    rmap = reach_map(model, data)
    specs = {s.name: s for s in PROPS}
    legal = {name: _legal(model, data, rmap, specs[name]) for name in SETTING}
    # Which arms can get to each prop where it stands now.
    start_arms = {}
    for name in SETTING:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        start_arms[name] = set(arms_reaching(model, data, name, data.xpos[bid][:2]))

    def targets(at: np.ndarray, seat: float) -> Dict[str, np.ndarray]:
        c, s_ = np.cos(seat), np.sin(seat)
        rot = np.array([[c, -s_], [s_, c]])
        return {n: at + rot @ np.asarray(off) for n, off in SETTING.items()}

    if anchor is not None:
        return targets(np.asarray(anchor, dtype=float), SEATS[0])

    # Distance from each cell to the nearest illegal one, so "furthest inside" is
    # a number rather than a feeling.  Eroding the mask repeatedly is enough at
    # this grid size and keeps the dependency list where it is.
    from envs.randomize import _morph
    depth = {n: np.zeros_like(mask, dtype=float) for n, mask in legal.items()}
    for name, mask in legal.items():
        eroded = mask.copy()
        for step in range(1, 12):
            eroded = _morph(eroded, ANCHOR_STEP, False)
            if not eroded.any():
                break
            depth[name][eroded] = step * ANCHOR_STEP

    from envs.randomize import CELL, GRID_X, GRID_Y
    # Every legal anchor in every seat, scored first by how many props one arm
    # could both pick up and put down -- a prop whose slot is across the midline
    # from where it lies cannot be moved at all -- and only then by how far inside
    # its own mask the setting sits.
    best = None
    for seat in SEATS:
        for i, j in zip(*np.nonzero(legal["plate"])):
            at = np.array([i * CELL + GRID_X[0], j * CELL + GRID_Y[0]])
            margins, movable = [], 0
            for name, target in targets(at, seat).items():
                ci, cj = _cell(rmap, target)
                if not (0 <= ci < legal[name].shape[0]
                        and 0 <= cj < legal[name].shape[1]
                        and legal[name][ci, cj]):
                    margins = None
                    break
                if start_arms[name] & set(arms_reaching(model, data, name, target)):
                    movable += 1
                margins.append(depth[name][ci, cj])
            if margins is None:
                continue
            score = (movable, min(margins))
            if best is None or score > best[0]:
                best = (score, at, seat)
    if best is None:
        raise RuntimeError("no anchor puts a whole setting inside the reach envelope")
    return targets(best[1], best[2])


def check_setting(model: mujoco.MjModel, layout: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Confirm a layout's props cannot overlap, and return the pairwise slack.

    The offsets above are chosen by hand against the reach envelope, so the thing
    worth checking is the one they could get wrong: two props asked to occupy the
    same piece of table.
    """
    radius = {s.name: s.radius for s in PROPS}
    slack = {}
    for i, a in enumerate(sorted(layout)):
        for b in sorted(layout)[i + 1:]:
            gap = float(np.linalg.norm(layout[a] - layout[b]))
            need = radius[a] + radius[b] + PROP_SPACING
            slack[f"{a}-{b}"] = gap - need
            if gap < need:
                raise RuntimeError(f"setting puts {a} and {b} {gap * 1000:.0f} mm apart, "
                                   f"which is inside their {need * 1000:.0f} mm footprints")
    return slack


def arms_reaching(model: mujoco.MjModel, data: mujoco.MjData, name: str,
                  xy: Sequence[float]) -> Tuple[str, ...]:
    """Which arms' reach masks cover ``xy`` at ``name``'s grasp height.

    The pick and the place have to be done by the same arm -- nothing here hands
    a prop over -- so the arm has to be chosen against both ends of the job.  An
    arm that can lift a prop but cannot reach where it is supposed to go will
    carry it to a pose its IK misses by centimetres and let go there.
    """
    spec = {s.name: s for s in PROPS}[name]
    rmap = reach_map(model, data)
    ci, cj = _cell(rmap, xy)
    out = []
    for arm in ("left", "right"):
        mask = _site_mask(model, data, rmap, spec, arm)
        if 0 <= ci < mask.shape[0] and 0 <= cj < mask.shape[1] and mask[ci, cj]:
            out.append(arm)
    return tuple(out)


def crossings(model: mujoco.MjModel, data: mujoco.MjData,
              layout: Dict[str, np.ndarray]) -> Dict[str, Tuple[str, ...]]:
    """Props no single arm can both pick up and put down, and the arms involved.

    The two arms split the table at y = 0 and nothing hands a prop across, so a
    prop that starts on one side of a slot on the other side cannot be moved at
    all.  Naming them is more useful than watching four picks fail.
    """
    out = {}
    for name, target in layout.items():
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        here = set(arms_reaching(model, data, name, data.xpos[bid][:2]))
        there = set(arms_reaching(model, data, name, target))
        if not (here & there):
            out[name] = (tuple(sorted(here)), tuple(sorted(there)))
    return out
