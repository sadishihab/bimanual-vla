"""Damped-least-squares inverse kinematics for one SO-101 arm.

The arm has five pose-controlling joints (the gripper does not move the tool
frame), so a full six-DOF target is generally not exactly reachable.  Damped
least squares handles that by returning the best-fit joint vector instead of
diverging; :meth:`IKSolver.solve` reports the residual so callers can decide
whether the result is good enough.

Gripper site convention, read off the SO-101 model:

    site +x   approach axis, pointing from the wrist out past the fingertips
    site +z   jaw opening axis, the direction the moving jaw swings
    site +y   completes the right-handed frame

The site itself sits between the jaws, so its position *is* the grasp point.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence, Tuple

import mujoco
import numpy as np

# Joints that move the gripper site, in kinematic order.  The gripper joint is
# excluded: it changes the jaw opening, not the tool frame.
ARM_JOINTS: Tuple[str, ...] = (
    "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll",
)

DAMPING = 0.10          # lambda in (J J^T + lambda^2 I); larger = slower but stabler
MAX_ITERS = 200
MAX_JOINT_STEP = 0.25   # rad, per joint per iteration
POS_TOL = 1e-3          # m
ROT_TOL = 2e-2          # rad
ROT_WEIGHT = 0.6        # orientation rows are down-weighted: position matters more
LIMIT_MARGIN = 0.02     # rad, stay this far off the hard joint stops
PATIENCE = 12           # iterations of no real progress before giving up
MIN_IMPROVEMENT = 1e-5  # residual norm drop that still counts as progress
CLAMP_ITERS = 8         # bisection steps when lowering a waypoint into the envelope
CLAMP_SAFETY = 0.5      # a clamped waypoint must converge to this fraction of POS_TOL


def top_down_mat(open_axis_yaw: float) -> np.ndarray:
    """Tool orientation for a vertical grasp, as a 3x3 world rotation matrix.

    The approach axis (site +x) points straight down and the jaws open along the
    horizontal direction given by ``open_axis_yaw``.
    """
    x_axis = np.array([0.0, 0.0, -1.0])
    z_axis = np.array([np.cos(open_axis_yaw), np.sin(open_axis_yaw), 0.0])
    y_axis = np.cross(z_axis, x_axis)
    return np.stack([x_axis, y_axis, z_axis], axis=1)


def solve_top_down(solver: "IKSolver", data: mujoco.MjData, pos: np.ndarray,
                   yaw: float, *, accept: Optional[Callable[[np.ndarray], bool]] = None,
                   **kwargs) -> Tuple[np.ndarray, Dict[str, object]]:
    """Solve for a vertical approach at ``pos``, honouring jaw symmetry.

    A parallel gripper grasps identically at ``yaw`` and ``yaw + pi``, but the two
    are different wrist poses and the arm often reaches only one of them -- near
    the edge of the workspace the wrong choice misses by centimetres.  Every
    caller wanting a top-down grasp should go through here rather than committing
    to one of the pair.  Returns the converged solution if either works, else
    whichever came closer.

    ``accept`` is an extra predicate a solution must satisfy to count as good --
    the reach map uses it to demand the pose be statically holdable, not merely
    reachable.  A candidate that converges but is rejected by ``accept`` does not
    stop the search: the other side of the pair is still tried.
    """
    best = None
    for candidate in (yaw, yaw + np.pi):
        q, info = solver.solve(data, pos, top_down_mat(candidate), **kwargs)
        info["yaw"] = float(candidate)
        if info["converged"] and (accept is None or accept(q)):
            return q, info
        score = info["pos_err"] + info["rot_err"]
        if best is None or score < best[2]:
            best = (q, info, score)
    best[1]["converged"] = False        # nothing satisfied both tests
    return best[0], best[1]


def reachable_above(solver: "IKSolver", data: mujoco.MjData, base: np.ndarray,
                    yaw: float, height: float, *, floor: float = 0.0,
                    iters: int = CLAMP_ITERS,
                    **kwargs) -> Tuple[np.ndarray, Dict[str, object]]:
    """The highest waypoint ``base + z`` for z in ``[floor, height]`` that IK converges at.

    Commanding a waypoint outside the arm's envelope does not merely miss it.  The
    position servos saturate, the tool frame lags behind the command, and the
    straight Cartesian path the caller asked for is not the path flown: measured
    on the spoon, the lift's sub-step spacing collapsed from 4.2 mm to 3.0 mm as
    the arm ran out of reach, and the resulting off-axis motion sheared the spoon
    out of the jaws.  So a waypoint is worth having only if it converges, and this
    lowers one until it does.

    Bisection, which assumes reachability is monotone in z over the segment.  That
    holds here because the binding constraint is the top-down envelope's ceiling
    -- folding the wrist under to keep the approach vertical costs reach, and
    costs more of it the higher the tool goes.  The returned point is always one
    that converged, except when even ``floor`` fails, which the caller is told
    about rather than left to infer.

    The test is run at :data:`CLAMP_SAFETY` of the normal position tolerance, so
    the answer is inside the envelope rather than on its edge.  DLS is seeded from
    the arm's current pose, and a pose that just scrapes tolerance from one seed
    can miss it from another -- the spoon's clamped lift landed at 1.25 mm when
    re-solved during the motion, having been accepted at 1.00 mm here.
    """
    kwargs.setdefault("pos_tol", POS_TOL * CLAMP_SAFETY)

    def at(z: float) -> Tuple[np.ndarray, Dict[str, object]]:
        pos = base + np.array([0.0, 0.0, z])
        _, info = solver.solve(data, pos, top_down_mat(yaw), **kwargs)
        return pos, info

    pos, info = at(height)
    if info["converged"]:
        return pos, {"height": float(height), "clamped": False, "floor_ok": True, "ik": info}

    low_pos, low_info = at(floor)
    if not low_info["converged"]:
        return low_pos, {"height": float(floor), "clamped": True, "floor_ok": False,
                         "ik": low_info}

    lo, hi = floor, height
    best_pos, best_info = low_pos, low_info
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        pos, info = at(mid)
        if info["converged"]:
            lo, best_pos, best_info = mid, pos, info
        else:
            hi = mid
    return best_pos, {"height": float(lo), "clamped": True, "floor_ok": True,
                      "ik": best_info}


def reachable_toward(solver: "IKSolver", data: mujoco.MjData, point: np.ndarray,
                     goal: np.ndarray, target_mat: np.ndarray, *,
                     iters: int = CLAMP_ITERS,
                     **kwargs) -> Tuple[np.ndarray, Dict[str, object]]:
    """Pull ``point`` toward ``goal`` until IK converges, and return where it stopped.

    The companion to :func:`reachable_above` for a waypoint that is off the
    envelope sideways rather than upward.  A transit interpolated in a straight
    line from the arm's parked pose leaves the workspace in the middle of the
    line -- 23 mm of residual on the way in to a plate that both ends of the line
    reach comfortably -- and that cannot be clamped by lowering it.  ``goal`` must
    itself be reachable, which is what makes the bisection well posed: the search
    is over the fraction of the way from ``point`` to ``goal``.
    """
    _, info = solver.solve(data, point, target_mat, **kwargs)
    if info["converged"]:
        return point, info

    lo, hi = 0.0, 1.0
    best_point, best_info = np.asarray(goal, dtype=float), info
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        candidate = point + (goal - point) * mid
        _, info = solver.solve(data, candidate, target_mat, **kwargs)
        if info["converged"]:
            hi, best_point, best_info = mid, candidate, info
        else:
            lo = mid
    return best_point, best_info


class IKSolver:
    """Reusable DLS solver for one arm.

    Holds a single scratch :class:`mujoco.MjData` so repeated solves do not
    allocate; the caller's ``data`` is never touched.
    """

    def __init__(self, model: mujoco.MjModel, arm: str,
                 joints: Sequence[str] = ARM_JOINTS):
        self.model = model
        self.arm = arm
        self.site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{arm}_gripperframe")
        if self.site < 0:
            raise ValueError(f"scene has no site {arm}_gripperframe")

        self.joints = tuple(joints)
        self.qadr, self.dofadr, lo, hi = [], [], [], []
        for name in self.joints:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{arm}_{name}")
            if jid < 0:
                raise ValueError(f"scene has no joint {arm}_{name}")
            self.qadr.append(model.jnt_qposadr[jid])
            self.dofadr.append(model.jnt_dofadr[jid])
            lo.append(model.jnt_range[jid, 0] + LIMIT_MARGIN)
            hi.append(model.jnt_range[jid, 1] - LIMIT_MARGIN)
        self.qadr = np.asarray(self.qadr)
        self.dofadr = np.asarray(self.dofadr)
        self.lo = np.asarray(lo)
        self.hi = np.asarray(hi)

        # Peak torque each joint's actuator can produce, for the static-hold check.
        limits = []
        for name in self.joints:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{arm}_{name}")
            aids = [a for a in range(model.nu)
                    if model.actuator_trnid[a, 0] == jid
                    and model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT]
            if not aids:
                raise ValueError(f"joint {arm}_{name} has no actuator")
            limits.append(float(np.max(np.abs(model.actuator_forcerange[aids]))))
        self.forcelimit = np.asarray(limits)

        self._scratch = mujoco.MjData(model)
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))

    def fk(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Gripper site position and 3x3 world orientation for joint vector ``q``."""
        s = self._scratch
        s.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.model, s)
        mujoco.mj_comPos(self.model, s)
        return s.site_xpos[self.site].copy(), s.site_xmat[self.site].reshape(3, 3).copy()

    def static_torque(self, q: np.ndarray) -> np.ndarray:
        """Torque each arm joint must produce to hold ``q`` motionless under gravity.

        Inverse dynamics at zero velocity and zero acceleration, so ``qfrc_inverse``
        is exactly the gravity-compensation torque.  Contacts are disabled for the
        evaluation: the question is what the arm must hold on its own, and leaving
        them on would both let the arm lean on whatever it happens to intersect
        (IK ignores collision) and make the answer depend on where the props are,
        which would defeat caching the map.  Joint friction is ignored too, which
        is conservative -- it resists motion, so it only ever helps hold a pose.
        """
        s = self._scratch
        s.qpos[self.qadr] = q
        s.qvel[:] = 0.0
        s.qacc[:] = 0.0
        saved = self.model.opt.disableflags
        self.model.opt.disableflags = int(saved) | int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
        try:
            mujoco.mj_inverse(self.model, s)
        finally:
            self.model.opt.disableflags = saved
        return s.qfrc_inverse[self.dofadr].copy()

    def holdable(self, q: np.ndarray) -> bool:
        """Whether every joint can statically hold ``q`` within its forcerange."""
        return bool(np.all(np.abs(self.static_torque(q)) <= self.forcelimit))

    def solve(self, data: mujoco.MjData, target_pos: np.ndarray,
              target_mat: Optional[np.ndarray] = None, *,
              seed: Optional[np.ndarray] = None,
              max_iters: int = MAX_ITERS,
              damping: float = DAMPING,
              pos_tol: float = POS_TOL,
              rot_tol: float = ROT_TOL,
              patience: int = PATIENCE) -> Tuple[np.ndarray, Dict[str, object]]:
        """Joint targets that put the gripper site at ``target_pos``/``target_mat``.

        Seeded from ``seed`` if given, otherwise from the arm's current pose in
        ``data``.  Returns ``(q, info)`` with the final residuals and whether they
        met tolerance.  DLS stalls rather than diverges at a local minimum, so the
        iteration also stops early once the residual stops improving -- without
        that, a target that cannot be reached costs the full ``max_iters``.
        """
        s = self._scratch
        s.qpos[:] = data.qpos          # other bodies stay put; only this arm is solved
        s.qvel[:] = 0.0
        start = data.qpos[self.qadr] if seed is None else np.asarray(seed, dtype=float)
        q = np.clip(np.asarray(start, dtype=float), self.lo, self.hi)

        target_pos = np.asarray(target_pos, dtype=float)
        target_quat = None
        if target_mat is not None:
            target_quat = np.zeros(4)
            mujoco.mju_mat2Quat(target_quat, np.asarray(target_mat, dtype=float).ravel())

        err = np.zeros(6 if target_quat is not None else 3)
        pos_err = rot_err = np.inf
        best, stalled, iters = np.inf, 0, 0
        for iters in range(1, max_iters + 1):
            s.qpos[self.qadr] = q
            mujoco.mj_kinematics(self.model, s)
            mujoco.mj_comPos(self.model, s)

            err[:3] = target_pos - s.site_xpos[self.site]
            pos_err = float(np.linalg.norm(err[:3]))
            if target_quat is not None:
                cur_quat, neg, diff, vel = (np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(3))
                mujoco.mju_mat2Quat(cur_quat, s.site_xmat[self.site])
                mujoco.mju_negQuat(neg, cur_quat)
                mujoco.mju_mulQuat(diff, target_quat, neg)   # world-frame rotation error
                mujoco.mju_quat2Vel(vel, diff, 1.0)
                err[3:] = ROT_WEIGHT * vel
                rot_err = float(np.linalg.norm(vel))
            else:
                rot_err = 0.0

            if pos_err < pos_tol and rot_err < rot_tol:
                break

            residual = float(np.linalg.norm(err))
            if residual < best - MIN_IMPROVEMENT:
                best, stalled = residual, 0
            else:
                stalled += 1
                if stalled >= patience:
                    break

            mujoco.mj_jacSite(self.model, s, self._jacp, self._jacr, self.site)
            jac = self._jacp[:, self.dofadr]
            if target_quat is not None:
                jac = np.vstack([jac, ROT_WEIGHT * self._jacr[:, self.dofadr]])

            # dq = J^T (J J^T + lambda^2 I)^-1 e
            n = jac.shape[0]
            dq = jac.T @ np.linalg.solve(jac @ jac.T + damping ** 2 * np.eye(n), err)
            dq = np.clip(dq, -MAX_JOINT_STEP, MAX_JOINT_STEP)
            q = np.clip(q + dq, self.lo, self.hi)

        return q, {
            "pos_err": pos_err,
            "rot_err": rot_err,
            "iters": iters,
            "converged": bool(pos_err < pos_tol and rot_err < rot_tol),
        }
