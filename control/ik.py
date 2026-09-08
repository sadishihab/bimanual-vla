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

from typing import Dict, Optional, Sequence, Tuple

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
                   yaw: float, **kwargs) -> Tuple[np.ndarray, Dict[str, object]]:
    """Solve for a vertical approach at ``pos``, honouring jaw symmetry.

    A parallel gripper grasps identically at ``yaw`` and ``yaw + pi``, but the two
    are different wrist poses and the arm often reaches only one of them -- near
    the edge of the workspace the wrong choice misses by centimetres.  Every
    caller wanting a top-down grasp should go through here rather than committing
    to one of the pair.  Returns the converged solution if either works, else
    whichever came closer.
    """
    best = None
    for candidate in (yaw, yaw + np.pi):
        q, info = solver.solve(data, pos, top_down_mat(candidate), **kwargs)
        info["yaw"] = float(candidate)
        if info["converged"]:
            return q, info
        score = info["pos_err"] + info["rot_err"]
        if best is None or score < best[2]:
            best = (q, info, score)
    return best[0], best[1]


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
