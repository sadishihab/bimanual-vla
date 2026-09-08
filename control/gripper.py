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

from typing import Dict, List

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
