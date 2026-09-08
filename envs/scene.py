"""Loading the bimanual table scene.

One place, so every script gets the same model.  The only thing this does beyond
compiling the XML is put the grippers on force control -- see
:mod:`control.gripper` for why that cannot live in the scene file.
"""

from __future__ import annotations

import pathlib
from typing import Optional, Union

import mujoco

from control.gripper import force_control_gripper

SCENE = pathlib.Path(__file__).resolve().parent.parent / "scenes" / "bimanual_table.xml"


def load_scene(path: Optional[Union[str, pathlib.Path]] = None) -> mujoco.MjModel:
    """Compile the scene and convert the gripper actuators to torque control."""
    model = mujoco.MjModel.from_xml_path(str(path or SCENE))
    force_control_gripper(model)
    return model
