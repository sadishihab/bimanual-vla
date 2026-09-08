"""Loading the bimanual table scene.

One place, so every script gets the same model.  Two things happen beyond
compiling the XML, and neither can be spelled in the scene file, because the arms
arrive through ``<attach>`` and a submodel's bodies and actuators are copied in
wholesale:

* the grippers go on force control, so a squeeze survives the lift;
* each jaw gets an explicit flat pad and the props stop colliding with the jaw
  meshes, so a grasp acts against the finger rather than against the convex hull
  that spans it.

:mod:`control.gripper` carries the measurements behind both.
"""

from __future__ import annotations

import pathlib
from typing import Optional, Union

import mujoco

from control.gripper import add_jaw_pads, force_control_gripper, isolate_grasp_contacts

SCENE = pathlib.Path(__file__).resolve().parent.parent / "scenes" / "bimanual_table.xml"


def load_scene(path: Optional[Union[str, pathlib.Path]] = None) -> mujoco.MjModel:
    """Compile the scene, pad the jaws and put the gripper actuators on torque."""
    spec = mujoco.MjSpec.from_file(str(path or SCENE))
    # Compiled twice on purpose: the pads are laid on the jaw faces as measured
    # off this first compile, rather than on numbers typed into the source.
    add_jaw_pads(spec, spec.compile())
    model = spec.compile()
    force_control_gripper(model)
    isolate_grasp_contacts(model)
    return model
