#!/usr/bin/env python
# demonstration of markers (visual-only geoms)

import os
import numpy as np
from mujoco_py import load_model_from_xml, MjSim, MjViewer

from simmod.modification.mujoco import MujocoTextureModifier
from simmod.modification.mujoco import MujocoCameraModifier
from simmod.modification.mujoco import MujocoJointModifier
from simmod.modification.mujoco import MujocoLightModifier
from simmod.modification.mujoco import MujocoMaterialModifier
from simmod.modification.mujoco import MujocoBodyModifier

MODEL_XML = """
<?xml version="1.0" encoding="utf-8"?>
<?xml version="1.0" encoding="utf-8"?>
<mujoco model="qube">
    <compiler angle="radian" meshdir="../meshes/" />
    <option timestep="0.01" integrator="RK4" />
    <size njmax="500" nconmax="100" />
    
    <asset>
        <texture builtin="flat" name="texred" height="1024" width="1024" rgb1="0.698 0.1333 0.1333" type="2d"/>
        <texture builtin="flat" name="texsilver" height="512" width="512" rgb1="0.7529 0.7529 0.7529" type="2d"/>
        <material name="redMat" shininess="0.9" specular="1.0" texture="texred"/>
        <material name="silverMat" shininess="0.3" specular="0.75" texture="texsilver"/>
    </asset>

    <visual>
        <global offwidth="2560" offheight="1329" />
    </visual>

    <worldbody>
        <light pos="0.051036 -0.0321818 0.5" dir="0 0 -1" diffuse="0.8 0.8 0.8" specular="0.9 0.9 0.9" />
        <body name="arm" pos="0 0.019632 0" xyaxes='1 0 0 0 -1 0'>
            <inertial pos="0 -0.0425 0" quat="0.707107 -0.707107 0 0" mass="0.095" diaginertia="5.7439e-05 5.7439e-05 4.82153e-07" />
            <joint name="base_motor" pos="0 0 0" axis="0 0 1" range="-90 90" damping="0.0005" />
            <joint name="arm_pole" pos="0 0 0" axis="0 1 0" range="-90 90" damping="3e-5" />
            <geom name="qube0:arm" size="0.003186 0.0425" pos="0 0.0425 0" quat="0.707107 -0.707107 0 0" type="cylinder" friction="0 0 0" material="silverMat"/>
            <body name="pole" pos="0 0.080312 0" xyaxes='1 0 0 0 1 0'>
                <inertial pos="0 0 0.05654" quat="0 1 0 0" mass="0.024" diaginertia="3.34191e-05 3.34191e-05 2.74181e-07" />
                <geom name="qube0:pole" size="0.00478 0.0645" pos="0 0 0.05654" quat="0 1 0 0" type="cylinder" friction="0 0 0" material="redMat"/>
            </body>
        </body>
    </worldbody>

    <actuator>
        <general name="motor_rotation" joint="base_motor" ctrllimited="true" ctrlrange="-18 18" gear="0.005 0 0 0 0 0" />
    </actuator>
</mujoco>
"""


def angle_normalize(x: float) -> float:
    return (x % (2 * np.pi)) - np.pi

model = load_model_from_xml(MODEL_XML)
sim = MjSim(model)
viewer = MjViewer(sim)
jnt_rand = MujocoJointModifier(sim)
body_rand = MujocoBodyModifier(sim)
step = 0
ctrl = 1

values = [6e-04, 4e-05]

while True:
    sim.data.ctrl[:] = ctrl
    sim.step()
    ctrl = 0

    viewer.render()
    for name, value in zip(sim.model.joint_names, values):
        jnt_rand.set_damping(name, value)
    #body_rand.set_mass("pole", 0.09)

    step += 1
    if step > 1000:
        break

print(sim.model.get_xml())
