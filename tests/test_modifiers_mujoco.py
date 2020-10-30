import pytest

import xml.etree.ElementTree as ET
import numpy as np
from mujoco_py import load_model_from_xml, MjSim, MjViewer, load_model_from_path

from simmod.modification.mujoco import MujocoBodyModifier
from simmod.modification.mujoco import MujocoJointModifier
from simmod.modification.mujoco import MujocoLightModifier
from simmod.modification.mujoco import MujocoCameraModifier

TEST_MODEL_FURUTA_PATH = 'assets/test_mujoco_furuta.xml'
TEST_MODEL_FRICTION_PATH = 'assets/test_mujoco_friction.xml'


def start_trajectory(sim, steps=1000, ctrl=3, viewer=None):
    traj = list()
    for _ in range(steps):
        if sim.data.ctrl is not None:
            sim.data.ctrl[:] = ctrl
        sim.step()
        traj.append(sim.get_state().qpos)
        ctrl = 0
        if viewer is not None:
            viewer.render()
    return traj


def find_body(body, name):
    for b in body.findall('body'):
        body_name = b.get('name')
        if body_name == name:
            return b
        else:
            if find_body(b, name) is not None:
                return find_body(b, name)
            else:
                continue
    else:
        if body.get('name') == "worldbody":
            raise NameError(f"Cannot find body {name}")


######################   BODY   ######################


def test_mujoco_body_modifier_mass():
    # Create the trajectory by changing the XML
    tree = ET.parse(TEST_MODEL_FURUTA_PATH)
    model_xml = tree.getroot()
    worldbody = model_xml.find('worldbody')
    body = find_body(worldbody, 'pole')
    inertia = body.find('inertial')
    inertia.set('mass', str(0.09))
    model_string = ET.tostring(model_xml, encoding='unicode', method='xml')
    model = load_model_from_xml(model_string)
    sim = MjSim(model)
    traj_xml = np.asarray(start_trajectory(sim))

    # Create the trajectory with the modifier which should be the same as the XML generated trajectory
    del sim, model, model_string, tree
    model = load_model_from_path(TEST_MODEL_FURUTA_PATH)
    sim = MjSim(model)
    body_mod = MujocoBodyModifier(sim)
    body_mod.set_mass("pole", 0.09)
    sim.set_constants()
    traj_mod = np.asarray(start_trajectory(sim))
    assert np.sum(traj_mod - traj_xml) == 0.0

    # Create another trajectory with the modifier which should be different now
    del sim, model, body_mod, traj_mod
    model = load_model_from_path(TEST_MODEL_FURUTA_PATH)
    sim = MjSim(model)
    body_mod = MujocoBodyModifier(sim)
    body_mod.set_mass("pole", 0.01)
    sim.set_constants()
    traj_mod = np.asarray(start_trajectory(sim))
    assert np.sum(traj_mod - traj_xml) != 0.0


def test_mujoco_body_modifier_friction():
    # Create the trajectory by changing the XML
    tree = ET.parse(TEST_MODEL_FRICTION_PATH)
    model_xml = tree.getroot()
    worldbody = model_xml.find('worldbody')
    body = find_body(worldbody, 'plane')
    geom = body.find('geom')
    geom.set('friction', f"{0.5} {0.5} {0.5}")
    body = find_body(worldbody, 'ball')
    geom = body.find('geom')
    geom.set('friction', f"{0.5} {0.5} {0.5}")
    model_string = ET.tostring(model_xml, encoding='unicode', method='xml')
    model = load_model_from_xml(model_string)
    sim = MjSim(model)
    traj_xml = np.asarray(start_trajectory(sim))

    # Create the trajectory with the modifier which should be the same as the XML generated trajectory
    del sim, model, model_xml, model_string, tree
    model = load_model_from_path(TEST_MODEL_FRICTION_PATH)
    sim = MjSim(model)
    body_mod = MujocoBodyModifier(sim)
    body_mod.set_friction("ball", [0.5, 0.5, 0.5])
    body_mod.set_friction("plane", [0.5, 0.5, 0.5])
    sim.set_constants()
    traj_mod = np.asarray(start_trajectory(sim))
    assert np.sum(traj_mod - traj_xml) == 0.0

    # Create another trajectory with the modifier which should be different now
    del sim, model, traj_mod, body_mod
    model = load_model_from_path(TEST_MODEL_FRICTION_PATH)
    sim = MjSim(model)
    body_mod = MujocoBodyModifier(sim)
    body_mod.set_friction("ball", [1., 0.5, 0.5])
    body_mod.set_friction("plane", [1., 0.5, 0.5])
    sim.set_constants()
    traj_mod = np.asarray(start_trajectory(sim))
    assert np.sum(traj_mod - traj_xml) != 0.0


######################   JOINT   ######################

def test_mujoco_joint_modifier_damping():
    # Create the trajectory by changing the XML
    tree = ET.parse(TEST_MODEL_FURUTA_PATH)
    model_xml = tree.getroot()
    worldbody = model_xml.find('worldbody')
    body = find_body(worldbody, 'arm')
    joints = body.findall('joint')
    for joint in joints:
        if joint.get('name') == "arm_pole":
            joint.set('damping', str(3e-4))
    model_string = ET.tostring(model_xml, encoding='unicode', method='xml')
    model = load_model_from_xml(model_string)
    sim = MjSim(model)
    traj_xml = np.asarray(start_trajectory(sim))

    # Create the trajectory with the modifier which should be the same as the XML generated trajectory
    del sim, model
    model = load_model_from_path(TEST_MODEL_FURUTA_PATH)
    sim = MjSim(model)
    jnt_mod = MujocoJointModifier(sim)
    jnt_mod.set_damping("arm_pole", 3e-4)
    sim.set_constants()
    traj_mod = np.asarray(start_trajectory(sim))
    assert np.sum(traj_mod - traj_xml) == 0.0

    # Create another trajectory with the modifier which should be different now
    del sim, model, traj_mod, jnt_mod
    model = load_model_from_path(TEST_MODEL_FURUTA_PATH)
    sim = MjSim(model)
    jnt_mod = MujocoJointModifier(sim)
    jnt_mod.set_damping("arm_pole", 1e-4)
    sim.set_constants()
    traj_mod = np.asarray(start_trajectory(sim))
    assert np.sum(traj_mod - traj_xml) != 0.0


if __name__ == "__main__":
    test_mujoco_body_modifier_friction()
    test_mujoco_joint_modifier_damping()
    test_mujoco_body_modifier_mass()
    print('Congratulations! No errors!')
