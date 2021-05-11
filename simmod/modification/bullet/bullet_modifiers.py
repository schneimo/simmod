"""
Modification Framework for the Bullet simulator for Python.
"""

from collections import defaultdict
from enum import Enum
from typing import AnyStr, List, Union, Dict, Tuple

import numpy as np

import warnings

from simmod.modification.base_modifier import BaseModifier, LightModifier, MaterialModifier, TextureModifier, \
    CameraModifier, JointModifier, InertialModifier
from simmod.utils.typings_ import *


class BulletBaseModifier(BaseModifier):

    def __init__(
            self,
            sim,
            client,
            *args,
            **kwargs
    ) -> None:
        self.sim = sim
        self._pybullet_client = client
        super().__init__(*args, **kwargs)

    @property
    def model(self):
        return self.sim.model

    def update(self):
        """
        Propagates the changes made up to this point through the simulation
        """
        pass


class BulletJointModifier(BulletBaseModifier, JointModifier):

    def __init__(self, *args, **kwargs) -> None:
        self._default_config_file_path = 'data/bullet/default_joint_config.yaml'
        super().__init__(*args, **kwargs)

        num_joints = self._pybullet_client.getNumJoints(self.sim)
        self._joint_name_to_id = {}
        for i in range(num_joints):
            joint_info = self._pybullet_client.getJointInfo(self.sim, i)
            self._joint_name_to_id[joint_info[1].decode("UTF-8")] = joint_info[0]

    @property
    def names(self) -> List:
        return self.model.joint_names

    @property
    def standard_setters(self) -> Dict:
        setters = {
            "range": self.set_range,
            "damping": self.set_damping,
            "armature": self.set_armature,
            "frictionloss": self.set_frictionloss,
            "stiffness": self.set_stiffness,
            "pos": self.set_pos,
            "quat": self.set_quat,
        }
        return setters

    def _get_jointid(self, name: AnyStr) -> int:
        return self.model.joint_name2id(name)

    def set_damping(self, name: AnyStr, value: float):
        link_id = self._get_jointid(name)
        self._pybullet_client.changeDynamics(self.sim, link_id, jointDamping=value)



class MujocoBodyModifier(BulletBaseModifier, InertialModifier):

    def __init__(self, *args, **kwargs) -> None:
        self._default_config_file_path = 'data/bullet/default_body_config.yaml'
        super().__init__(*args, **kwargs)

    @property
    def names(self) -> List:
        return self.model.body_names

    @property
    def standard_setters(self) -> Dict:
        setters = {
            "mass": self.set_mass,
            "diaginertia": self.set_diaginertia,
            "pos": self.set_pos,
            "quat": self.set_quat,
            "friction": self.set_friction,
        }
        return setters

    def _get_bodyid(self, name: AnyStr) -> int:
        assert self.model.body_name2id(name) > -1, "Unknown body %s" % name
        return self.model.body_name2id(name)

    def set_mass(self, name: AnyStr, value: float):
        link_id = self._get_bodyid(name)
        self._pybullet_client.changeDynamics(self.sim, link_id, mass=value)

    def set_friction(self, name: AnyStr, value: float):
        link_id = self._get_bodyid(name)
        self._pybullet_client.changeDynamics(self.sim, link_id, lateralFriction=value)

    def set_diaginertia(self, name: AnyStr, value: Array):
        link_id = self._get_bodyid(name)

        value = list(value)
        assert len(value) == 3, "Expected 3-dim value, got %s" % value

        self._pybullet_client.changeDynamics(self.sim, link_id, localInertiaDiagnoal=value[:])


class BulletOptionModifier(BulletBaseModifier):

    def __init__(self, *args, **kwargs) -> None:
        #self._default_config_file_path = 'data/mujoco/default_option_config.yaml'
        super().__init__(*args, **kwargs)

    @property
    def names(self) -> List:
        return self.model.options # TODO

    @property
    def standard_setters(self) -> Dict:
        setters = {
            "gravity": self.set_gravity,
        }
        return setters

    def set_gravity(self, name: AnyStr, value: Union[float, Array]) -> None:
        if isinstance(value, float):
            value = [0, 0, value]
        elif len(value) < 3:
            assert len(value) <= 3, "Expected value of max. length 3, instead got %s" % value
            value = list(reversed([value[i] if i < len(value) else 0 for i in range(3)]))
        elif len(value) == 3:
            pass
        else:
            raise ValueError
        self._pybullet_client.setGravity(*value)
