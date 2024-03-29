"""Base class for the modifiers which defines how an modifier should be structured.

Each modifier is built on a configuration. Each entry in this configuration is stored as a parameterization object and
stored for later use.
If no configuration is given it is taken from a standard configuration file which
usually defines the object parameter bounds at infinity. This might make the simulation unstable. Therefore, it is
recommended to define a configuration for each modifier. If a configuration is given only the objects which are defined in
this configuration are used by the domain randomization algorithms.
"""
import abc
from abc import ABC, abstractmethod
from typing import List, Dict

import numpy as np

from simmod.common.parametrization import Parametrization
from simmod.utils.load_utils import load_yaml


def register_as_setter(name):
    def register(f):
        if isinstance(f, property):
            f.fget.registered_as_setter = True
            f.fget.registered_setter_name = name
        else:
            f.registered_as_setter = True
            f.registered_setter_name = name
        return f
    return register


class BaseModifier(metaclass=abc.ABCMeta):

    def __init__(
            self,
            config: Dict = None,
            random_state=None,
            *args,
            **kwargs
    ):
        self._registered_setters = dict()
        self._init_setters()

        if random_state is None:
            self.random_state = np.random.RandomState()
        elif isinstance(random_state, int):
            # random_state assumed to be an int
            self.random_state = np.random.RandomState(random_state)
        else:
            self.random_state = random_state

        self._basic_config = self._get_basic_config()

        if config is None:
            config = self._basic_config

        default = self._get_default_from_config(config)
        self.instrumentation = list()
        #self.execution_point = execution_point = config['options']['execution']
        for setter in config:

            # 'options' not for configuration of properties
            if setter == 'options':
                continue

            # Get the parameter ranges for every object specified
            # and store this parameterization
            assert setter in self.standard_setters.keys(), \
                "Unknown setter function %s" % setter
            object_names = config[setter].keys()
            use_default = False
            for object_name in object_names:
                if object_name == 'default':
                    use_default = True
                    continue
                properties = config[setter][object_name]
                range_val = properties['values']
                distribution = properties['distribution'] \
                    if 'distribution' in properties.keys() else 'uniform'
                operation = properties['operation'] \
                    if 'operation' in properties.keys() else 'replacing'
                execution_point = properties['execution'] \
                    if 'execution' in properties.keys() else 'RESET'
                mod_inst = Parametrization(setter, object_name, range_val,
                                           distribution, operation,
                                           execution_point)
                self.instrumentation.append(mod_inst)

            # If 'default' was specified for this setter, then we have to get
            # the default configuration for all unspecified objects
            if use_default:
                diff = list(set(self.names) - set(object_names))
                for object_name in diff:
                    properties = config[setter]['default']
                    default_range_val = properties['values']
                    distribution = properties['distribution'] \
                        if 'distribution' in properties.keys() else 'uniform'
                    operation = properties['operation'] \
                        if 'operation' in properties.keys() else 'replacing'
                    execution_point = properties['execution'] \
                        if 'execution' in properties.keys() else 'RESET'
                    mod_inst = Parametrization(setter, object_name,
                                               default_range_val, distribution,
                                               operation, execution_point)
                    self.instrumentation.append(mod_inst)

    def _get_basic_config(self) -> Dict:
        import os
        path = os.path.dirname(__file__)
        path = os.path.join(path, self._default_config_file_path)
        return load_yaml(path)

    def _init_setters(self):
        methods = [getattr(self, method_name) for method_name in dir(self)
                   if callable(getattr(self, method_name))]
        for method in methods:
            if isinstance(method, property):
                method = method.fget
            if hasattr(method, 'registered_as_setter'):
                if method.registered_as_setter:
                    self._registered_setters[method.registered_setter_name] = method

    @staticmethod
    def _get_default_from_config(config: Dict) -> Dict:
        setter_default = dict()
        for setter in config:
            if "default" in config[setter].keys():
                setter_default[setter] = config[setter]["default"]

        return setter_default

    @property
    @abstractmethod
    def names(self) -> List:
        raise NotImplementedError

    @property
    def standard_setters(self) -> Dict:
        return self._registered_setters

    @abstractmethod
    def update(self):
        raise NotImplementedError


class LightModifier(BaseModifier):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def set_pos(self, name, value):
        raise NotImplementedError

    @abstractmethod
    def set_dir(self, name, value):
        raise NotImplementedError

    @abstractmethod
    def set_active(self, name, value):
        raise NotImplementedError

    @abstractmethod
    def set_specular(self, name, value):
        raise NotImplementedError

    @abstractmethod
    def set_ambient(self, name, value):
        raise NotImplementedError

    @abstractmethod
    def set_diffuse(self, name, value):
        raise NotImplementedError

    @abstractmethod
    def set_castshadow(self, name, value):
        raise NotImplementedError


class CameraModifier(BaseModifier):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def set_fovy(self, name, value):
        raise NotImplementedError

    @abstractmethod
    def get_quat(self, name):
        raise NotImplementedError

    @abstractmethod
    def set_quat(self, name, value):
        raise NotImplementedError

    @abstractmethod
    def get_pos(self, name):
        raise NotImplementedError

    @abstractmethod
    def set_pos(self, name, value):
        raise NotImplementedError


class MaterialModifier(BaseModifier):
    """
    Modify material properties of a model. Example use:

        sim = MjSim(...)
        modder = MaterialModifier(sim)
        modder.set_specularity('some_geom', 0.5)
        modder.rand_all('another_geom')

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def set_specularity(self, name, value):
        raise NotImplementedError

    @abstractmethod
    def set_shininess(self, name, value):
        raise NotImplementedError

    @abstractmethod
    def set_reflectance(self, name, value):
        raise NotImplementedError

    @abstractmethod
    def set_texrepeat(self, name, repeat_x, repeat_y):
        raise NotImplementedError

    @abstractmethod
    def rand_all(self, name):
        raise NotImplementedError

    @abstractmethod
    def rand_specularity(self, name):
        raise NotImplementedError

    @abstractmethod
    def rand_shininess(self, name):
        raise NotImplementedError

    @abstractmethod
    def rand_reflectance(self, name):
        raise NotImplementedError

    @abstractmethod
    def rand_texrepeat(self, name, max_repeat=5):
        raise NotImplementedError


class TextureModifier(BaseModifier):
    """
    Modify textures in model. Example use:

        sim = MjSim(...)
        modder = TextureModifier(sim)
        modder.whiten_materials()  # ensures materials won't impact colors
        modder.set_checker('some_geom', (255, 0, 0), (0, 0, 0))
        modder.rand_all('another_geom')

    Note: in order for the textures to take full effect, you'll need to set
    the rgba values for all materials to [1, 1, 1, 1], otherwise the texture
    colors will be modulated by the material colors. Call the
    `whiten_materials` helper method to set all material colors to white.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def get_texture(self, name):
        raise NotImplementedError

    @abstractmethod
    def get_checker_matrices(self, name):
        raise NotImplementedError

    @abstractmethod
    def set_checker(self, name, rgb1, rgb2):
        raise NotImplementedError

    @abstractmethod
    def set_gradient(self, name, rgb1, rgb2, vertical=True):
        """
        Creates a linear gradient from rgb1 to rgb2.

        Args:
        - rgb1 (array): start color
        - rgb2 (array): end color
        - vertical (bool): if True, the gradient in the positive
            y-direction, if False it's in the positive x-direction.
        """
        # NOTE: MuJoCo's gradient uses a sigmoid. Here we simplify
        # and just use a linear gradient... We could change this
        # to just use a tanh-sigmoid if needed.
        raise NotImplementedError

    @abstractmethod
    def set_rgb(self, name, rgb):
        raise NotImplementedError

    @abstractmethod
    def set_noise(self, name, rgb1, rgb2, fraction=0.9):
        """
        Args:
        - name (str): name of geom
        - rgb1 (array): background color
        - rgb2 (array): color of mod_func noise foreground color
        - fraction (float): fraction of pixels with foreground color
        """
        raise NotImplementedError

    @abstractmethod
    def randomize(self):
        raise NotImplementedError

    @abstractmethod
    def rand_all(self, name):
        raise NotImplementedError

    @abstractmethod
    def rand_checker(self, name):
        raise NotImplementedError

    @abstractmethod
    def rand_gradient(self, name):
        raise NotImplementedError

    @abstractmethod
    def rand_rgb(self, name):
        raise NotImplementedError

    @abstractmethod
    def rand_noise(self, name):
        raise NotImplementedError

    @abstractmethod
    def upload_texture(self, name):
        raise NotImplementedError

    @abstractmethod
    def whiten_materials(self, geom_names=None):
        """
        Helper method for setting all material colors to white, otherwise
        the texture setters won't take full effect.

        Args:
        - geom_names (list): list of geom names whose materials should be
            set to white. If omitted, all materials will be changed.
        """
        raise NotImplementedError

    @abstractmethod
    def get_rand_rgb(self, n=1):
        raise NotImplementedError


class JointModifier(BaseModifier):

    @abstractmethod
    def set_range(self, name, value):
        raise NotImplementedError

    @abstractmethod
    def set_damping(self, name, value):
        raise NotImplementedError

    @abstractmethod
    def set_armature(self, name, value):
        raise NotImplementedError

    @abstractmethod
    def set_stiffness(self, name, value):
        raise NotImplementedError

    @abstractmethod
    def set_frictionloss(self, name, value):
        raise NotImplementedError


class InertialModifier(BaseModifier):

    @abstractmethod
    def set_mass(self, name, value):
        raise NotImplementedError

    @abstractmethod
    def set_diaginertia(self, name, value):
        raise NotImplementedError
