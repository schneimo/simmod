"""
Copyright (c) 2020, Moritz Schneider
@Author: Moritz Schneider
"""
from simmod import modification
from simmod.modification import mujoco, builtin
from simmod.modification.mujoco import *
from simmod.modification.builtin import *
from simmod.wrappers import *
from simmod.utils.load_utils import load_yaml
import os
import typing as tp
from gym import Env


class ExperimentConfiguration:

    def __init__(self, name: tp.AnyStr) -> None:
        self.name = name
        self.configurations = list()

    def add(self, config: tp.Dict, modifier_cls: tp.Optional[tp.Callable] = None, wrapper_cls: tp.Optional[tp.Callable] = None):
        self.configurations.append((modifier_cls, config, wrapper_cls))


class GymExperimentScheduler:
    """Class to create experiments with modified simulations for the OpenAI Gym framework.

    The needed modifiers and wrappers can be specified and loaded via a single configuration file.
    """

    def __init__(self) -> None:
        self._idx = 0
        self._experiment_list = list()

    def __iter__(self) -> tp.Iterator:
        return iter(self._experiment_list)

    def create_modifiers(self, config: tp.Dict, env: Env) -> tp.Optional[tp.List]:
        """
        Creates modifier from the given configuration.

        Args:
            config: Configuration dictionary including the callable class and the configuration dict of the modifier
            env:    OpenAI gym environment for the modifier

        Returns:    List of created modifiers or None if no modifier was created.

        """
        modifiers = list()
        for modifier_cls, modifier_config, _ in config:
            if modifier_cls is None:
                continue
            mod = self._get_modifier(modifier_cls, env, modifier_config)
            modifiers.append(mod)

        if not modifiers:
            return None

        return modifiers

    def _get_modifier(self, modifier_cls: tp.ClassVar, env: Env, modifier_config: tp.Dict):
        if issubclass(modifier_cls, BuiltInModifier):
            return modifier_cls(sim=env.qube, config=modifier_config)  # TODO: env.qube not standardized
        elif issubclass(modifier_cls, mujoco.mujoco_modifier.MujocoBaseModifier):
            mujoco_sim = env.unwrapped.qube.sim  # TODO: env.qube not standardized
            return modifier_cls(sim=mujoco_sim, config=modifier_config)

    def create_wrapped_env(self, config: tp.Dict, env: Env):
        """ Creates the necessary wrappers specified via the configuration file and wraps the given environment
        with them.

        Args:
            config:  Configuration for the wrapper which specifies
            env:    OpenAI gym environment which should be wrapped around.

        Returns:    Wrapped OpenAI gym environment

        """
        for _, wrapper_config, wrapper_cls in config:
            if wrapper_cls is None:
                continue
            env = wrapper_cls(env=env, **wrapper_config)
        return env

    def load_experiments(self, config_path: tp.AnyStr = None, config: tp.Dict = None) -> None:
        """
        Loads many experiments from a given file.

        Args:
            config_path:    File path of the experiment configurations
            config:         Existing configuration in a dictionary
        """
        if config_path is not None:
            experiments_config = load_yaml(config_path)
        elif config is not None:
            experiments_config = config
        else:
            raise ValueError('Parameters config and config_path are None. One must be not None.')
        for exp_name, value in experiments_config.items():
            new_exp = ExperimentConfiguration(exp_name)
            for modifier_name, config in value.items():
                if modifier_name == "wrapper":
                    for wrapper_name, wrapper_config in config.items():
                        wrapper_cls = eval(wrapper_name)
                        new_exp.add(config=wrapper_config, wrapper_cls=wrapper_cls)
                else:
                    modifier_cls = eval(modifier_name)
                    new_exp.add(config=config, modifier_cls=modifier_cls)
            self._experiment_list.append(new_exp)

