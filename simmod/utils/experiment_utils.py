"""
Copyright (c) 2020, Moritz Schneider
@Author: Moritz Schneider
"""
from simmod.modification.mujoco import *
from simmod.utils.load_utils import load_yaml
import os
import typing as tp


class ExperimentConfiguration:

    def __init__(self, name: tp.AnyStr) -> None:
        self.name = name
        self.configurations = list()

    def add(self, modifier_cls: tp.Callable, config: tp.Dict, wrapper_cls: tp.Optional[tp.Callable] = None):
        self.configurations.append((modifier_cls, config, wrapper_cls))


class ExperimentScheduler:

    def __init__(self) -> None:
        self._idx = 0
        self._experiment_list = list()

    def __iter__(self) -> tp.Iterator:
        return iter(self._experiment_list)

    def create_modifiers(self, config: tp.Dict, env) -> tp.List:
        """
        Creates modifier from the given configuration.

        Args:
            config: Configuration dictionary including the callable class and the configuration dict of the modifier
            env:    OpenAI gym environment for the modifier

        Returns:    List of created modifiers

        """
        mujoco_sim = env.unwrapped.qube.sim
        modifiers = list()
        for modifier_cls, modifier_config, _ in config:
            mod = modifier_cls(sim=mujoco_sim, config=modifier_config)
            modifiers.append(mod)
        return modifiers

    def create_wrapper(self, config):
        pass

    def load_experiments(self, config_path: tp.AnyStr) -> None:
        """
        Loads many experiments from a given file.

        Args:
            config_path: File path of the experiment configurations
        """
        experiments_config = load_yaml(config_path)
        for exp_name, value in experiments_config.items():
            new_exp = ExperimentConfiguration(exp_name)
            for modifier_name, config in value.items():
                if modifier_name == "wrapper":
                    wrapper_func = self.create_wrapper(config)
                else:
                    modifier_cls = eval(modifier_name)
                    new_exp.add(modifier_cls, config)
            self._experiment_list.append(new_exp)

