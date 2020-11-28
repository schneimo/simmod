"""
Copyright (c) 2020, Moritz Schneider
@Author: Moritz Schneider
"""
from typing import List, Dict, Optional, Callable, Tuple

import gym
import copy
import numpy as np

from simmod.modification.base_modifier import BaseModifier


def delayed_buffer_item(buffer_item, buffer_item_len, item):
    """Maintains delays using lists."""
    item_copy = copy.copy(item)
    if buffer_item is None:
        buffer_item = buffer_item_len * [item_copy]
    else:
        buffer_item.append(item_copy)
    item_cur = copy.copy(buffer_item.pop(0))

    return buffer_item, item_cur


class BuiltInModifier(BaseModifier):

    def __init__(self, sim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sim = sim

    @property
    def standard_setters(self) -> Dict:
        pass

    @property
    def names(self) -> List:
        pass

    def set_value(self, name: str, value):
        setattr(self.sim, name, value)


class ActionWrapper(gym.ActionWrapper):

    def __init__(self,
                 env,
                 noise_process: Optional[Callable] = None,
                 min_action_latency: int = 1,
                 max_action_latency: int = 3,
                 *args, **kwargs):
        super().__init__(env)

        self.action_shape = self.action_space.shape
        self._noise_process = np.random.normal if noise_process is None else noise_process
        self._buffer_actions = None
        self.min_action_latency = min_action_latency
        self.max_action_latency = max_action_latency
        self._buffer_actions_len = np.random.randint(min_action_latency, max_action_latency+1)  # in timesteps

    def _get_shape(self):
        raise self.action_shape

    @property
    def standard_setters(self) -> Dict:
        setters = {
            "noise": self.noise_step,
            "latency": self.latency_step,
            #"delay": self.delay_step,
            #"repetition": self.repetition_step,
        }
        return setters

    @property
    def names(self) -> List:
        return ["action"]

    def _get_noise(self):
        return self._noise_process(self.action_shape)

    def latency_step(self, action):
        if self._buffer_actions_len > 1:
            # Delay the actions.
            self._buffer_actions, action = delayed_buffer_item(
                self._buffer_actions,
                self._buffer_actions_len,
                action
            )

        return action

    def noise_step(self, action):
        return action + self._get_noise()

    def delay_step(self, action):
        raise NotImplementedError

    def repetition_step(self, action):
        raise NotImplementedError

    def reset(self, **kwargs):
        self._buffer_actions = None
        self._buffer_actions_len = np.random.randint(self.min_action_latency, self.max_action_latency + 1)
        return self.env.reset(**kwargs)

    def action(self, action):
        action = self.latency_step(action)
        action = self.noise_step(action)
        return action


class ObservationWrapper(gym.ObservationWrapper):

    def __init__(self, env,
                 noise_process: Optional[Callable] = None,
                 *args, **kwargs):
        super().__init__(env)
        self.observation_shape = self.observation_space.shape
        self._noise_process = np.random.normal if noise_process is None else noise_process

    @property
    def names(self) -> List:
        return ["observation"]

    @property
    def standard_setters(self) -> Dict:
        setters = {
            "noise": self.noise_step,
        }
        return setters

    def _get_noise(self):
        return self._noise_process(self.observation_shape)

    def observation(self, observation):
        return observation + self._get_noise()


class RewardWrapper(gym.RewardWrapper):
    pass



