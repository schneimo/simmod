from typing import List, Dict, Optional, Callable, Tuple

import gym
import copy
import numpy as np

from simmod.modification.base_modifier import BaseModifier


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


class ActionModifier(BaseModifier):

    def __init__(self,
                 sim,
                 noise_process: Optional[Callable] = None,
                 action_shape: Optional[Tuple] = None,
                 action_latency: int = 0,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sim = sim
        if action_shape is not None:
            self.act_shape = action_shape
        elif isinstance(self.sim, gym.Env):
            if isinstance(self.sim.action_space, gym.spaces.Box):
                self.act_shape = self.sim.action_space.shape
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        self._noise_process = np.random.normal if noise_process is None else noise_process
        self._buffer_actions = None
        self._buffer_actions_len = action_latency # in timesteps

    def _get_shape(self):
        raise self.act_shape

    @property
    def standard_setters(self) -> Dict:
        setters = {
            "noise": self.noise_step,
            "latency": self.latency_step,
            "delay": self.delay_step,
            "repetition": self.repetition_step,
        }
        return setters

    @property
    def names(self) -> List:
        return ["action"]

    def _get_noise(self):
        return self._noise_process(self.act_shape)

    def delayed_buffer_item(self, buffer_item, buffer_item_len, item):
        """Maintains delays using lists."""
        item_copy = copy.copy(item)
        if buffer_item is None:
            buffer_item = buffer_item_len * [item_copy]
        else:
            buffer_item.append(item_copy)
        item_cur = copy.copy(buffer_item.pop(0))

        return buffer_item, item_cur

    def latency_step(self, action):
        if self._buffer_actions_len > 1:
            # Delay the actions.
            self._buffer_actions, action = self.delayed_buffer_item(self._buffer_actions,
                                                                    self._buffer_actions_len,
                                                                    action)
        return action

    def noise_step(self, action):
        return action + self._get_noise()

    def delay_step(self, action):
        raise NotImplementedError

    def repetition_step(self, action):
        raise NotImplementedError


class ObservationModifier(BaseModifier):

    def __init__(self, sim,
                 noise_process: Optional[Callable] = None,
                 observation_shape: Optional[Tuple] = None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sim = sim
        if observation_shape is not None:
            self.obs_shape = observation_shape
        elif isinstance(self.sim, gym.Env):
            if isinstance(self.sim.observation_space, gym.spaces.Box):
                self.obs_shape = self.sim.observation_space.shape
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

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
        return self._noise_process(self.obs_shape)

    def noise_step(self, observation):
        return observation + self._get_noise()


class RewardModifier(BaseModifier):
    pass



