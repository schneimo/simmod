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


class _WelfordsAlgorithm:

    def __init__(self, mean_init: int = 0.):
        self.current_aggregate = (0, mean_init, 0.)

    def update(self, new_value):
        # For a new value newValue, compute the new count, new mean, the new M2.
        # mean accumulates the mean of the entire dataset
        # M2 aggregates the squared distance from the mean
        # count aggregates the number of samples seen so far
        (count, mean, M2) = self.current_aggregate
        count += 1
        delta_one = new_value - mean
        mean += delta_one / count
        delta_two = new_value - mean
        M2 += delta_one * delta_two
        self.current_aggregate = (count, mean, M2)
        return self.get_stats()

    # Retrieve the mean, variance and sample variance from an aggregate
    def get_stats(self):
        (count, mean, M2) = self.current_aggregate
        if count < 2:
            return float("nan")
        else:
            (mean, variance, sampleVariance) = (mean, M2 / count, M2 / (count - 1))
            return (mean, variance, sampleVariance)


class _Range:

    def __init__(self, init_high=0.0, fixed_high=False, init_low = 0.0, fixed_low=False):
        self.high = init_high
        self.low = init_low
        self.fixed_high = fixed_high
        self.fixed_low = fixed_low

    def update(self, new_value):
        if self.fixed_low and self.fixed_high:
            self.high = new_value if self.high < new_value else self.high
            self.low = -self.high
        elif self.fixed_high:
            self.low = new_value if self.low > new_value else self.low
        elif self.fixed_low:
            self.high = new_value if self.high < new_value else self.high

    def get_stats(self):
        return self.high - self.low


class _Noise():

    def __init__(self,
                 space,
                 noise_process: Optional[Callable] = None,
                 noise_scale: Optional[List, float] = 1.,
                 noise_baseline: Optional[str] = 'range',
                 dtype=np.float32,
                 *args, **kwargs):
        self.dtype = np.dtype(dtype)
        self.shape = space.shape
        self._noise_process = np.random.normal if noise_process is None else noise_process

        if np.isscalar(noise_scale):
            self.noise_scale = np.full(self.shape, noise_scale, dtype=dtype)
        else:
            assert np.array(noise_scale).shape == self.shape
            self.noise_scale = np.array(noise_scale)

        self._noise_range = []
        for low, high in zip(space.low, space.high):
            fixed_high = high != np.inf
            fixed_low = low != -np.inf
            low = 0 if not fixed_low else low
            high = 0 if not fixed_high else high
            self._noise_range.append(_Range(high, fixed_high, low, fixed_low))

    def _update(self, values):
        values = [values] if np.isscalar(values) else values
        for idx, range in enumerate(self._noise_range):
            range.update(values[idx])

    def _get_noise(self):
        half_range = [r.get_stats()/2 for r in self._noise_range]
        noise = np.array([self._noise_process(scale=s) for s in half_range], dtype=self.dtype)
        assert noise.shape == self.shape
        return self.noise_scale * noise


class ActionWrapper(gym.ActionWrapper, _Noise):

    def __init__(self,
                 env: gym.Env,
                 noise_process: Optional[Callable] = None,
                 noise_scale: Optional[List, float] = 1.,
                 noise_baseline: Optional[str] = 'range',
                 min_action_latency: int = 1,
                 max_action_latency: int = 3,
                 dtype=np.float32,
                 *args, **kwargs):
        super(gym.ActionWrapper).__init__(env)
        super(_Noise).__init__(self.action_space, noise_process, noise_scale, noise_baseline, dtype, *args,
                               **kwargs)
        self.dtype = np.dtype(dtype)

        self.min_action_latency, self.max_action_latency = min_action_latency, max_action_latency
        self._noise_process = np.random.normal if noise_process is None else noise_process
        self._buffer_actions = None
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

    def latency_step(self, action):
        if self._buffer_actions_len > 1:
            # Delay the actions.
            self._buffer_actions, action = delayed_buffer_item(
                self._buffer_actions,
                self._buffer_actions_len,
                action
            )

        return action

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
        self._update(action)
        action += self._get_noise()
        return action


class ObservationWrapper(gym.ObservationWrapper, _Noise):

    def __init__(self,
                 env: gym.Env,
                 noise_process: Optional[Callable] = None,
                 noise_scale: Optional[List, float] = 1.,
                 noise_baseline: Optional[str] = 'range',
                 dtype=np.float32,
                 *args, **kwargs):
        super(gym.ObservationWrapper).__init__(env)
        super(_Noise).__init__(self.observation_space, noise_process, noise_scale, noise_baseline, dtype, *args, **kwargs)
        self.dtype = np.dtype(dtype)

    @property
    def names(self) -> List:
        return ["observation"]

    def observation(self, observation):
        self._update(observation)
        return observation + self._get_noise()


class RewardWrapper(gym.RewardWrapper):
    pass



