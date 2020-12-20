"""
Copyright (c) 2020, Moritz Schneider
@Author: Moritz Schneider
"""
from typing import List, Dict, Optional, Callable, Tuple, Union

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

    def __init__(self, init_high: float = 0.0, bounded_high: bool = False, init_low: float = 0.0,
                 bounded_low: bool = False, reset_after: Optional[int] = None):
        self.low, self.high = init_low, init_high
        self.bounded_low, self.bounded_high = bounded_low, bounded_high
        self.unbounded = (not bounded_low) and (not bounded_high)
        self.reset_after = reset_after if reset_after is not None else np.inf
        self.count = 0

    def update(self, new_value):
        self.count += 1
        if self.count >= self.reset_after:
            self.count = 0

        if self.unbounded:
            self.high = abs(new_value) if self.high < abs(new_value) else self.high
            self.low = -self.high
        elif self.bounded_high:
            self.low = new_value if self.low > new_value else self.low
        elif self.bounded_low:
            self.high = new_value if self.high < new_value else self.high

    def get_stats(self):
        return self.high - self.low


class _NoiseWrapper(gym.Wrapper):

    def __init__(self,
                 env,
                 space,
                 noise_process: Optional[Callable] = None,
                 noise_scale: Union[List[float], float] = 1.0,
                 noise_baseline: Union[str, List[float]] = 'range',
                 dtype=np.float32,
                 *args, **kwargs):
        super().__init__(env)
        self.dtype = np.dtype(dtype)
        self.shape = space.shape
        self._noise_process = np.random.uniform if noise_process is None else noise_process

        if np.isscalar(noise_scale):
            self.noise_scale = np.full(self.shape, noise_scale, dtype=dtype)
        else:
            assert np.array(noise_scale).shape == self.shape
            self.noise_scale = np.array(noise_scale)

        self._noise_range = []
        if isinstance(noise_baseline, str):
            for low, high in zip(space.low, space.high):
                bounded_high = high < np.inf
                bounded_low = low > -np.inf
                low = low if bounded_low else 0.0
                high = high if bounded_high else 0.0
                if noise_baseline == 'range':
                    self._noise_range.append(_Range(high, bounded_high, low, bounded_low))
        elif isinstance(noise_baseline, list):
            assert len(noise_baseline) == len(space.low)
            for val in noise_baseline:
                self._noise_range.append(_Range(val, True, -val, True))
        self._setup_env_metadata()

    def reset(self, **kwargs):
        self.metadata['randomization.parameter_value'] = {f'wrapper:{type(self).__name__}': []}
        return super().reset(**kwargs)

    def _setup_env_metadata(self):
        range_values = {f'wrapper:{type(self).__name__}:noise_scale': self.noise_scale}
        self.metadata['randomization.parameter_range'] = range_values

    def _update_env_metadata(self, noise):
        self.metadata['randomization.parameter_value'][f'wrapper:{type(self).__name__}'].append(noise)

    def _update(self, values):
        values = [values] if np.isscalar(values) else values
        #for idx, range in enumerate(self._noise_range):
        #    range.update(values[idx])

    def _get_noise(self):
        values = [r.get_stats() for r in self._noise_range]
        #noise = np.array([self._noise_process(loc=0.0, scale=s) for s in values], dtype=self.dtype)
        noise = np.array([self._noise_process(low=-s, high=s) for s in values], dtype=self.dtype)
        assert noise.shape == self.shape
        self._update_env_metadata(noise)
        return self.noise_scale * noise


class ActionWrapper(_NoiseWrapper):

    def __init__(self,
                 env: gym.Env,
                 noise_process: Optional[Callable] = None,
                 noise_scale: Union[List[float], float] = 1.,
                 noise_baseline: Union[str, List[float]] = 'range',
                 min_action_latency: int = 1,
                 max_action_latency: int = 5,
                 dtype=np.float32,
                 *args, **kwargs):
        super().__init__(env, env.action_space, noise_process, noise_scale, noise_baseline, dtype, *args, **kwargs)
        self.dtype = np.dtype(dtype)

        self.min_action_latency, self.max_action_latency = min_action_latency, max_action_latency
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
        return super().reset(**kwargs)

    def step(self, action):
        return self.env.step(self.action(action))

    def action(self, action):
        action = self.latency_step(action)
        self._update(action)
        action += self._get_noise()
        return action


class ObservationWrapper(_NoiseWrapper):

    def __init__(self,
                 env: gym.Env,
                 noise_process: Optional[Callable] = None,
                 noise_scale: Union[List[float], float] = 1.,
                 noise_baseline: Union[str, List[float]] = 'range',
                 dtype=np.float32,
                 *args, **kwargs):
        super().__init__(env, env.observation_space, noise_process, noise_scale, noise_baseline, dtype, *args, **kwargs)
        self.dtype = np.dtype(dtype)

    @property
    def names(self) -> List:
        return ["observation"]

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return self.observation(observation), reward, done, info

    def observation(self, observation):
        self._update(observation)
        return observation + self._get_noise()


class RewardWrapper(gym.RewardWrapper):
    pass



