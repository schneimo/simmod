"""Additional OpenAI Gym wrapper to change received observations, rewards and actions.

Copyright (c) 2021, Moritz Schneider
@Author: Moritz Schneider
"""
from typing import List, Optional, Callable, Union, Tuple, Any

import gym
import copy
import numpy as np


def delayed_buffer_item(buffer_item: List, buffer_item_len: int, item: Any):
    """Maintains delays using lists."""
    item_copy = copy.copy(item)
    if buffer_item is None:
        buffer_item = buffer_item_len * [item_copy]
    else:
        buffer_item.append(item_copy)
    item_cur = copy.copy(buffer_item.pop(0))

    return buffer_item, item_cur


class ValueLoggerWrapper(gym.Wrapper):

    def __init__(self, env: gym.Env, ):
        super(ValueLoggerWrapper, self).__init__(env)


class _WelfordsAlgorithm:
    """Implementation of the Welford's algorithm for calculating the variance on the go.

    More information: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    """

    def __init__(self, mean_init: int = 0.):
        self.current_aggregate = (0, mean_init, 0.)

    def update(self, new_value: float) -> Union[Tuple, np.ndarray, float]:
        """Update the current data aggregate with the new value.

        For a new value `new_value`, compute the new `count`, new `mean`, the new `M2`.
        - `mean` accumulates the mean of the entire dataset
        - `M2` aggregates the squared distance from the mean
        - `count` aggregates the number of samples seen so far

        Args:
            new_value: The value which should be integrated into the data

        Returns:
            The new aggregate as a tuple.
        """
        (count, mean, M2) = self.current_aggregate
        count += 1
        delta_one = new_value - mean
        mean += delta_one / count
        delta_two = new_value - mean
        M2 += delta_one * delta_two
        self.current_aggregate = (count, mean, M2)
        return self.get_stats()

    def get_stats(self) -> Union[Tuple, np.ndarray, float]:
        """Retrieve the mean, variance and sample variance from an aggregate."""
        count, mean, M2 = self.current_aggregate
        if count < 2:
            return float("nan")
        else:
            (mean, var, sample_var) = (mean, M2 / count, M2 / (count - 1))
            return mean, var, sample_var


class _Range:
    """Class to store the range of values.

    The upper and lower bound of the range can be either set directly (bounded) or can be set as unbounded. In the
    latter case the upper and/or lower bound gets updated in every update step
    """

    def __init__(self, init_high: float = 0.0, bounded_high: bool = False, init_low: float = 0.0,
                 bounded_low: bool = False, reset_after: Optional[int] = None):
        """Initialize the range.

        Args:
            init_high: Initializing value of the upper bound.
            bounded_high: Specify is the upper bound is fixed.
            init_low: Initializing value of the lower bound.
            bounded_low: Specify is the lower bound is fixed
            reset_after: Reset the values of the upper and lower bound after resetting the environment
        """
        self.low, self.high = init_low, init_high
        self.bounded_low, self.bounded_high = bounded_low, bounded_high
        self.unbounded = (not bounded_low) and (not bounded_high)
        self.reset_after = reset_after if reset_after is not None else np.inf
        self.count = 0

    def update(self, new_value: float) -> Union[np.ndarray, float]:
        """Update the range if it is unbounded.

        If the absolute of the new value is greater than the lower and/or upper bound, the corresponding value is#
        updated accordingly.

        Args:
            new_value: The value which should be integrated into the data
        """
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
        return self.high - self.low

    def get_stats(self) -> Union[np.ndarray, float]:
        return self.high - self.low


class _NoiseWrapper(gym.Wrapper):
    """Wrapper to apply noise to the given input"""

    def __init__(self,
                 env: gym.Env,
                 space: gym.spaces.Box,
                 noise_process: Optional[Callable] = None,
                 noise_scale: Union[List[float], float] = 1.0,
                 noise_baseline: Union[str, List[float]] = 'range',
                 dtype: Optional = np.float32,
                 *args, **kwargs):
        """Creates the wrapper.

        Args:
            env: OpenAI Gym environment
            space: Relevant space of the environment to calculate noise for
            noise_process: Noise process (at the moment only np.uniform)
            noise_scale: Scaling variable to multiplicate the noise with
            noise_baseline: Metric on which the noise calculation will be based (i.e. range, variance)
            dtype: Numpy dtype for the noise matrix
        """
        super().__init__(env)
        self.dtype = np.dtype(dtype)
        self.shape = space.shape
        self._noise_process = np.random.uniform if noise_process is None else noise_process  # TODO: Allow more types

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
                # TODO: More baseline types, i.e. calculated by the Welford algorithm
                if noise_baseline == 'range':
                    self._noise_range.append(_Range(high, bounded_high, low, bounded_low))
        elif isinstance(noise_baseline, list):
            assert len(noise_baseline) == len(space.low)
            for val in noise_baseline:
                self._noise_range.append(_Range(val, True, -val, True))
        self.key_name = f'wrapper:{type(self).__name__}'
        self._setup_env_metadata()

    def reset(self, **kwargs):
        self.metadata['randomization.parameter_value'].update({self.key_name: []})
        return super().reset(**kwargs)

    def _setup_env_metadata(self):
        if 'randomization.parameter_range' not in self.metadata.keys():
            self.metadata['randomization.parameter_range'] = dict()
        if 'randomization.parameter_value' not in self.metadata.keys():
            self.metadata['randomization.parameter_value'] = dict()

        range_values = {f'{self.key_name}:noise_scale': self.noise_scale}
        self.metadata['randomization.parameter_range'].update(range_values)
        self.metadata['randomization.parameter_value'].update({self.key_name: []})

    def _update_env_metadata(self, noise: np.ndarray):
        self.metadata['randomization.parameter_value'][self.key_name].append(noise)

    def _get_noise(self):
        values = [r.get_stats() for r in self._noise_range]
        noise = np.array([self._noise_process(low=-s, high=s) for s in values], dtype=self.dtype)
        assert noise.shape == self.shape
        self._update_env_metadata(noise)
        return self.noise_scale * noise


class ActionWrapper(_NoiseWrapper):
    """Change calculated actions after prediction before they are passed to the main environment.

    Noise can be applied by specifying a Numpy distribution (at the moment only np.uniform) at the argument
    `noise_process`.

    Actions can be delayed by specifying `min_action_latency` and `max_action_latency`. The amount of latency steps are
    determined by a uniform distribution after each reset of the wrapped environment and are fixed during the episode.
    If the amount of delay should be fixed over all episodes, `min_action_latency` and `max_action_latency` should be
    set to the same value.
    If no latency should be applied `min_action_latency` and `max_action_latency` should be set to 1.

    Repetition of actions is not supported at the moment.
    """

    def __init__(self,
                 env: gym.Env,
                 noise_process: Optional[Callable] = None,
                 noise_scale: Union[List[float], float] = 1.,
                 noise_baseline: Union[str, List[float]] = 'range',
                 min_action_latency: int = 1,
                 max_action_latency: int = 5,
                 dtype: np.number = np.float32,
                 *args, **kwargs):
        """Creates the wrapper and wraps it around the given environment.

        Args:
            env: OpenAI Gym environment
            space: Relevant space of the environment to calculate noise for
            noise_process: Noise process (at the moment only np.uniform)
            noise_scale: Scaling scalar or vector to multiplicate the noise with
            noise_baseline: Metric on which the noise calculation will be based (i.e. range, variance)
            min_action_latency: Minimum number of timesteps the action can be delayed; 1 is no delay
            max_action_latency: Maximum number of timesteps the action can be delayed
            dtype: Numpy dtype for the noise matrix
        """
        super().__init__(env, env.action_space, noise_process, noise_scale, noise_baseline, dtype, *args, **kwargs)
        self.dtype = np.dtype(dtype)

        self.min_action_latency, self.max_action_latency = min_action_latency, max_action_latency
        self._buffer_actions = None
        self._buffer_actions_len = np.random.randint(min_action_latency, max_action_latency+1)  # in timesteps

    def _get_shape(self):
        raise self.action_shape

    @property
    def names(self) -> List:
        return ["action"]

    def latency_step(self, action):
        if self._buffer_actions_len > 1:
            # Delay the actions.
            self._buffer_actions, action = delayed_buffer_item(self._buffer_actions, self._buffer_actions_len, action)
        return action

    def repetition_step(self, action):
        # TODO: Implementation of action repetition
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
    """Change observations before they are passed to a policy.

    Noise can be applied by specifying a Numpy distribution (at the moment only np.uniform) at the argument
    `noise_process`.
    """

    def __init__(self,
                 env: gym.Env,
                 noise_process: Optional[Callable] = None,
                 noise_scale: Union[List[float], float] = 1.,
                 noise_baseline: Union[str, List[float]] = 'range',
                 dtype: np.number = np.float32,
                 *args, **kwargs):
        """Creates the wrapper and wraps it around the given environment.

        Args:
            env: OpenAI Gym environment
            space: Relevant space of the environment to calculate noise for
            noise_process: Noise process (at the moment only np.uniform)
            noise_scale: Scaling scalar or vector to multiplicate the noise with
            noise_baseline: Metric on which the noise calculation will be based (i.e. range, variance)
            dtype: Numpy dtype for the noise matrix
        """
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



