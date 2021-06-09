from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class Noise(ABC):
    """The action noise base class"""

    def __init__(self):
        super(Noise, self).__init__()

    def reset(self) -> None:
        """Call end of episode reset for the noise"""
        pass

    @abstractmethod
    def __call__(self) -> np.ndarray:
        raise NotImplementedError()


class NormalNoise(Noise):
    """Gaussian action noise"""

    def __init__(self, mean, sigma, shape):
        """A Gaussian action noise

        Args:
            mean: (float) the mean value of the noise
            sigma: (float) the scale of the noise (std here)
            shape:
        """
        super().__init__()
        self._mu = mean
        self._sigma = sigma
        self.shape = shape

    def __call__(self) -> np.ndarray:
        return np.random.normal(self._mu, self._sigma, size=self.shape)

    def __repr__(self) -> str:
        return 'NormalActionNoise(mu={}, sigma={})'.format(self._mu, self._sigma)


class OrnsteinUhlenbeckNoise(Noise):
    """Ornstein Uhlenbeck noise. Designed to approximate brownian motion with friction.

    Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
    """

    def __init__(self, mean: float, sigma: float, theta: float = .15, dt: float = 1e-2,
                 initial_noise: Optional[float] = None):
        """

        Args:
            mean: (float) the mean of the noise
            sigma: (float) the scale of the noise
            theta: (float) the rate of mean reversion
            dt: (float) the timestep for the noise
            initial_noise: ([float]) the initial value for the noise output, (if None: 0)
        """
        super().__init__()
        self._theta = theta
        self._mu = mean
        self._sigma = sigma
        self._dt = dt
        self.initial_noise = initial_noise
        self.noise_prev = None
        self.reset()

    def __call__(self) -> np.ndarray:
        noise = self.noise_prev + self._theta * (self._mu - self.noise_prev) * self._dt + \
                self._sigma * np.sqrt(self._dt) * np.random.normal(size=self._mu.shape)
        self.noise_prev = noise
        return noise

    def reset(self) -> None:
        """Reset the Ornstein Uhlenbeck noise, to the initial position"""
        self.noise_prev = self.initial_noise if self.initial_noise is not None \
            else np.zeros_like(self._mu)

    def __repr__(self) -> str:
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(
            self._mu, self._sigma)


class AdaptiveNoise(Noise):
    """Adaptive parameter noise"""

    def __init__(self, initial_stddev=0.1, desired_action_stddev=0.1, adoption_coefficient=1.01):
        """

        Args:
            initial_stddev: (float) the initial value for the standard deviation of the noise
            desired_action_stddev: (float) the desired value for the standard deviation of the noise
            adoption_coefficient: (float) the update coefficient for the standard deviation of the noise
        """
        super().__init__()
        self.initial_stddev = initial_stddev
        self.desired_action_stddev = desired_action_stddev
        self.adoption_coefficient = adoption_coefficient

        self.current_stddev = initial_stddev

    def adapt(self, distance):
        """Update the standard deviation for the parameter noise

        Args:
            distance: (float) the noise distance applied to the parameters
        """
        if distance > self.desired_action_stddev:
            # Decrease stddev.
            self.current_stddev /= self.adoption_coefficient
        else:
            # Increase stddev.
            self.current_stddev *= self.adoption_coefficient

    def get_stats(self):
        """Return the standard deviation for the parameter noise

        Returns:
            (dict) the stats of the noise
        """
        return {'param_noise_stddev': self.current_stddev}

    def __repr__(self):
        fmt = 'AdaptiveParamNoiseSpec(initial_stddev={}, desired_action_stddev={}, adoption_coefficient={})'
        return fmt.format(self.initial_stddev, self.desired_action_stddev,
                          self.adoption_coefficient)
