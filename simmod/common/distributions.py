"""Defines noise distributions in a standardized way so that they can be replaced by each other in different methods
without changing other arguments to much.

TODO: Currently under construction
"""

from abc import ABC, abstractmethod
from simmod.utils.typings_ import *

import numpy as np


class Distribution(ABC):
    """Abstract base class for distributions."""

    def __init__(self):
        super(Distribution, self).__init__()

    @abstractmethod
    def proba_distribution_net(self, *args, **kwargs) -> Union[NDarray, Tuple[NDarray, NDarray]]:
        """Create the layers and parameters that represent the distribution.

        Subclasses must define this, but the arguments and return type vary between concrete classes.
        """

    @abstractmethod
    def proba_distribution(self, *args, **kwargs) -> "Distribution":
        """Set parameters of the distribution.

        Args:
            *args:
            **kwargs:

        Returns:
            self
        """

    @abstractmethod
    def log_prob(self, x: NDarray) -> NDarray:
        """Returns the log likelihood

        Args:
            x: The taken action

        Returns:
            The log likelihood of the distribution
        """

    @abstractmethod
    def entropy(self) -> Optional[NDarray]:
        """Shannon's entropy of the probability

        Returns:
            the entropy, or None if no analytical form is known
        """

    @abstractmethod
    def sample(self) -> NDarray:
        """Returns a sample from the probability distribution

        Returns:
            the stochastic action
        """

    @abstractmethod
    def mode(self) -> NDarray:
        """Returns the most likely action (deterministic output) from the probability distribution

        Returns:
            the stochastic action
        """

    def get_actions(self, deterministic: bool = False) -> NDarray:
        """Return actions according to the probability distribution.

        Args:
            deterministic:

        Returns:

        """
        if deterministic:
            return self.mode()
        return self.sample()

    @abstractmethod
    def actions_from_params(self, *args, **kwargs) -> NDarray:
        """Returns samples from the probability distribution given its parameters.

        Args:
            *args:
            **kwargs:

        Returns:

        """

    @abstractmethod
    def log_prob_from_params(self, *args, **kwargs) -> Tuple[NDarray, NDarray]:
        """Returns samples and the associated log probabilities from the probability distribution given its parameters.

        Args:
            *args:
            **kwargs:

        Returns:
            actions and log prob
        """


class UniformDistribution(Distribution):
    pass



class NormalDistribution(Distribution):
    pass


class BernoulliDistribution(Distribution):
    pass




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
        self.noise_prev = self.initial_noise if self.initial_noise is not None else np.zeros_like(self._mu)

    def __repr__(self) -> str:
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self._mu, self._sigma)


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
        return fmt.format(self.initial_stddev, self.desired_action_stddev, self.adoption_coefficient)