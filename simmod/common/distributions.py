"""Defines noise distributions in a standardized way so that they can be replaced by each other in different methods
without changing other arguments to much.

TODO: Currently under construction
"""

from abc import ABC, abstractmethod
from simmod.utils.typings_ import *


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
