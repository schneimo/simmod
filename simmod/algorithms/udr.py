from inspect import signature
from typing import Any

import numpy as np

from simmod.algorithms.base import BaseAlgorithm
from simmod.common.parametrization import Parametrization
from simmod.modification.base_modifier import BaseModifier


class UniformDomainRandomization(BaseAlgorithm):

    def __init__(self, *modifiers: BaseModifier, random_state=None, **kwargs: Any) -> None:
        if random_state is None:
            self.random_state = np.random.RandomState()
        elif isinstance(random_state, int):
            # random_state assumed to be an int
            self.random_state = np.random.RandomState(random_state)
        else:
            self.random_state = random_state
        super().__init__(*modifiers, **kwargs)

    def _randomize_object(self, modifier: BaseModifier, instrumentation: Parametrization) -> None:
        object_name = instrumentation.object_name
        setter_func = modifier.standard_setters[instrumentation.setter]

        if setter_func.__defaults__ is not None:  # in case there are no kwargs
            n_kwargs = len(setter_func.__defaults__)
        else:
            n_kwargs = 0

        sig = signature(setter_func)
        n_params = len(
            sig.parameters) - n_kwargs - 1  # Exclude name & non-positional arguments # TODO: Randomize non-positional arguments
        lower_bound = instrumentation.lower_bound
        upper_bound = instrumentation.upper_bound
        val = list()
        assert len(lower_bound) == len(upper_bound)
        n = len(lower_bound)
        for _ in range(n_params):
            val.append(np.array([self.random_state.uniform(lower_bound[i], upper_bound[i]) for i in range(n)]))
        setter_func(object_name, *val)
