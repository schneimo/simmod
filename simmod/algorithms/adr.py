"""Automatic Domain Randomization (ADR) algorithm

Introduced in:
Akkaya, Ilge, et al. "Solving rubik's cube with a robot hand."
arXiv preprint arXiv:1910.07113 (2019).
"""
import random
from inspect import signature
from typing import Any, AnyStr, Union, Sequence, Optional
import numpy as np

from collections import OrderedDict

from simmod.algorithms.udr import UniformDomainRandomization
from simmod.modification.base_modifier import BaseModifier
from simmod.common.parametrization import Parametrization
from simmod.common.parametrization import Execution

EXECUTION_POINTS = Union[Execution]


class AutomaticDomainRandomization(UniformDomainRandomization):

    def __init__(self, *modifiers: BaseModifier, random_state: Optional[
        np.random.Generator] = None,
                 buffer_threshold, performance_thresholds: Sequence,
                 step_size, **kwargs:
        Any) -> \
            None:

        if len(performance_thresholds) > 2 or performance_thresholds[0] > \
                performance_thresholds[1]:
            raise ValueError("'performance_thresholds' should be Tuple "
                             "containing two values whereas the first "
                             "corresponds to the lower threshold t_L and the "
                             "second to the upper threshold t_H (t_L < t_H)")

        if random_state is None:
            self.random_state = np.random.default_rng()
        elif isinstance(random_state, int):
            # random_state assumed to be an int
            self.random_state = np.random.RandomState(random_state)
        else:
            self.random_state = random_state
        super().__init__(*modifiers, random_state=random_state, **kwargs)

        buffer = OrderedDict()

        for modifier in self.modifiers:
            buffer[modifier] = OrderedDict()
            for instrumentation in modifier.instrumentation:
                buffer[modifier][instrumentation] = ([], [])

    def _bound_value(self, modifier: BaseModifier, instrumentation: Parametrization,
                     bound_low: bool):
        a, b = instrumentation.parameter_values
        object_name = instrumentation.object_name
        setter_func = modifier.standard_setters[instrumentation.setter]

        if setter_func.__defaults__ is not None:  # in case there are no kwargs
            n_kwargs = len(setter_func.__defaults__)
        else:
            n_kwargs = 0

        sig = signature(setter_func)
        n_params = len(
            sig.parameters) - n_kwargs - 1  # Exclude name & non-positional arguments
        # TODO: Randomize non-positional arguments

        new_values = instrumentation.sample(n_params)

        if bound_low:
            new_values[0] = a
        else:
            new_values[1] = b

        instrumentation.update(new_values)
        return setter_func(object_name, *new_values)

    def adapt_boundaries(self, instrumentation: Parametrization,
                              step_size: float, select_low: bool):
        pass

    def entropy(self):
        n = 0
        entropy = 0
        for modifier in self.modifiers:
            for instrumentation in modifier.instrumentation:
                entropy += instrumentation.entropy
                n += 1
        assert n != 0
        return entropy / n

    def step(self, execution: EXECUTION_POINTS = 'RESET', **kwargs) -> None:
        mod = random.choice(self.modifiers)
        bounded_param = random.choice(mod.instrumentation)

        x = self.random_state.uniform()
        select_low = (x < 0.5)

        for modifier in self.modifiers:
            for instrumentation in modifier.instrumentation:
                if instrumentation is bounded_param:
                    self._bound_value(modifier, instrumentation, select_low)
                else:
                    self._randomize_object(modifier, instrumentation)
