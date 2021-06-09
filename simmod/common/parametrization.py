"""The parameterization is the connection between the setter functions, the simulation objects and their respective
value ranges. All those variables are stored in an individual parameterization instance.
"""
from abc import ABC
from enum import Enum, auto
from typing import Union, List, Callable, AnyStr, Tuple, Any, Dict, Optional, Sequence

import numpy as np

from simmod.utils.typings_ import *


class Execution(Enum):
    BEFORE_STEP = auto()
    AFTER_STEP = auto()
    RESET = auto()


class Operation(Enum):
    ADDITIVE = auto()
    SCALING = auto()
    REPLACING = auto()


class Parametrization:

    def __init__(
            self,
            setter: AnyStr,
            object_name: AnyStr,
            parameter_range: Array,
            distribution: Optional[object],
            operation: Optional[AnyStr],
            execution: AnyStr = None,
            shape: Optional[Tuple] = None,
            name: Optional[AnyStr] = None
    ) -> None:
        self.setter = setter
        self.object_name = object_name
        self.distribution = distribution
        self.operation = Operation[operation.upper()]
        self.execution = Execution[execution.upper()] if execution is not None \
            else Execution.RESET

        if shape is not None:
            assert len(parameter_range) == 2 and shape[-1] == 2, \
                "If a shape is given the last dimension and the length of " \
                "the range must match 2 (lower and higher bound)"
            self.shape = shape
            parameter_range = np.tile(parameter_range, (shape[0], 1))
        else:
            if not isinstance(parameter_range, np.ndarray):
                parameter_range = np.asarray(parameter_range)
            self.shape = parameter_range.shape
        self._values_one = parameter_range.T[0]
        self._values_two = parameter_range.T[1]
        self.name = name
        self.history = []
        self.current_val = None

    def __str__(self):
        return f'{self.setter}:{self.object_name}={self.current_val}'

    def sample(self, n_params,
               random_state: Optional[RandomStateOrGenerator] = None):

        if random_state is None:
            random_state = np.random.default_rng()

        a, b = self._values_one, self._values_two
        new_values = list()
        assert len(a) == len(b)
        n = len(a)
        for _ in range(n_params):
            dist = self.distribution
            if dist == 'uniform':
                values = np.array(
                    [random_state.uniform(a[i], b[i]) for i in range(n)])
            elif dist == 'normal' or dist == 'gaussian':
                values = np.array(
                    [random_state.normal(a[i], b[i]) for i in range(n)])
            elif dist == 'loguniform':
                values = np.exp(
                    [random_state.uniform(a[i], b[i]) for i in range(n)])
            else:
                raise ValueError(
                    f"Distribution type '{dist}' not available, use 'uniform', "
                    f"'loguniform' or 'normal'")
            new_values.append(values)
        return new_values

    @property
    def entropy(self) -> float:
        a, b = self._values_one, self._values_two
        if self.distribution == "uniform":
            return float(np.sum(np.log(b - a)))
        elif self.distribution == "normal" or self.distribution == "gaussian":
            return float(np.sum(0.5 * (1 + np.log(2 * np.pi * b ** 2))))
        elif self.distribution == "loguniform":
            return float(np.sum(0.5 * np.log(a*b) + np.log(np.log(b*1.0 / a))))
        else:
            raise ValueError(f"Unknown distribution type '{self.distribution}'")

    @property
    def parameter_values(self) -> Tuple[float, float]:
        return (self._values_one, self._values_two)

    def update(self, new_values, **kwargs) -> None:
        if self.current_val is not None:
            self.history.append(self.current_val)
        self.current_val = new_values

    def get_json(self) -> Dict:
        result = {
            self.setter: {
                self.object_name: self.current_val
            }
        }
        return result


class Parameter(ABC):

    def __init__(
            self,
            mod_func: Callable,
            init: Any
    ) -> None:
        self.mod_func = mod_func
        self.init = init


class Array(Parameter):

    def __init__(
            self,
            mod_func: Callable,
            init: Optional[ArrayOrNum] = None,
            lower: Optional[ArrayOrNum] = None,
            upper: Optional[ArrayOrNum] = None
    ) -> None:
        self.lower = lower
        self.upper = upper
        super().__init__(mod_func, init=init)


class Scalar(Array):

    def __init__(
            self,
            mod_func: Callable,
            init: Optional[Num] = None,
            lower: Num = 0,
            upper: Num = 1
    ) -> None:
        super().__init__(mod_func,
                         init=np.array([init]), lower=np.array([lower]),
                         upper=np.array([upper]))


class Boolean(Parameter):

    def __init__(
            self,
            mod_func: Callable,
            init: bool = True,
    ) -> None:
        super().__init__(mod_func, init=init)
