from abc import ABC
from typing import Union, List, Callable, AnyStr, Tuple, Any, Dict, Optional
from enum import Enum, auto

import numpy as np

Num = Union[int, float]
Array = Union[np.ndarray, List]
ArrayOrNum = Union[Array, Num]


# TODO: How can we get this into JSON?
# TODO: Algorithms as Wrappers?

class Execution(Enum):
    BEFORE_STEP = auto()
    AFTER_STEP = auto()
    RESET = auto()


class Parametrization:
    def __init__(
            self,
            setter: AnyStr,
            object_name: AnyStr,
            parameter_range: Array,
            execution: AnyStr,
            parameter_type: Optional[object] = None,
            shape: Optional[Tuple] = None,
            name: Optional[AnyStr] = None
    ) -> None:
        self.setter = setter
        self.object_name = object_name
        self.execution = Execution[execution]
        if shape is not None:
            assert len(parameter_range) == 2 and shape[-1] == 2, \
                "If a shape is given the last dimension and the length of the range must match 2 (lower and higher bound)"
            self.shape = shape
            parameter_range = np.tile(parameter_range, (shape[0], 1))
        else:
            if not isinstance(parameter_range, np.ndarray):
                parameter_range = np.asarray(parameter_range)
            self.shape = parameter_range.shape
        self.lower_bound = parameter_range.T[0]
        self.upper_bound = parameter_range.T[1]
        self.name = name

    def get_json(self) -> Dict:
        result = {
            self.setter: {
                self.object_name: 0
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
            init: ArrayOrNum = None,
            lower: ArrayOrNum = None,
            upper: ArrayOrNum = None
    ) -> None:
        self.lower = lower
        self.upper = upper
        super().__init__(mod_func, init=init)


class Scalar(Array):

    def __init__(
            self,
            mod_func: Callable,
            init: Num = None,
            lower: Num = 0,
            upper: Num = 1
    ) -> None:
        super().__init__(mod_func, init=np.array([init]), lower=np.array([lower]), upper=np.array([upper]))


class Boolean(Parameter):

    def __init__(
            self,
            mod_func: Callable,
            init: bool = True,
    ) -> None:
        super().__init__(mod_func, init=init)
