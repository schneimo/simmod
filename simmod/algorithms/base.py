from abc import ABC, abstractmethod

from simmod.common.parametrization import Parametrization
from simmod.modification.base_modifier import BaseModifier
from simmod.common.parametrization import Execution

import typing as tp

EXECUTION_POINTS = tp.Union[Execution]


class BaseAlgorithm(ABC):

    def __init__(self, *modifiers: BaseModifier, **kwargs):
        self.modifiers = modifiers  # List of modifiers

    @abstractmethod
    def _randomize_object(self, modifier: BaseModifier, instrumentation: Parametrization, **kwargs) -> None:
        pass

    def step(self, execution: EXECUTION_POINTS = 'RESET', **kwargs) -> None:
        input = kwargs
        for modifier in self.modifiers:
            for instrumentation in modifier.instrumentation:
                if input is None:
                    input = self._randomize_object(modifier, instrumentation)
                else:
                    input = self._randomize_object(modifier, instrumentation, **dict(input))
        return input
