from abc import ABC, abstractmethod

from simmod.common.parametrization import Parametrization
from simmod.modification.base_modifier import BaseModifier
from simmod.common.parametrization import Execution

import typing as tp

#EXECUTION_POINTS = tp.Union[Execution.BEFORE_STEP, Execution.AFTER_STEP, Execution.RESET]


class BaseAlgorithm(ABC):

    def __init__(self, *modifiers: BaseModifier, **kwargs):
        self.modifiers = modifiers  # List of modifiers

    @abstractmethod
    def _randomize_object(self, modifier: BaseModifier, instrumentation: Parametrization, **kwargs) -> None:
        pass

    def step(self, **kwargs) -> None:
        input = kwargs
        for modifier in self.modifiers:
            for instrumentation in modifier.instrumentation:
                input = self._randomize_object(modifier, instrumentation, **input)
        return input
