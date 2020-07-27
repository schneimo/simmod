from abc import ABC, abstractmethod

from typing import Union, AnyStr

from simmod.common.parametrization import Parametrization
from simmod.modification.base_modifier import BaseModifier


class BaseAlgorithm(ABC):

    def __init__(self, *args: BaseModifier, **kwargs):
        self.modifiers = args    # List of modifiers
        #self.object_names = [i.object_name for i in m.instrumentation for m in self.modifiers]

    @abstractmethod
    def _randomize_object(self, modifier: BaseModifier, instrumentation: Parametrization) -> None:
        pass

    def step(self) -> None:
        for modifier in self.modifiers:
            for instrumentation in modifier.instrumentation:
                self._randomize_object(modifier, instrumentation)
