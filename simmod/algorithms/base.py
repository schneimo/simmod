from abc import ABC, abstractmethod

from simmod.common.parametrization import Parametrization
from simmod.modification.base_modifier import BaseModifier
from simmod.modification.builtin import ActionModifier, ObservationModifier, RewardModifier


class BaseAlgorithm(ABC):

    def __init__(self, *modifiers: BaseModifier, **kwargs):
        self.modifiers = modifiers  # List of modifiers

    @abstractmethod
    def _randomize_object(self, modifier: BaseModifier, instrumentation: Parametrization) -> None:
        pass

    def step(self) -> None:
        for modifier in self.modifiers:
            for instrumentation in modifier.instrumentation:
                self._randomize_object(modifier, instrumentation)
