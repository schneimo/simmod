from abc import ABC, abstractmethod

from simmod.common.parametrization import Parametrization
from simmod.modification.base_modifier import BaseModifier
from simmod.common.parametrization import Execution

import typing as tp

#EXECUTION_POINTS = tp.Union[Execution.BEFORE_STEP, Execution.AFTER_STEP, Execution.RESET]


class BaseAlgorithm(ABC):

    def __init__(self, *modifiers: BaseModifier, **kwargs):
        self.modifiers = modifiers  # List of modifiers
        self._current_values = {mod: {} for mod in modifiers}

    @abstractmethod
    def _randomize_object(self, modifier: BaseModifier, instrumentation: Parametrization, **kwargs) -> None:
        """Randomize the parameter of an object (both defined in the instrumentation variable) with
        the given modifier.

        Args:
            modifier:           Modifier to change the parameter defined in the instrumentation
            instrumentation:    Configuration of the parameter we want  change
            **kwargs:           Additional arguments for the setter function of the modifier

        Returns:                Return of the setter function
        """
        pass

    def _record_new_val(self, modifier, instrumentation, values):
        self._current_values[modifier].update({instrumentation.object_name: values})

    def get_current_val(self, modifier, instrumentation):
        return self._current_values[modifier][instrumentation.object_name]

    def step(self, **kwargs) -> None:
        """ Modify the pre-defined parameters of the simulation with each modifiers.

        Args:
            **kwargs:   Additional arguments for the parameter setter functions of the

        Returns:        Return of the setter functions
        """
        input = kwargs
        for modifier in self.modifiers:
            for instrumentation in modifier.instrumentation:
                input = self._randomize_object(modifier, instrumentation, **input)
        return input
