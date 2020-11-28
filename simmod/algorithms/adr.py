from typing import Any, AnyStr, Union
import numpy as np

from simmod.algorithms.base import BaseAlgorithm
from simmod.modification.base_modifier import BaseModifier
from simmod.common.parametrization import Parametrization
from simmod.common.parametrization import Execution

EXECUTION_POINTS = Union[Execution]


class AutomaticDomainRandomization(BaseAlgorithm):

    def __init__(self, *modifiers: BaseModifier, random_state=None, **kwargs: Any) -> None:
        if random_state is None:
            self.random_state = np.random.RandomState()
        elif isinstance(random_state, int):
            # random_state assumed to be an int
            self.random_state = np.random.RandomState(random_state)
        else:
            self.random_state = random_state
        super().__init__(*modifiers, **kwargs)

    def _randomize_object(self, modifier: BaseModifier, instrumentation: Parametrization):
        pass

    def _bound_value(self, modifier: BaseModifier, instrumentation: Parametrization):
        pass

    def _adapt_instrumentation(self, instrumentation: Parametrization, new_bounds):
        pass

    def step(self, execution: EXECUTION_POINTS = 'RESET', **kwargs) -> None:
        random_idx = np.random.randint(0, len(self.modifiers) - 1)
        bounded_modifier = self.modifiers[random_idx]
        for modifier in self.modifiers:
            if modifier is bounded_modifier:
                continue
            for instrumentation in modifier.instrumentation:
                self._randomize_object(modifier, instrumentation)