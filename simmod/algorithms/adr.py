from typing import Any, AnyStr
import numpy as np

from simmod.algorithms.base import BaseAlgorithm
from simmod.modification.base_modifier import BaseModifier
from simmod.common.parametrization import Parametrization


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
