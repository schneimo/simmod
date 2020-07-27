from typing import Any, AnyStr

from simmod.algorithms.base import BaseAlgorithm
from simmod.common.parametrization import ADRInstrumentation


class AutomaticDomainRandomization(BaseAlgorithm):

    def __init__(self, *args: ADRInstrumentation, **kwargs: Any):
        super().__init__(*args, **kwargs)

    def _randomize(self, instrumentation: ADRInstrumentation):
        pass

    def _randomize_object(self, instrumentation: ADRInstrumentation, object_name: AnyStr):
        pass
