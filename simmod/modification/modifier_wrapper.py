from abc import ABCMeta, abstractmethod
from simmod.utils.typings_ import *


class ModifierWrapper(metaclass=ABCMeta):

    def __init__(self, instrumentation):
        self.instrumentation = instrumentation

    def __str__(self):
        return str(self.instrumentation)

    def sample(self, n_params,
               random_state: Optional[RandomStateOrGenerator] = None):
        return self.instrumentation.sample(n_params, random_state)

    @property
    def entropy(self) -> float:
        return self.instrumentation.entropy

    @property
    def parameter_values(self) -> Sequence[float, float]:
        return self.instrumentation.parameter_values

    def update(self, new_values, **kwargs) -> None:
        self.instrumentation.update(new_values, **kwargs)

    def get_json(self) -> Dict:
        return self.instrumentation.get_json()

    def setter(self, name: AnyStr, *value: Array):
        value = self.adjust(value)
        self.instrumentation.setter(name, *value)

    @abstractmethod
    def adjust(self, value) -> Array:
        raise NotImplementedError
