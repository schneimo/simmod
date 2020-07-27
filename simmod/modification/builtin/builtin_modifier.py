from simmod.modification.base_modifier import BaseModifier
import numpy as np

from collections import defaultdict


class BuiltInModifyer(BaseModifier):

    def __init__(self, sim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sim = sim

    @property
    def model(self):
        return None

    def set_value(self, name: str, value):
        setattr(self.sim, name, value)
