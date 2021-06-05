import pytest

from simmod.modification.builtin import BuiltInModifier


class TestClass():

    def __init__(self):
        self.parameter = 1


def test_builtin_modifier():
    t = TestClass()
    mod = BuiltInModifier(t)

    assert t.parameter == 1
    mod.set_value('parameter', 2)
    assert t.parameter == 2 and t.parameter != 1

    mod.set_value('parameter', 1)
    assert t.parameter != 2 and t.parameter == 1

    try:
        mod.set_value('not_available', 1)
    except ValueError:
        pass