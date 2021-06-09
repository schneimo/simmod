from typing import Union
from typing import Tuple
from typing import TypeVar
from typing import AnyStr
from typing import Any
from typing import OrderedDict
from typing import Optional
from typing import Dict
from typing import List
from typing import Iterable
from typing import Sequence

from numpy.typing import ArrayLike
import numpy


Num = Union[int, float]
Array = Union[List, ArrayLike]
ArrayOrNum = Union[Array, Num]
RGB = Union[ArrayLike, Tuple[ArrayLike]]
NDarray = TypeVar("numpy.ndarray")

RandomStateOrGenerator = Union[numpy.random.RandomState, numpy.random.Generator]
