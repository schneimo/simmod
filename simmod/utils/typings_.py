from typing import Union, Tuple, TypeVar, AnyStr, Any, OrderedDict, Dict, List, Iterable, Optional
from numpy.typing import ArrayLike
import numpy


Array = Union[List, ArrayLike]
RGB = Union[ArrayLike, Tuple[ArrayLike]]
NDarray = TypeVar("numpy.ndarray")
