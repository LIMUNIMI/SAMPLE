"""Abstract base classes"""
import numpy as np
import itertools
import abc
from typing import List, Dict


class _AutoABC(abc.ABCMeta):
  def __subclasshook__(cls, c):
    return all(map(
      dir(c).__contains__,
      itertools.filterfalse(dir(abc.ABC).__contains__, dir(cls))
    )) or NotImplemented


class AbstractLinearRegressor(metaclass=_AutoABC):
  """Abstract class for linear regressors"""
  def fit(self, x, y, **kwargs):  # pylint: disable=W0613
    return self


class AbstractSinusoidalModel(metaclass=_AutoABC):
  """Abstract class for sinusoidal models"""
  def __init__(self, fs: float, h: float, **kwargs):  # pylint: disable=W0613
    self.fs = fs
    self.h = h

  def fit(self, x, y, **kwargs):  # pylint: disable=W0613
    return self

  @property
  @abc.abstractmethod
  def tracks_(self) -> List[Dict[str, np.ndarray]]:
    pass
