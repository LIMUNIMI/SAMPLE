"""Utilities for responsive SAMPLE GUI widgets"""
import tkinter as tk
from tkinter import ttk
import ttkthemes
import logging
from typing import Iterable, Union, Optional, Dict


class ResponsiveMixin:
  """Mixin class for responsive widgets"""
  def grid(self, *args, sticky=tk.NSEW, **kwargs):  # pylint: disable=W0235
    """Add the widget to a grid layout

    Args:
      sticky: Default is :data:`NSEW`
      args: Positional arguments for :data:`super().grid`
      kwargs: Keyword arguments for :data:`super().grid`"""
    return super().grid(*args, sticky=sticky, **kwargs)

  def responsive(
    self,
    rows: Optional[Union[int, Iterable[int]]] = None,
    cols: Optional[Union[int, Iterable[int]]] = None,
  ):
    """Set up responsive rows and columns

    Args:
      rows (int): If it is an integer, then it is the number of responsive
        rows. If it is an iterable, then only the rows at the specified
        indices will be responsive
      cols (int): If it is an integer, then it is the number of responsive
        columns. If it is an iterable, then only the columns at the specified
        indices will be responsive

    Returns:
      self"""
    for it, meth in (
      (rows, "grid_rowconfigure"),
      (cols, "grid_columnconfigure"),
    ):
      if not hasattr(self, meth):
        logging.warning(
          "Object of type '%s' doesn't have attribute '%s'",
          type(self).__name__, meth
        )
      elif it is not None:
        if not isinstance(it, Iterable):
          it = range(it)
        for i in it:
          getattr(self, meth)(i, weight=1)
    return self


def responsive(cls: type) -> type:
  """Mix :class:`ResponsiveMixin` with the given class

  Args:
    cls (type): Class to mix

  Returns:
    type: Child class"""
  return type(
    "Responsive{}".format(cls.__name__),
    (ResponsiveMixin, cls), dict()
  )


__responsive_classes: Dict[type, type] = dict()
__tk_modules = (
  ttkthemes, ttk, tk
)


def __getattr__(name):  # pylint: disable=C0103
  """Wrap classes from tkinter mixing in :class:`ResponsiveMixin`"""
  for m in __tk_modules:
    if hasattr(m, name):
      cls = getattr(m, name)
      break
  else:
    raise AttributeError(
      "module '{}' has no attribute '{}'".format(
        __name__, name
      )
    )
  if not isinstance(cls, type):
    return cls
  if cls not in __responsive_classes:
    __responsive_classes[cls] = responsive(cls)
  return __responsive_classes[cls]
