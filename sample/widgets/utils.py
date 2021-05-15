"""Utilities for SAMPLE GUI widgets"""
from sample.widgets import logging
import tkinter as tk
from typing import Generator


def get_root(w: tk.Widget):
  """Get root widget

  Args:
    w (Widget): Widget

  Returns:
    The root widget"""
  return w._nametowidget(w.winfo_parent())  # pylint: disable=W0212


class RootProperty(property):
  """Property for getting and setting shared attributes via the root widget

  Args:
    name (str): Name of the property"""
  def __init__(self, name: str):
    def fget(self_):
      return getattr(get_root(self_), name, None)

    def fset(self_, value):
      setattr(get_root(self_), name, value)
    fget.__name__ = name
    fset.__name__ = name
    super().__init__(fget=fget, fset=fset)


class DataOnRootMeta(type):
  """Metaclass for getting and setting shared
  attributes via the root widget"""
  def __new__(mcs, name, bases, classdict):
    k = "sample_gui_root_properties"
    if k in classdict:
      for pn in classdict[k]:
        if pn not in classdict:
          classdict[pn] = RootProperty(pn)
    return super().__new__(mcs, name, bases, classdict)


class DataOnRootMixin(metaclass=DataOnRootMeta):
  """Mixin class for getting and setting shared
  attributes via the root widget"""
  sample_gui_root_properties = (
    "filedialog_dir",
    "audio_x",
    "audio_sr",
    "audio_trim_start",
    "audio_trim_stop",
  )

  @property
  def audio_loaded(self) -> bool:
    """Check if audio is loaded or not"""
    return all(
      v is not None
      for v in (
        self.audio_x,
        self.audio_sr,
        self.audio_trim_start,
        self.audio_trim_stop,
      )
    )

  def log_root_properties(self, *args, **kwargs):  # pylint: disable=W0613
    """Log all :class:`RootProperty` to debug level"""
    for p in self.sample_gui_root_properties:
      logging.debug("Property '%s': %s", p, getattr(self, p))


def widget_children(
  w: tk.Widget, include_root: bool = False
) -> Generator[tk.Widget, None, None]:
  """Get all children of the current widget

  Args:
    w (Widget): Widget
    include_root (bool): If :data:`True`, then include :data:`w`.
      Default is :data:`False`

  Returns:
    Generator of children widgets"""
  if include_root:
    yield w
  for v in w.winfo_children():
    for u in widget_children(v, True):
      yield u
