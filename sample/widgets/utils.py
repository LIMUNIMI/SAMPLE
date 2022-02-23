"""Utilities for SAMPLE GUI widgets"""
from sample.widgets import logging, responsive
import tkinter as tk
from tkinter import ttk
from matplotlib import colors
from typing import Generator, Optional


def get_root(w: tk.Widget):
  """Get root widget

  Args:
    w (Widget): Widget

  Returns:
    The root widget"""
  return w._nametowidget(w.winfo_parent())  # pylint: disable=W0212


def root_style(w: tk.Widget, wname: str, wattr: str):
  """Get the style of a widget from the root

  Args:
    w (Widget): Get the root of this widget as a style source
    wname (str): Get the style for this type of widgets
    wattr (str): Get the style for this attribute of a widget"""
  return ttk.Style(get_root(w)).lookup(wname, wattr)


def root_color(w: tk.Widget,
               wname: str,
               wattr: str,
               default: Optional = None,
               key: Optional[str] = None):
  """Get the color style of a widget from the root

  Args:
    w (Widget): Get the root of this widget as a style source
    wname (str): Get the style for this type of widgets
    wattr (str): Get the style for this attribute of a widget
    default: If the attribute is not a valid color, then return this value
    key (str): If specified, then output as a key-value dictionary"""
  c = root_style(w=w, wname=wname, wattr=wattr)
  if not colors.is_color_like(c):
    c = default
  if key is not None:
    return {key: c} if c is not None else {}
  return c


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
      "filedialog_file",
      "filedialog_dir_save",
      "audio_x",
      "audio_sr",
      "audio_trim_start",
      "audio_trim_stop",
      "sample_object",
      "audio_resynth_x",
      "persistent_dir",
      "settings_file",
  )

  @property
  def audio_loaded(self) -> bool:
    """Check if audio is loaded or not"""
    return all(v is not None for v in (
        self.audio_x,
        self.audio_sr,
        self.audio_trim_start,
        self.audio_trim_stop,
    ))

  def log_root_properties(self, *args, **kwargs):  # pylint: disable=W0613
    """Log all :class:`RootProperty` to debug level"""
    for p in self.sample_gui_root_properties:
      logging.debug("Property '%s': %s", p, getattr(self, p))


def widget_children(
    w: tk.Widget,
    include_root: bool = False) -> Generator[tk.Widget, None, None]:
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


class ScrollableFrame(responsive.Frame):
  """Wrapper frame for scrolling

  Args:
    args: Positional arguments for :class:`tkinter.ttk.Frame`
    kwargs: Keyword arguments for :class:`tkinter.ttk.Frame`"""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.responsive(1, 1)
    self.canvas = responsive.Canvas(self,
                                    highlightthickness=0,
                                    **root_color(self,
                                                 "TFrame",
                                                 "background",
                                                 key="background"))
    self.canvas.grid(row=0, column=0)
    self.canvas.responsive(1, 1)

    self.scrollbar = responsive.Scrollbar(self,
                                          orient="vertical",
                                          command=self.canvas.yview)
    self.scrollbar.grid(row=0, column=1)

    self.scrollable_frame = responsive.Frame(self.canvas)
    self.scrollable_frame.bind(
        "<Configure>",
        lambda _: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
    self.canvas.create_window((0, 0), anchor="nw", window=self.scrollable_frame)
    self.canvas.configure(yscrollcommand=self.scrollbar.set)
