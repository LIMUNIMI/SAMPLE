"""Pyplot widgets"""
from sample.widgets import responsive as tk, logging, utils
from matplotlib.backends import backend_tkagg
from matplotlib import figure
from unittest import mock
import contextlib


@contextlib.contextmanager
def mock_tk2ttk(*args):
  """Context manager to mock :mod:`tkinter` classes
  with classes from :mod:`tkinter.ttk`

  Args:
    args: Class names"""
  n = len(args)

  @contextlib.contextmanager
  def mock_tk2ttk_inner(depth: int = 0):
    if n <= depth:
      yield
    else:
      with mock_tk2ttk_inner(depth=depth + 1):
        k = args[depth]
        with mock.patch.object(tk.tk, k, getattr(tk, k)):
          yield

  with mock_tk2ttk_inner():
    yield


class PyplotFrame(tk.Frame):
  """Frame with pyplot figure and navigation toolbar

  Args:
    fig_dpi (int): Figure DPI
    fig_w (int): Figure width in pixels
    fig_h (int): Figure height in pixels
    args: Positional arguments for :class:`tkinter.ttk.Frame`
    kwargs: Keyword arguments for :class:`tkinter.ttk.Frame`"""
  _remap_style = (
      (tk.tk.Button, (
          ("background", "bg"),
          ("foreground", "fg"),
          ("height", "height"),
          ("width", "width"),
          ("borderwidth", "bd"),
          ("active background", "active background"),
          ("active foreground", "active foreground"),
      )),
      (tk.tk.Label, (
          ("background", "bg"),
          ("foreground", "fg"),
          ("height", "height"),
          ("width", "width"),
          ("borderwidth", "bd"),
      )),
      (tk.tk.Checkbutton, (
          ("background", "bg"),
          ("foreground", "fg"),
          ("height", "height"),
          ("width", "width"),
          ("borderwidth", "bd"),
          ("active background", "active background"),
          ("active foreground", "active foreground"),
      )),
      (tk.tk.Frame, (("background", "bg"),)),
  )

  def __init__(self,
               *args,
               fig_dpi: int = 72,
               fig_w: int = 640,
               fig_h: int = 360,
               **kwargs):
    super().__init__(*args, **kwargs)
    self.responsive((1,), 1)

    self.fig = figure.Figure(
        figsize=(fig_w / fig_dpi, fig_h / fig_dpi),
        dpi=fig_dpi,
    )

    self.canvas = backend_tkagg.FigureCanvasTkAgg(self.fig, master=self)
    self.canvas.draw()

    self.canvas_widget = self.canvas.get_tk_widget()
    self.canvas_widget.grid(row=1, sticky=tk.NSEW)

    self.toolbar = backend_tkagg.NavigationToolbar2Tk(self.canvas,
                                                      self,
                                                      pack_toolbar=False)

    for c in utils.widget_children(self.toolbar, True):
      logging.debug("Toolbar Children: %s (%s)", c, type(c))
      for cls, mappings in self._remap_style:
        if isinstance(c, cls):
          if cls in (tk.tk.Button, tk.tk.Checkbutton):
            c.config(relief=tk.tk.FLAT)
          for k_from, k_to in mappings:
            v = utils.root_style(self, f"T{cls.__name__}", k_from)
            logging.debug("%s of %s from %s of %s: %s", k_to,
                          type(c).__name__, k_from, cls.__name__, v)
            if v != "":
              c.config(**{k_to: v})
          break

    self.toolbar.update()
    self.toolbar.grid(row=0, sticky=tk.NSEW)
