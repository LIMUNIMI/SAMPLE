"""Main widgets for the SAMPLE GUI"""
from sample.widgets import responsive as tk, images


class SAMPLEGUI(tk.Frame):
  """Main frame for the SAMPLE GUI

  Args:
    parent: Parent widget
    kwargs: Keyword arguments for :class:`tkinter.Frame`"""
  def __init__(self, parent, **kwargs):
    super().__init__(parent, **kwargs)
    # TODO: actually implement the GUI
    self.responsive(1, 1)
    self.button = tk.Button(
      self, text="Hello!"
    )
    self.button.grid()


class SAMPLERoot(tk.ThemedTk):
  """Root widget for the SAMPLE GUI

  Args:
    theme (str): Theme name. Default is :data:`"equilux"`
    kwargs: Keyword arguments for :class:`ttkthemes.ThemedTk`"""
  def __init__(self, theme: str = "equilux", **kwargs):
    super().__init__(**kwargs, theme=theme)
    self.title("SAMPLE")
    self.tk.call('wm', 'iconphoto', self._w, images.LogoIcon())
    self.responsive(1, 1)
    self.sample_frame = SAMPLEGUI(self)
    self.sample_frame.grid()
