"""Main widgets for the SAMPLE GUI"""
from sample.widgets import responsive as tk, images
from typing import Iterable, Tuple, Dict, Any


class SAMPLERoot(tk.ThemedTk):
  """Root widgets for the SAMPLE GUI

  Args:
    theme (str): Theme name. Default is :data:`"equilux"`
    kwargs: Keyword arguments for :class:`ttkthemes.ThemedTk`"""
  def __init__(self, theme: str = "equilux", **kwargs):
    super().__init__(**kwargs, theme=theme)
    self.title("SAMPLE")
    self.tk.call('wm', 'iconphoto', self._w, images.LogoIcon())
    self.responsive(1, 1)


class DummyTab(tk.Frame):
  """Dummy tab"""
  def __init__(self, *args, text: str, **kwargs):
    super().__init__(*args, **kwargs)
    self.responsive(1, 1)
    self.label = tk.Label(self, text=text)
    self.label.grid()


_default_tabs = (
  ("Tab 1", DummyTab, dict(text="Alice")),
  ("Tab 2", DummyTab, dict(text="Bob")),
  ("Tab 3", DummyTab, dict(text="Charlie")),
  ("Tab 4", DummyTab, dict(text="Delia")),
  ("Tab 5", DummyTab, dict(text="Eric")),
)


class SAMPLEGUI(SAMPLERoot):
  """Root widget for the SAMPLE GUI

  Args:
    tabs: Tab initialization specifications as an iterable of tuples.
      Every tuple should have three elements: tab title (:class:`str`),
      tab init function (:class:`callable`), tab init function keyword
      arguments (:class:`dict`)
    kwargs: Keyword arguments for :class:`SAMPLERoot`"""
  def __init__(
    self,
    tabs: Iterable[Tuple[str, callable, Dict[str, Any]]] = _default_tabs,
    **kwargs
  ):
    super().__init__(**kwargs)
    self.notebook = tk.Notebook(self)
    self.notebook.grid()
    self.tabs = []
    for k, func, kw in tabs:
      v = func(self.notebook, **kw)
      self.tabs.append(v)
      self.notebook.add(v, text=k)


class SAMPLESplashScreen(SAMPLERoot):
  """Splash screen widget for the SAMPLE GUI

  Args:
    kwargs: Keyword arguments for :class:`SAMPLERoot`"""
  def __init__(self, splash_time: float = 3000, **kwargs):
    super().__init__(**kwargs)
    self.__img = images.LogoImage()
    self.label = tk.Label(self, image=self.__img)
    self.label.grid()
    self.overrideredirect(True)
    self.after(splash_time, self.splash_cbk)

  def splash_cbk(self):
    """Callback for the splash screen"""
    self.destroy()
    SAMPLEGUI()


main = SAMPLESplashScreen
