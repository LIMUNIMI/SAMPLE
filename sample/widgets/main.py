"""Main widgets for the SAMPLE GUI"""
from sample.widgets import responsive as tk, images, audioload, settings, analysis
from typing import Iterable, Tuple, Dict, Any


class SAMPLERoot(tk.ThemedTk):
  """Root widgets for the SAMPLE GUI

  Args:
    theme (str): Theme name. Default is :data:`"arc"`
    kwargs: Keyword arguments for :class:`ttkthemes.ThemedTk`"""
  def __init__(self, theme: str = "arc", **kwargs):
    super().__init__(**kwargs, theme=theme)
    self.title("SAMPLE")
    self.tk.call("wm", "iconphoto", self._w, images.LogoIcon())
    self.responsive(1, 1)


_default_tabs = (
  ("Load Audio", audioload.AudioLoadTab, None),
  ("Settings", settings.SettingsTab, None),
  ("Analysis", analysis.AnalysisTab, None),
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
      if kw is None:
        kw = dict()
      v = func(self.notebook, **kw)
      self.tabs.append(v)
      self.notebook.add(v, text=k)


class SAMPLESplashScreen(SAMPLERoot):
  """Splash screen widget for the SAMPLE GUI

  Args:
    kwargs: Keyword arguments for :class:`SAMPLERoot`"""
  def __init__(self, splash_time: float = 3000, gui_kwargs=None, **kwargs):
    super().__init__(**kwargs)
    if gui_kwargs is None:
      gui_kwargs = dict()
    self.gui_kwargs = gui_kwargs
    self.__img = images.LogoImage()
    self.label = tk.Label(self, image=self.__img)
    self.label.grid()
    self.overrideredirect(True)
    self.after(splash_time, self.splash_cbk)

  def splash_cbk(self):
    """Callback for the splash screen"""
    self.destroy()
    SAMPLEGUI(**self.gui_kwargs)


main = SAMPLESplashScreen
