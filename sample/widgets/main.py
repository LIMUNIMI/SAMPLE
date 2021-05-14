"""Main widgets for the SAMPLE GUI"""
from sample.widgets import responsive as tk, images


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


class SAMPLEGUI(SAMPLERoot):
  """Root widget for the SAMPLE GUI

  Args:
    kwargs: Keyword arguments for :class:`SAMPLERoot`"""
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.label = tk.Label(self, text="Actual GUI")
    self.label.grid()


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
