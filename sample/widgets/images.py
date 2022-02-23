"""Utilities for images"""
from PIL import Image, ImageTk
from sample import vid
import io


class TempImage(ImageTk.PhotoImage):
  """Temporary image made with a plot function

  Args:
    plt_fn (callable): Function to print to file. It will be called
      with a file as the only argument
    args: Positional arguments for :class:`PIL.ImageTk.PhotoImage`
    kwargs: Keyword arguments for :class:`PIL.ImageTk.PhotoImage`"""

  def __init__(self, *args, plt_fn: callable, **kwargs):
    with io.BytesIO() as buf:
      plt_fn(buf)
      buf.seek(0)
      img = Image.open(buf)
      super().__init__(*args, **kwargs, image=img)


class LogoIcon(TempImage):
  """Temporary image for the GUI logo icon

  Args:
    plt_fn (callable): Function to print to file. It will be called
      with a file as the only argument
    args: Positional arguments for :class:`TempImage`
    kwargs: Keyword arguments for :class:`TempImage`"""

  def __init__(self, *args, plt_fn: callable = vid.icon_plt_fn, **kwargs):
    super().__init__(*args, plt_fn=plt_fn, **kwargs)


class LogoImage(TempImage):
  """Temporary image for the GUI logo image

  Args:
    plt_fn (callable): Function to print to file. It will be called
      with a file as the only argument
    args: Positional arguments for :class:`TempImage`
    kwargs: Keyword arguments for :class:`TempImage`"""

  def __init__(self, *args, plt_fn: callable = vid.logo_plt_fn, **kwargs):
    super().__init__(*args, plt_fn=plt_fn, **kwargs)
