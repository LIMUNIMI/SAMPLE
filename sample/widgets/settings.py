"""Settings tab"""
import functools
import inspect
from tkinter import messagebox
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Type, Union

from matplotlib.backends import _backend_tk
from scipy import signal
import ttkthemes

from sample.widgets import logging
from sample.widgets import responsive as tk
from sample.widgets import sample, userfiles, utils


# --- Parsers ----------------------------------------------------------------
def try_func(func: Callable,
             exc: Union[Type[Exception], Tuple[Type[Exception], ...]],
             default: Optional = None):
  """Function wrapper for returning a default value on fail

  Args:
    func (callable): Function to wrap
    exc (exception): Exception class to catch (or tuple of classes). Default
      is :class:`Exception`
    default: Value returned on exception. Default is :data:`None`

  Returns:
    callable: Wrapped function"""

  @functools.wraps(func)
  def func_(*args, **kwargs):
    try:
      return func(*args, **kwargs)
    except exc:
      return default

  return func_


try_float = try_func(float, ValueError, 0.)


def custom_bool(x: str) -> bool:
  """Custom function for parsing booleans from strings
  Args:
    x (str): String to parse
  Returns:
    bool: Boolean value"""
  if x.lower() in ("false", "f", ""):
    return False
  try:
    x = float(x)
  except ValueError:
    return True
  return bool(x)


def strip_time_parse(x: str) -> Optional[float]:
  """Parse function for strip time.
  Invalid values are mapped to :data:`None` (don't strip)
  Args:
    x (str): Input number encoded as string
  Returns:
    float or None: If the string encodes a positive float, then that
    float is returned, else :data:`None`"""
  try:
    x = float(x)
  except ValueError:
    return None
  if x <= 0:
    return None
  return x


def non_negative(x: str) -> float:
  """Clip floats at zero. Invalid values are also mapped to zero
  Args:
    x (str): Input number as a string
  Returns:
    float: :data:`x` if positive, else :data:`0`"""
  try:
    x = float(x)
  except ValueError:
    return 0
  return max(0., x)


def custom_positive_int(x: str, dflt: Optional[int] = 1) -> int:
  """Custom function for parsing positive integers from strings

  Args:
    x (str): String to parse
    dflt (int): Default value to return on parse error

  Returns:
    int: Integer value"""
  try:
    x = float(x)
  except ValueError:
    return dflt
  return int(max(x, 1))


def next_power_of_2(x: str) -> int:
  """Parse floats from strings and then round them to next power of 2
  Args:
    x (str): Input number
  Returns:
    int: The minimum power of two greater or equal to :data:`floor(x)`"""
  return 2**(custom_positive_int(x) - 1).bit_length()


# ----------------------------------------------------------------------------


# --- Post-processing --------------------------------------------------------
def postprocess_fbound(smfb_0: float,
                       smfb_1: float) -> Tuple[Dict[str, Any], Dict[str, Any]]:
  """Postprocess frequency bounds

  Args:
    smfb_0 (float): Lower bound
    smfb_1 (float): Upper bound

  Returns:
    dict, dict: Postprocessed settings and parameters as dictionaries"""
  if smfb_0 > smfb_1:
    smfb_0, smfb_1 = (smfb_1, smfb_0)
  elif smfb_0 == smfb_1:
    smfb_0 = 20
    smfb_1 = 16000
  in_kw = dict(
      smfb_0=smfb_0,
      smfb_1=smfb_1,
  )
  out_kw = dict(sinusoidal_model__frequency_bounds=(smfb_0, smfb_1))
  return in_kw, out_kw


def postprocess_windows(sinusoidal_model__n: int, wsize: int,
                        wtype: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
  """Postprocess frequency bounds

  Args:
    sinusoidal_model__n (int): FFT size
    wsize (int): FFT window size
    wtype (str): FFT window type

  Returns:
    dict, dict: Postprocessed settings and parameters as dictionaries"""
  w = None
  wsize = min(wsize, sinusoidal_model__n)
  while True:
    try:
      w = signal.get_window(window=wtype, Nx=wsize)
    except ValueError:
      if wsize < 1:
        wsize = 4096
      else:
        wtype = "blackman"
      continue
    else:
      break
  in_kw = dict(sinusoidal_model__n=sinusoidal_model__n,
               wsize=wsize,
               wtype=wtype)
  out_kw = dict(sinusoidal_model__n=sinusoidal_model__n, sinusoidal_model__w=w)
  return in_kw, out_kw


def postprocess_guitheme(
    gui_theme: str,) -> Tuple[Dict[str, Any], Dict[str, Any]]:
  """Postprocess GUI theme

  Args:
    gui_theme (str): GUI theme name

  Returns:
    dict, dict: Postprocessed settings and parameters as dictionaries"""
  if not userfiles.UserTtkTheme.is_valid(
      gui_theme, log=True, messagebox=len(gui_theme)):
    gui_theme = ""
  return dict(gui_theme=gui_theme), {}


# ----------------------------------------------------------------------------

_settings = (
    ("resynth_group", dict(label="Resynthesis", is_spacer=True)),
    ("max_n_modes",
     dict(label="n modes",
          get_fn=functools.partial(custom_positive_int, dflt=None),
          init_value=None,
          tooltip="Maximum number of modes to use for resynthesis")),
    ("analysis_space", dict(is_spacer=True, label=" ")),
    ("analysis_group", dict(label="Analysis", is_spacer=True)),
    ("sinusoidal_model__max_n_sines",
     dict(label="n sines",
          get_fn=custom_positive_int,
          init_value=64,
          tooltip="Maximum number of sinusoidal tracks per frame")),
    ("sinusoidal_model__n",
     dict(label="fft size",
          get_fn=next_power_of_2,
          init_value=4096,
          tooltip="FFT size (in bins)")),
    ("sinusoidal_model__h",
     dict(label="hop size",
          get_fn=custom_positive_int,
          init_value=1024,
          tooltip="FTT analysis window hop size (in samples)")),
    ("wsize",
     dict(
         label="window size",
         get_fn=custom_positive_int,
         init_value=4096,
         tooltip="FFT analysis window size (in samples)",
     )),
    ("wtype",
     dict(
         label="window type",
         init_value="blackman",
         options=sorted(
             ("boxcar", "triang", "blackman", "hamming", "hann", "bartlett",
              "flattop", "parzen", "bohman", "blackmanharris", "nuttall",
              "barthann", "cosine", "exponential", "tukey", "taylor")),
         tooltip="FFT analysis window type",
     )),
    ("sinusoidal_model__freq_dev_offset",
     dict(
         label="frequency deviation offset",
         get_fn=try_float,
         init_value=20,
         tooltip="Frequency deviation threshold at 0 Hz (in Hertz)",
     )),
    ("sinusoidal_model__freq_dev_slope",
     dict(label="frequency deviation slope",
          get_fn=try_float,
          init_value=.0025,
          tooltip="Slope of frequency deviation threshold")),
    ("smfb_0",
     dict(
         label="lower frequency bound",
         get_fn=try_float,
         init_value=20,
         tooltip="Minimum and accepted mean frequency (in Hertz)",
     )),
    ("smfb_1",
     dict(
         label="upper frequency bound",
         get_fn=try_float,
         init_value=16000,
         tooltip="Maximum and accepted mean frequency (in Hertz)",
     )),
    ("sinusoidal_model__peak_threshold",
     dict(
         label="onset threshold",
         get_fn=try_float,
         init_value=-66,
         tooltip="Minimum peak magnitude for modal tracks "
         "(magnitude at time=0, in dB)",
     )),
    ("sinusoidal_model__t",
     dict(label="peak detection threshold",
          get_fn=try_float,
          init_value=-90,
          tooltip="Threshold in dB for the peak detection algorithm")),
    ("sinusoidal_model__min_sine_dur",
     dict(label="minimum sine duration",
          get_fn=non_negative,
          init_value=0.1,
          tooltip="Minimum duration of a track (in seconds)")),
    ("sinusoidal_model__strip_t",
     dict(
         label="strip time",
         get_fn=strip_time_parse,
         init_value=0.5,
         tooltip="Strip time (in seconds). Tracks starting later "
         "than this time will be omitted from the track "
         "list. If is None, then don't strip",
     )),
    ("sinusoidal_model__reverse",
     dict(
         label="reverse",
         init_value=True,
         boolean=True,
         tooltip="If True, then process audio in reverse order of time",
     )),
    ("gui_space", dict(is_spacer=True, label=" ")),
    ("gui_group", dict(label="GUI", is_spacer=True)),
    ("gui_theme",
     dict(
         label="gui theme",
         init_value="",
         options=ttkthemes.THEMES,
         tooltip="GUI theme",
     )),
)

_postprocess = (
    postprocess_windows,
    postprocess_fbound,
    postprocess_guitheme,
)


class SettingsTab(utils.DataOnRootMixin, tk.Frame):
  """Tab for setting SAMPLE parameters

  Args:
    setting_specs: Setting specifications as a sequence of key-value tuple
      where the value is an optional dictionary of keyword
      arguments for :class:`SettingsTab.Setting`
    postprocess: Postprocessing functions as a sequence of callables that take
      as keyword arguments setting values and return two dictionaries. The
      first is used to update the settings values and the second is used to
      update the parameter values
    args: Positional arguments for :class:`tkinter.ttk.Frame`
    kwargs: Keyword arguments for :class:`tkinter.ttk.Frame`"""

  class Setting:
    """Setting wrapper

    Args:
      parent (Widget): Parent widget
      name (str): Parameter keyword
      label (str): Parameter label. If :data:`None`, then use name as label
      is_spacer (bool): Set to :data:`True` to get only a spacer, not a setting
      tooltip (str): Parameter popup tool tip
      get_fn (callable): Parse function from entry value
      set_fn (callable): Entry value set function
      init_value: Initial value
      options (list): List of options for a dropdown menu
      boolean (bool): Set to :data:`True` for a checkbutton"""

    def __init__(
        self,
        parent: tk.Widget,
        name: str,
        label: Optional[str] = None,
        is_spacer: bool = False,
        tooltip: Optional[str] = None,
        get_fn: Optional[Callable] = None,
        set_fn: Optional[Callable] = None,
        init_value: Optional = None,
        options: Optional[Sequence[str]] = None,
        boolean: bool = False,
    ):
      self.name = name
      self.label = tk.Label(parent, text=label or name)
      self.spacer = tk.Frame(parent, width=32)
      self.is_spacer = is_spacer
      if self.is_spacer:
        self.label.config(font="-weight bold")
        return
      if boolean:
        self.var = tk.BooleanVar(parent)
        self.entry = tk.Checkbutton(parent, variable=self.var)
        if init_value is not None:
          self.var.set(init_value)
      else:
        self.var = tk.StringVar(parent)
        if options is None:
          self.entry = tk.Entry(parent, textvariable=self.var)
        else:
          self.entry = tk.OptionMenu(
              parent, self.var,
              options[0] if init_value is None else init_value, *options)
      self.get_fn = get_fn
      self.set_fn = set_fn
      if tooltip is None:
        self.tooltip = None
      else:
        self.tooltip = _backend_tk.ToolTip.createToolTip(self.label, tooltip)
      if init_value is not None:
        self.set(init_value)

    def get(self):
      """Get setting value

      Returns:
        The value"""
      if self.is_spacer:
        return
      v = self.var.get()
      if self.get_fn is not None:
        v = self.get_fn(v)
      return v

    def set(self, value):
      """Set entry value

      Args:
        value: The value to set

      Returns:
        self"""
      if self.is_spacer:
        return
      if self.set_fn is not None:
        value = self.set_fn(value)
      self.var.set(value)
      return self

  def __init__(self,
               *args,
               setting_specs: Sequence[Tuple[str,
                                             Optional[Dict[str,
                                                           Any]]]] = _settings,
               postprocess: Sequence[Callable[...,
                                              Tuple[Dict[str, Any],
                                                    Dict[str,
                                                         Any]]]] = _postprocess,
               **kwargs):
    super().__init__(*args, **kwargs)
    self._postprocess = postprocess
    self.responsive(1, 1)

    self.scrollframe = utils.ScrollableFrame(self)
    self.scrollframe.responsive(1, 1)
    self.scrollframe.grid(row=0)
    self.scrollframe.scrollable_frame.responsive(len(setting_specs), (0, 2))
    self._settings: Dict[str, SettingsTab.Setting] = {}
    for k, kw in setting_specs:
      if kw is None:
        kw = {}
      self.add_setting(k, **kw)

    self.bottom_row = tk.Frame(self)
    self.bottom_row.grid(row=1)
    self.bottom_row.responsive(1, 1)

    self.button = tk.Button(self.bottom_row, text="Apply")
    self.button.bind("<Button-1>", self.apply_cbk)
    self.button.grid()

    self.sample_object = sample.SAMPLE4GUI()
    self.apply_cbk(from_file=True)

  def reset_selections(self, *args, **kwargs):  # pylint: disable=W0613
    """Reset selections in entries"""
    for v in self._settings.values():
      v.entry.selection_clear()

  def add_setting(self,
                  name,
                  i: Optional[int] = None,
                  grid: bool = True,
                  **kwargs):
    """Add a setting to the tab

    Args:
      name: Setting name
      i (int): Setting index. If :data:`None`, then the
        setting is added as the last
      grid (bool): If :data:`True` (default), then add
        setting widgets to the parent grid layout
      kwargs: Keyword arguments for :class:`SettingsTab.Setting`"""
    if i is None:
      i = len(self._settings)
    v = self._settings[name] = self.Setting(self.scrollframe.scrollable_frame,
                                            name=name,
                                            **kwargs)
    if grid:
      v.label.grid(row=i, column=0)
      v.spacer.grid(row=i, column=1, columnspan=1 + v.is_spacer)
      if not v.is_spacer:
        v.entry.grid(row=i, column=2)
    return v

  def apply_cbk(self, *args, from_file: bool = False, **kwargs):  # pylint: disable=W0613
    """Callback for updating parameters from the settings"""
    ttk_theme = userfiles.UserTtkTheme(self.settings_file)
    prev_theme = ttk_theme.get()
    settings = {}
    if from_file and self.settings_file.is_valid(
    ) and self.settings_file.exists():
      settings = self.settings_file.load_json()
    settings = {
        k: settings.get(k, s.get())
        for k, s in self._settings.items()
        if not s.is_spacer
    }
    params = settings
    for func in self._postprocess:
      keys = inspect.signature(func).parameters.keys()
      kw = {}
      tp = {}
      for k, v in params.items():
        d = kw if k in keys else tp
        d[k] = v
      sett_update, param_update = func(**kw)
      for k, v in sett_update.items():
        settings[k] = v
      for k, v in param_update.items():
        tp[k] = v
      params = tp
    if len(settings["gui_theme"]) == 0:
      settings["gui_theme"] = prev_theme
    logging.debug("Settings: %s", settings)
    if self.settings_file.is_valid():
      self.settings_file.save_json(settings, indent=2)
    for k, v in settings.items():
      self._settings[k].set(v)
    self.sample_object.set_params(**params)
    logging.debug("SAMPLE: %s", self.sample_object)
    if prev_theme != ttk_theme.get():
      logging.info("Reload GUI to apply changes")
      if messagebox.askyesno(
          "Reload",
          "Reload GUI to apply changes to the theme. Do you want to reload now?"
      ):
        if self.audio_cache_file.is_valid():
          logging.info("Caching state to %s", self.audio_cache_file.path)
          self.audio_cache_file.save_pickled(
              dict(
                  audio_x=self.audio_x,
                  audio_sr=self.audio_sr,
                  audio_trim_start=self.audio_trim_start,
                  audio_trim_stop=self.audio_trim_stop,
              ))
        r = utils.get_root(self)
        r.master.should_reload = True
        r.quit()
