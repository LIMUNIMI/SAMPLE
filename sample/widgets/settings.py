"""Settings tab"""
from sample.widgets import responsive as tk, utils, logging, sample
from matplotlib.backends import _backend_tk
from scipy import signal
import functools
import inspect
from typing import Optional, Union, Type, Tuple, Dict, Any, Sequence, Callable


# --- Parsers ----------------------------------------------------------------
def try_func(
  func: Callable,
  exc: Union[Type[Exception], Tuple[Type[Exception], ...]],
  default: Optional = None
):
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


def custom_positive_int(x: str) -> int:
  """Custom function for parsing positive integers from strings
  Args:
    x (str): String to parse
  Returns:
    int: Integer value"""
  try:
    x = float(x)
  except ValueError:
    return 1
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
def postprocess_fbound(
  smfb_0: float, smfb_1: float
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
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
  out_kw = dict(
    sinusoidal_model__frequency_bounds=(
      smfb_0, smfb_1
    )
  )
  return in_kw, out_kw


def postprocess_windows(
  sinusoidal_model__n: int,
  wsize: int, wtype: str
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
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
  in_kw = dict(
    sinusoidal_model__n=sinusoidal_model__n,
    wsize=wsize, wtype=wtype
  )
  out_kw = dict(
    sinusoidal_model__n=sinusoidal_model__n,
    sinusoidal_model__w=w
  )
  return in_kw, out_kw
# ----------------------------------------------------------------------------


_settings = (
  ("sinusoidal_model__max_n_sines", dict(
    label="n sines", get_fn=custom_positive_int, init_value=64,
    tooltip="Maximum number of sinusoidal tracks per frame"
  )),
  ("sinusoidal_model__n", dict(
    label="fft size", get_fn=next_power_of_2, init_value=4096,
    tooltip="FFT size (in bins)"
  )),
  ("sinusoidal_model__h", dict(
    label="hop size", get_fn=custom_positive_int, init_value=1024,
    tooltip="FTT analysis window hop size (in samples)"
  )),
  ("wsize", dict(
    label="window size", get_fn=custom_positive_int, init_value=4096,
    tooltip="FFT analysis window size (in samples)",
  )),
  ("wtype", dict(
    label="window type", init_value="blackman",
    tooltip="FFT analysis window type",
  )),
  ("sinusoidal_model__freq_dev_offset", dict(
    label="frequency deviation offset", get_fn=try_float, init_value=20,
    tooltip="Frequency deviation threshold at 0 Hz (in Hertz)",
  )),
  ("sinusoidal_model__freq_dev_slope", dict(
    label="frequency deviation slope", get_fn=try_float, init_value=.0025,
    tooltip="Slope of frequency deviation threshold"
  )),
  ("smfb_0", dict(
    label="lower frequency bound", get_fn=try_float, init_value=20,
    tooltip="Minimum and accepted mean frequency (in Hertz)",
  )),
  ("smfb_1", dict(
    label="upper frequency bound", get_fn=try_float, init_value=16000,
    tooltip="Maximum and accepted mean frequency (in Hertz)",
  )),
  ("sinusoidal_model__peak_threshold", dict(
    label="onset threshold", get_fn=try_float, init_value=-66,
    tooltip="Minimum peak magnitude for modal tracks "
            "(magnitude at time=0, in dB)",
  )),
  ("sinusoidal_model__t", dict(
    label="peak detection threshold", get_fn=try_float, init_value=-90,
    tooltip="Threshold in dB for the peak detection algorithm"
  )),
  ("sinusoidal_model__min_sine_dur", dict(
    label="minimum sine duration", get_fn=non_negative, init_value=0.1,
    tooltip="Minimum duration of a track (in seconds)"
  )),
  ("sinusoidal_model__strip_t", dict(
    label="strip time", get_fn=strip_time_parse, init_value=0.5,
    tooltip="Strip time (in seconds). Tracks starting later "
            "than this time will be omitted from the track "
            "list. If is None, then don't strip",
  )),
  ("sinusoidal_model__reverse", dict(
    label="reverse", get_fn=custom_bool, set_fn=str, init_value=True,
    tooltip="If True, then process audio in reverse order of time",
  )),
)


_postprocess = (
  postprocess_windows,
  postprocess_fbound,
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
      tooltip (str): Parameter popup tool tip
      get_fn (callable): Parse function from entry value
      set_fn (callable): Entry value set function
      init_value: Initial value"""
    def __init__(
      self,
      parent: tk.Widget,
      name: str,
      label: Optional[str] = None,
      tooltip: Optional[str] = None,
      get_fn: Optional[Callable] = None,
      set_fn: Optional[Callable] = None,
      init_value: Optional = None,
    ):
      self.name = name
      self.label = tk.Label(parent, text=label or name)
      self.var = tk.StringVar(parent)
      self.spacer = tk.Frame(parent, width=32)
      self.entry = tk.Entry(parent, textvariable=self.var)
      self.get_fn = get_fn
      self.set_fn = set_fn
      if tooltip is None:
        self.tooltip = None
      else:
        self.tooltip = _backend_tk.ToolTip.createToolTip(
          self.label, tooltip
        )
      if init_value is not None:
        self.set(init_value)

    def get(self):
      """Get setting value

      Returns:
        The value"""
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
      if self.set_fn is not None:
        value = self.set_fn(value)
      self.var.set(value)
      return self

  def __init__(
    self,
    *args,
    setting_specs: Sequence[Tuple[str, Optional[Dict[str, Any]]]] = _settings,
    postprocess: Sequence[
      Callable[..., Tuple[Dict[str, Any], Dict[str, Any]]]
    ] = _postprocess,
    **kwargs
  ):
    super().__init__(*args, **kwargs)
    self._postprocess = postprocess
    self.responsive(1, 1)

    self.scrollframe = utils.ScrollableFrame(self)
    self.scrollframe.responsive(1, 1)
    self.scrollframe.grid(row=0)
    self.scrollframe.scrollable_frame.responsive(len(setting_specs), (0, 2))
    self._settings = dict()
    for k, kw in setting_specs:
      if kw is None:
        kw = dict()
      self.add_setting(k, **kw)

    self.bottom_row = tk.Frame(self)
    self.bottom_row.grid(row=1)
    self.bottom_row.responsive(1, 1)

    self.button = tk.Button(self.bottom_row, text="Apply")
    self.button.bind("<Button-1>", self.apply_cbk)
    self.button.grid()

    self.sample_object = sample.SAMPLE()
    self.apply_cbk()

  def add_setting(
    self, name,
    i: Optional[int] = None, grid: bool = True,
    **kwargs
  ):
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
    v = self._settings[name] = self.Setting(
      self.scrollframe.scrollable_frame,
      name=name, **kwargs
    )
    if grid:
      v.label.grid(row=i, column=0)
      v.spacer.grid(row=i, column=1)
      v.entry.grid(row=i, column=2)
    return v

  def apply_cbk(self, *args, **kwargs):  # pylint: disable=W0613
    """Callback for updating parameters from the settings"""
    settings = {
      k: s.get()
      for k, s in self._settings.items()
    }
    params = settings
    for func in self._postprocess:
      keys = inspect.signature(func).parameters.keys()
      kw = dict()
      tp = dict()
      for k, v in params.items():
        d = kw if k in keys else tp
        d[k] = v
      sett_update, param_update = func(**kw)
      for k, v in sett_update.items():
        settings[k] = v
      for k, v in param_update.items():
        tp[k] = v
      params = tp
    logging.debug("Settings: %s", settings)
    for k, v in settings.items():
      self._settings[k].set(v)
    self.sample_object.set_params(**params)
    logging.debug("SAMPLE: %s", self.sample_object)
