"""Sinusoidal model"""
import functools
import itertools
import math
from typing import Any, Callable, Dict, Iterable, List, Tuple

import numpy as np
from sklearn import base

import sample.sms
import sample.sms.dsp
import sample.utils
import sample.utils.dsp
import sample.utils.learn

utils = sample.utils
sms = sample.sms


def min_key(it: Iterable, key: Callable) -> Tuple[Any, Any]:
  """Minimum value and corresponding argument

  Args:
    it (iterable): Iterable of function arguments
    key (callable): Function

  Returns:
    Argmin and min of :data:`key(i) for i in it`"""
  i_ = None
  x_ = None
  for i in it:
    x = key(i)
    if x_ is None or x < x_:
      i_ = i
      x_ = x
  return i_, x_


class SineTracker(base.BaseEstimator):
  """Model for keeping track of sines across frames

  Args:
    fs (int): Sampling frequency
    h (int): Hop size (in samples)
    max_n_sines (int): Maximum number of tracks per frame
    min_sine_dur (float): Minimum duration of a track in seconds
    freq_dev_offset (float): Frequency deviation threshold at 0Hz
    freq_dev_slope (float): Slope of frequency deviation threshold
    **kwargs: Additional parameters for sub-models

  Attributes:
    tracks_ (list of dict): Deactivated tracks"""

  def __init__(self,
               fs: int = 44100,
               h: int = 500,
               max_n_sines: int = 100,
               min_sine_dur: int = 0.04,
               freq_dev_offset: float = 20,
               freq_dev_slope: float = 0.01,
               **kwargs):
    self.fs = fs
    self.h = h
    self.max_n_sines = max_n_sines
    self.min_sine_dur = min_sine_dur
    self.freq_dev_offset = freq_dev_offset
    self.freq_dev_slope = freq_dev_slope
    self.set_params(**kwargs)

  @property
  def frame_rate(self) -> float:
    """Number of DFT frames per seconds"""
    return self.fs / self.h

  def reset(self):
    """Reset tracker state

    Returns:
      SineTracker: self"""
    self.tracks_ = []
    self._active_tracks = []
    self._frame = 0
    return self

  @property
  def n_active_tracks(self) -> int:
    """Number of currently active tracks"""
    return len(self._active_tracks)

  @property
  def min_sine_len(self) -> int:
    """Minimum duration of a track in number of frames"""
    return max(2, math.ceil(self.min_sine_dur * self.frame_rate))

  @property
  def all_tracks_(self) -> Iterable[dict]:
    """All deactivated tracks in :attr:`tracks_` and those active tracks that
    would pass the cleanliness check at the current state of the tracker"""
    return itertools.chain(
        self.tracks_,
        filter(self.track_ok, map(self.numpy_track, self._active_tracks)))

  def df(self, f: float) -> float:
    """Frequency deviation threshold at given frequency

    Args:
      f (float): Frequency in Hz

    Returns:
      float: Frequency deviation threshold in Hz"""
    return self.freq_dev_offset + self.freq_dev_slope * f

  @staticmethod
  def numpy_track(track: dict) -> dict:
    """Convert to numpy arrays all values in a track

    Args:
      track (dict): Track to convert

    Returns:
      dict: Converted track"""
    return {k: np.array(v) for k, v in track.items()}

  def track_ok(self, track: dict) -> bool:
    """Check if a deactivated track is ok to be saved

    Args:
      track (dict): Track to check

    Returns:
      bool: Whether the track is ok or not"""
    return len(track["freq"]) >= self.min_sine_len

  def deactivate(self, track_index: int) -> dict:
    """Remove track from list of active tracks and save it in
    :attr:`tracks_` if it meets cleanliness criteria

    Args:
      track_index (int): Index of track to deactivate

    Returns:
      dict: Deactivated track"""
    t = self.numpy_track(self._active_tracks.pop(track_index))
    if self.track_ok(t):
      self.tracks_.append(t)
    return t

  def __call__(self, pfreq: np.ndarray, pmag: np.ndarray, pph: np.ndarray):
    """Update tracking with another frame

    Args:
      pfreq (array): Peak frequencies in Hz
      pmag (array): Peak magnitudes in dB
      pph (array): Peak phases

    Returns:
      SineTracker: self"""
    peak_order = np.argsort(-pmag)  # decreasing order of magnitude
    free_track = np.ones(self.n_active_tracks, dtype=bool)
    free_peak = np.ones(pmag.size, dtype=bool)

    # Try to continue active tracks
    for p_i in peak_order:
      if not any(free_track):
        break
      t_i, df = min_key(
          # choose amongst free tracks only
          filter(free_track.__getitem__, range(self.n_active_tracks)),
          # absolute difference from last peak's frequency
          lambda i, p=p_i: np.abs(self._active_tracks[i]["freq"][-1] - pfreq[p])
      )
      # If deviation is below threshold, add peak to active track
      if df < self.df(pfreq[p_i]):
        self._active_tracks[t_i]["freq"].append(pfreq[p_i])
        self._active_tracks[t_i]["mag"].append(pmag[p_i])
        self._active_tracks[t_i]["phase"].append(pph[p_i])
        free_track[t_i] = False
        free_peak[p_i] = False

    # Deactivate non-continued tracks
    for t_i in filter(free_track.__getitem__,
                      reversed(range(self.n_active_tracks))):
      self.deactivate(t_i)

    # Activate new tracks for free peaks
    for p_i in filter(free_peak.__getitem__, range(pmag.size)):
      if self.n_active_tracks >= self.max_n_sines:
        break
      self._active_tracks.append({
          "start_frame": self._frame,
          "freq": [pfreq[p_i]],
          "mag": [pmag[p_i]],
          "phase": [pph[p_i]],
      })

    self._frame += 1
    return self


def _decorate_sinusoidal_model(func):
  """Decorator for deprecated arguments of :class:`SinusoidalModel`"""

  @utils.deprecated_argument("max_n_sines", "tracker__max_n_sines")
  @utils.deprecated_argument("min_sine_dur", "tracker__min_sine_dur")
  @utils.deprecated_argument(
      "safe_sine_len",
      convert=lambda _, **kwargs:
      ("tracker__min_sine_dur",
       max(-1 if kwargs["safe_sine_len"] is None else kwargs["safe_sine_len"],
           kwargs.get("tracker__min_sine_dur", 1))),
      msg="'safe_sine_len' is no longer a parameter, since the "
      "'min_sine_dur' of the sine tracker is already expressed in frames")
  @utils.deprecated_argument("freq_dev_offset", "tracker__freq_dev_offset")
  @utils.deprecated_argument("freq_dev_slope", "tracker__freq_dev_slope")
  @utils.deprecated_argument("sine_tracker_cls",
                             convert=lambda sine_tracker_cls, **kwargs:
                             ("tracker", sine_tracker_cls()))
  @utils.deprecated_argument("save_intermediate", "intermediate__save")
  @utils.deprecated_argument("fs", "tracker__fs")
  @utils.deprecated_argument("h", "tracker__h")
  @functools.wraps(func)
  def func_(*args, **kwargs):
    return func(*args, **kwargs)

  return func_


class SinusoidalModel(base.TransformerMixin, base.BaseEstimator):
  """Model for sinusoidal tracking

  Args:
    w: Analysis window
    n (int): FFT size. Defaults to 2048
    t (float): threshold in dB. Defaults to -90
    tracker (SineTracker): Sine tracker
    intermediate (OptionalStorage): Optionally-activatable storage
    padded (bool): Analyse a zero-padded version of the input
    **kwargs: Additional parameters for sub-models. See
      :class:`sample.sms.sm.SineTracker` and
      :class:`sample.utils.learn.OptionalStorage`

  Attributes:
    w_ (array): Effective analysis window"""

  @_decorate_sinusoidal_model
  def __init__(
      self,
      w: np.ndarray = None,
      n: int = 2048,
      t: float = -90,
      tracker: SineTracker = None,
      intermediate: utils.learn.OptionalStorage = None,
      padded: bool = False,
      **kwargs,
  ):
    self.w = w
    self.n = n
    self.t = t
    self.tracker = tracker
    self.intermediate = intermediate
    self.padded = padded
    self.set_params(**kwargs)

  @_decorate_sinusoidal_model
  def set_params(self, **kwargs):
    return super().set_params(**kwargs)

  @utils.learn.default_property
  def intermediate(self):
    """Optionally-activatable storage"""
    return utils.learn.OptionalStorage()

  @utils.learn.default_property
  def tracker(self):
    """Sine tracker"""
    return SineTracker()

  @utils.learn.default_property
  def w(self) -> np.ndarray:
    """Analysis window"""
    return np.hamming(2001)

  @property
  def frame_rate(self) -> float:
    """Number of DFT frames per seconds"""
    return self.tracker.frame_rate

  @property
  def fs(self):
    """Sampling frequency"""
    return self.tracker.fs

  @property
  def h(self):
    """Hop size (in samples)"""
    return self.tracker.h

  def fit(
      self,
      x: np.ndarray,
      y=None,  # pylint: disable=W0613
      **kwargs):
    """Analyze audio data

    Args:
      x (array): audio input
      y (ignored): exists for compatibility
      kwargs: Any parameter, overrides initialization. Mainly meant for setting
        the sampling frequency (:data:`tracker__fs`)

    Returns:
      SinusoidalModel: self"""
    # Avoid _fit_args appearing in signatures, since it's considered
    # an implementation detail
    fit_args = kwargs.pop("_fit_args", None)
    self.intermediate.reset()
    self.set_params(**kwargs)
    self.tracker.reset()
    self.w_ = self._normalized_window
    if getattr(self.tracker, "reverse", False):
      x = np.flip(x)

    for mx, px in map(functools.partial(self.intermediate.append, "stft"),
                      self.dft_frames(x)):
      ploc, pmag, pph = self.intermediate.append(
          "peaks", sms.dsp.peak_detect_interp(mx, px, self.t))
      pfreq = ploc * self.fs / self.n  # indices to frequencies in Hz
      self.tracker(pfreq, pmag, pph)
      if fit_args is not None:
        fit_args.progress_update()
    return self

  @property
  def tracks_(self) -> List[Dict[str, np.ndarray]]:
    """Tracked sinusoids"""
    return list(self.tracker.all_tracks_)

  @property
  def _normalized_window(self) -> np.ndarray:
    """Normalized analysis window"""
    return self.w / np.sum(self.w)

  def pad_input(self, x: np.ndarray) -> Tuple[np.ndarray, int]:
    """Pad input at the beginning (so that the first window is centered at
    the first sample) and at the end (to analyze all samples)

    Args:
      x (array): The input array

    Returns:
      (array, int): The padded array and the initial padding length"""
    if not self.padded:
      return x, 0
    a = (self.w.size + 1) // 2
    b = self.w.size // 2
    y = np.zeros(x.size + a + b)
    y[a:(a + x.size)] = x
    return y, a

  def time_frames(self, x: np.ndarray) -> np.ndarray:
    """Generator of frames for a given input

    Args:
      x (array): Input

    Returns:
      generator: Generator of overlapping frames of the padded input"""
    y, _ = self.pad_input(x)
    return utils.dsp.overlapping_windows(y, self.w.size, self.h)

  def dft_frames(self,
                 x: np.ndarray) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    """Iterable of DFT frames for a given input

    Args:
      x (array): Input

    Returns:
      iterable: Iterable of overlapping DFT frames (magnitude and phase)
      of the padded input"""
    return map(functools.partial(sms.dsp.dft, w=self.w_, n=self.n),
               self.time_frames(x))
