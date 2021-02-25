"""Sinusoidal model"""
from sample.sms import dsp
import numpy as np
from sklearn import base
import functools
import itertools
from typing import Optional, Tuple, Generator, Iterable, Callable, Any, List, Dict


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


class SineTracker:
  """Model for keeping track of sinusiods across frames

  Args:
    max_n_sines (int): Maximum number of tracks per frame
    min_sine_dur (float): Minimum duration of a track in number of frames
    freq_dev_offset (float): Frequency deviation threshold at 0Hz
    freq_dev_slope (float): Slope of frequency deviation threshold

  Attributes:
    tracks_ (list of dict): Deactivated tracks"""
  def __init__(
    self,
    max_n_sines: int,
    min_sine_dur: float,
    freq_dev_offset: float,
    freq_dev_slope: float,
  ):
    self.max_n_sines = max_n_sines
    self.min_sine_dur = min_sine_dur
    self.freq_dev_offset = freq_dev_offset
    self.freq_dev_slope = freq_dev_slope
    self.tracks_ = []
    self._active_tracks = []
    self._frame = 0

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
    return len(self._active_tracks)

  @property
  def all_tracks_(self) -> Iterable[dict]:
    """All deactivated tracks in :attr:`tracks_` and those active tracks
    that would pass the cleanness check at the current state of the tracker"""
    return itertools.chain(
      self.tracks_,
      filter(self.track_ok, map(self.numpy_track, self._active_tracks))
    )

  def df(self, f: float) -> float:
    """Frequency deviation threshold at given frequency

    Args:
      f (float): Frequency in Hz

    Returns:
      float: Frequency deviation threshold in Hz"""
    return self.freq_dev_offset + self.freq_dev_slope * f

  @staticmethod
  def numpy_track(track: dict) -> dict:
    """Convert to numpy arrays all values in track

    Args:
      track (dict): Track to convert

    Returns:
      dict: Converted track"""
    return {
      k: np.array(v)
      for k, v in track.items()
    }

  def track_ok(self, track: dict) -> bool:
    """Check if deactivated track is ok to be saved

    Args:
      track (dict): Track to check

    Returns:
      bool: Whether the track is ok or not"""
    return len(track["freq"]) > self.min_sine_dur

  def deactivate(self, track_index: int) -> dict:
    """Remove track from list of active tracks and save it in
    :attr:`tracks_` if it meets cleanness criteria

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
    for t_i in filter(
      free_track.__getitem__,
      reversed(range(self.n_active_tracks))
    ):
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


class SinusoidalModel(base.TransformerMixin, base.BaseEstimator):
  """Model for sinusoidal tracking

  Args:
    fs (int): sampling frequency in Hz. Defaults to 44100
    w: Analysis window. Defaults to None (if None,
      the :attr:`default_window` is used)
    n (int): FFT size. Defaults to 2048
    h (int): Window hop size. Defaults to 500
    t (float): threshold in dB. Defaults to -90
    max_n_sines (int): Maximum number of tracks per frame. Defaults to 100
    min_sine_dur (float): Minimum duration of a track in seconds.
      Defaults to 0.04
    freq_dev_offset (float): Frequency deviation threshold at 0Hz.
      Defaults to 20
    freq_dev_slope (float): Slope of frequency deviation threshold.
      Defaults to 0.01
    sine_tracker_cls (type): Sine tracker class
    save_intermediate (bool): If True, save intermediate data structures in
      the attribute :attr:`intermediate_`. Defaults to False

  Attributes:
    w_ (array): Effective analysis window
    intermediate_ (dict): Dictionary of intermediate data structures"""
  def __init__(
    self,
    fs: int = 44100,
    w: Optional[np.ndarray] = None,
    n: int = 2048,
    h: int = 500,
    t: float = -90,
    max_n_sines: int = 100,
    min_sine_dur: float = 0.04,
    freq_dev_offset: float = 20,
    freq_dev_slope: float = 0.01,
    sine_tracker_cls: type = SineTracker,
    save_intermediate: bool = False,
  ):
    self.fs = fs
    self.w = w
    self.n = n
    self.h = h
    self.t = t
    self.max_n_sines = max_n_sines
    self.min_sine_dur = min_sine_dur
    self.freq_dev_offset = freq_dev_offset
    self.freq_dev_slope = freq_dev_slope
    self.sine_tracker_cls = sine_tracker_cls
    self.save_intermediate = save_intermediate

  def fit(
    self,
    x: np.ndarray,
    y=None,  # pylint: disable=W0613
    **kwargs
  ):
    """Analyze audio data

    Args:
      x (array): audio input
      y (ignored): exists for compatibility
      kwargs: Any parameter, overrides initialization

    Returns:
      SinusoidalModel: self
    """
    if hasattr(self, "intermediate_"):
      del self.intermediate_
    self.set_params(**kwargs)
    self.w_ = self.normalized_window
    self.sine_tracker_ = self.sine_tracker_cls(
      **self.sine_tracker_kwargs
    )

    for mx, px in map(
      functools.partial(self.intermediate, "stft"),
      self.dft_frames(x)
    ):
      ploc, pmag, pph = self.intermediate(
        "peaks",
        dsp.peak_detect_interp(mx, px, self.t)
      )
      pfreq = ploc * self.fs / self.n  # indices to frequencies in Hz
      self.sine_tracker_(pfreq, pmag, pph)
    return self

  @property
  def tracks_(self) -> List[Dict[str, np.ndarray]]:
    """Tracked sinusoids"""
    return list(self.sine_tracker_.all_tracks_)

  @property
  def sine_tracker_kwargs(self) -> dict:
    """Arguments for sine tracker initialization"""
    return dict(
      max_n_sines=self.max_n_sines,
      min_sine_dur=self.min_sine_dur * self.fs / self.h,
      freq_dev_offset=self.freq_dev_offset,
      freq_dev_slope=self.freq_dev_slope,
    )

  def intermediate(self, key: str, value):
    """Save intermediate results if :data:`save_intermediate` is True

    Arguments:
      key (str): Data name
      value: Data

    Returns:
      object: The input value"""
    if self.save_intermediate:
      if not hasattr(self, "intermediate_"):
        self.intermediate_ = {}
      if key not in self.intermediate_:
        self.intermediate_[key] = []
      self.intermediate_[key].append(value)
    return value

  @property
  def default_window(self) -> np.ndarray:
    """Default window (2001 samples Hamming window)"""
    return np.hamming(2001)

  @property
  def normalized_window(self) -> np.ndarray:
    """Normalized analysis window (if None, normalized default window)"""
    w = self.default_window if self.w is None else self.w
    return w / np.sum(w)

  def pad_input(self, x: np.ndarray) -> Tuple[np.ndarray, int]:
    """Pad input at the beginning (so that the first window is centered at
    the first sample) and at the end (to analyze all samples)

    Args:
      x (array): The input array

    Returns:
      (array, int): The padded array and the initial padding length"""
    a = (self.w_.size + 1) // 2
    b = self.w_.size // 2
    y = np.zeros(x.size + a + b)
    y[a:(a + x.size)] = x
    return y, a

  def time_frames(self, x: np.ndarray) -> Generator[np.ndarray, None, None]:
    """Generator of frames for a given input

    Args:
      x (array): Input

    Returns:
      generator: Generator of overlapping frames of the padded input"""
    y, a = self.pad_input(x)
    for i in range(a, y.size - a, self.h):
      yield y[(i - a):(i - a + self.w_.size)]

  def dft_frames(
    self,
    x: np.ndarray
  ) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    """Iterable of DFT frames for a given input

    Args:
      x (array): Input

    Returns:
      iterable: Iterable of overlapping DFT frames (magnitude and phase)
      of the padded input"""
    return map(
      functools.partial(dsp.dft, w=self.w_, n=self.n),
      self.time_frames(x)
    )
