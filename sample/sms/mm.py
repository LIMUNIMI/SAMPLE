"""Sinusoidal model with added functionality for modal sounds"""
from sample.sms import sm
from sklearn import linear_model
import numpy as np
import functools
from typing import Tuple, Optional, Dict, Callable, Iterable

TractT = Dict[str, np.ndarray]


def _track_diff_avg(t: TractT, u: TractT) -> float:
  """Average frequency difference

  Args:
    t (track): First track
    u (track): Second track

  Returns:
    float: Difference of average frequencies"""
  return np.abs(np.nanmean(t["freq"]) - np.nanmean(u["freq"]))


def _track_diff_single(t: TractT, u: TractT) -> float:
  """Single frequency difference: difference of the last frequency of
  :data:`t` and the first difference of :data:`u`

  Args:
    t (track): First track
    u (track): Second track

  Returns:
    float: Frequency difference"""
  return np.abs(t["freq"][-1] - u["freq"][0])


class ModalTracker(sm.SineTracker):
  """Sinusoidal tracker with added functionality for modal sounds:

  - check decreasing magnitude
  - check frequency constraints
  - check peak threshold
  - group non-contiguous tracks

  Args:
    max_n_sines (int): Maximum number of tracks per frame
    min_sine_dur (float): Minimum duration of a track in number of frames
    freq_dev_offset (float): Frequency deviation threshold at 0Hz
    freq_dev_slope (float): Slope of frequency deviation threshold
    frequency_bounds (float, float): Minimum and maximum accepted mean frequency
    peak_threshold (float): Minimum peak magnitude in dB for modal tracks
    reverse (bool): Whether the input audio will be reversed
    merge_strategy (str): Track merging strategy. Supported strategies are:
      :data:`"single"` (based on the frequency difference of the tail of older
      track and head of new track), :data:`"average"` (based on frequency
      difference of average frequencies)
    strip_t (float): Strip time (in frames). Tracks starting later than this
      time will be omitted from the track list. If :data:`None`, don't strip"""

  def __init__(
      self,
      max_n_sines: int,
      min_sine_dur: float,
      freq_dev_offset: float,
      freq_dev_slope: float,
      frequency_bounds: Tuple[Optional[float], Optional[float]],
      peak_threshold: float,
      reverse: bool,
      merge_strategy: str,
      strip_t: Optional[float],
  ):
    super().__init__(
        max_n_sines=max_n_sines,
        min_sine_dur=min_sine_dur,
        freq_dev_offset=freq_dev_offset,
        freq_dev_slope=freq_dev_slope,
    )
    self.frequency_bounds = frequency_bounds
    self.peak_threshold = peak_threshold
    self.reverse = reverse
    self.merge_strategy = merge_strategy
    self.strip_t = strip_t

  def track_ok(self, track: dict) -> bool:
    """Check if deactivated track is ok to be saved

    Args:
      track (dict): Track to check

    Returns:
      bool: Whether the track is ok or not"""
    if not super().track_ok(track):
      return False

    mf = np.mean(track["freq"])
    if self.frequency_bounds[0] is not None and mf < self.frequency_bounds[0]:
      return False
    if self.frequency_bounds[1] is not None and mf > self.frequency_bounds[1]:
      return False

    lm_x = np.arange(track["mag"].size).reshape((-1, 1))
    if self.reverse:
      lm_x = np.flip(lm_x)
    lm_y = track["mag"]
    lm = linear_model.LinearRegression().fit(lm_x, lm_y)
    if np.squeeze(lm.coef_) > 0:
      return False
    if np.squeeze(lm.intercept_) < self.peak_threshold:
      return False

    return True

  _diff_funcion_dict: Dict[str, Callable] = {
      "single": _track_diff_single,
      "average": _track_diff_avg,
  }

  @property
  def freq_diff_function(self) -> Callable:
    """Frequency difference function for current merge strategy"""
    if self.merge_strategy not in self._diff_funcion_dict:
      raise KeyError(
          f"merge strategy for object of type '{self.__class__.__name__}'" +
          " should be one of the following: " +
          ", ".join(f"'{k}'" for k in self._diff_funcion_dict))
    return self._diff_funcion_dict[self.merge_strategy]

  def deactivate(self, track_index: int) -> dict:
    """Remove track from list of active tracks and save it in
    :attr:`tracks_` if it meets cleanness criteria

    Args:
      track_index (int): Index of track to deactivate

    Returns:
      dict: Deactivated track"""
    t = self.numpy_track(self._active_tracks.pop(track_index))
    if self.track_ok(t):
      u, df = sm.min_key(
          # choose amongst previous tracks only
          filter(
              lambda u: (u["start_frame"] + u["mag"].size) < t["start_frame"],
              self.tracks_),
          # absolute difference of the two tracks according to the strategy
          functools.partial(self.freq_diff_function, u=t))
      if u is not None and df < self.df(t["freq"][0]):
        pad = np.full(t["start_frame"] - u["start_frame"] - u["mag"].size,
                      np.nan)
        for k in ("freq", "mag", "phase"):
          u[k] = np.concatenate((u[k], pad, t[k]), axis=None)
      else:
        self.tracks_.append(t)
    return t

  @property
  def all_tracks_(self) -> Iterable[TractT]:
    """All deactivated tracks in :attr:`tracks_` and those active tracks
    that would pass the cleanness check at the current state of the tracker"""
    tracks = super().all_tracks_
    if self.strip_t is None:
      return tracks

    def onset_ok(t):
      if self.reverse:
        o = self._frame - (t["start_frame"] + t["mag"].size)
      else:
        o = t["start_frame"]
      return o <= self.strip_t

    return filter(onset_ok, tracks)


class ModalModel(sm.SinusoidalModel):
  """Sinusoidal model with a :class:`ModalTracker` as sine tracker

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
    reverse (bool): Whether to process audio in reverse.
      Defaults to False
    sine_tracker_cls (type): Sine tracker class
    save_intermediate (bool): If True, save intermediate data structures in
      the attribute :attr:`intermediate_`. Defaults to False
    frequency_bounds (float, float): Minimum and maximum accepted mean frequency
    peak_threshold (float): Minimum peak magnitude (magnitude at time=0) in dB
      for modal tracks
    merge_strategy (str): Track merging strategy. Supported strategies are:
      :data:`"single"` (based on the frequency difference of the tail of older
      track and head of new track), :data:`"average"` (based on frequency
      difference of average frequencies)
    strip_t (float): Strip time (in seconds). Tracks starting later than this
      time will be omitted from the track list.
      Default is :data:`None` (don't strip)"""

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
      reverse: bool = False,
      sine_tracker_cls: type = ModalTracker,
      save_intermediate: bool = False,
      frequency_bounds: Tuple[Optional[float], Optional[float]] = (20, 16000),
      peak_threshold: float = -90,
      merge_strategy: str = "average",
      strip_t: Optional[float] = None,
  ):
    super().__init__(
        fs=fs,
        w=w,
        n=n,
        h=h,
        t=t,
        max_n_sines=max_n_sines,
        min_sine_dur=min_sine_dur,
        freq_dev_offset=freq_dev_offset,
        freq_dev_slope=freq_dev_slope,
        reverse=reverse,
        sine_tracker_cls=sine_tracker_cls,
        save_intermediate=save_intermediate,
    )
    self.frequency_bounds = frequency_bounds
    self.peak_threshold = peak_threshold
    self.merge_strategy = merge_strategy
    self.strip_t = strip_t

  @property
  def sine_tracker_kwargs(self) -> dict:
    """Arguments for sine tracker initialization"""
    kwargs = super().sine_tracker_kwargs
    if self.strip_t is None:
      strip_t = self.strip_t
    else:
      strip_t = int(self.strip_t * self.fs / self.h)
    kwargs.update(
        dict(
            frequency_bounds=self.frequency_bounds,
            peak_threshold=self.peak_threshold,
            reverse=self.reverse,
            merge_strategy=self.merge_strategy,
            strip_t=strip_t,
        ))
    return kwargs
