"""Sinusoidal model with added functionality for modal sounds"""
from sample.sms import sm
from sklearn import linear_model
import numpy as np
from typing import Tuple, Optional


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
    peak_threshold (float): Minimum peak magnitude in dB for modal tracks"""
  def __init__(
    self,
    max_n_sines: int,
    min_sine_dur: float,
    freq_dev_offset: float,
    freq_dev_slope: float,
    frequency_bounds: Tuple[Optional[float], Optional[float]],
    peak_threshold: float,
  ):
    super().__init__(
      max_n_sines=max_n_sines,
      min_sine_dur=min_sine_dur,
      freq_dev_offset=freq_dev_offset,
      freq_dev_slope=freq_dev_slope,
    )
    self.frequency_bounds = frequency_bounds
    self.peak_threshold = peak_threshold

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

    lm = linear_model.LinearRegression().fit(
      np.arange(track["mag"].size).reshape((-1, 1)),
      track["mag"]
    )
    if np.squeeze(lm.coef_) > 0:
      return False
    if np.squeeze(lm.intercept_) < self.peak_threshold:
      return False

    return True

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
          self.tracks_
        ),
        # absolute difference from last peak's frequency
        lambda u: np.abs(u["freq"][-1] - t["freq"][0])
      )
      if u is not None and df < self.df(t["freq"][0]):
        pad = np.full(
          t["start_frame"] - u["start_frame"] - u["mag"].size,
          np.nan
        )
        for k in ("freq", "mag", "phase"):
          u[k] = np.concatenate((
            u[k], pad, t[k]
          ), axis=None)
      else:
        self.tracks_.append(t)
    return t


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
    sine_tracker_cls (type): Sine tracker class
    save_intermediate (bool): If True, save intermediate data structures in
      the attribute :attr:`intermediate_`. Defaults to False
    frequency_bounds (float, float): Minimum and maximum accepted mean frequency
    peak_threshold (float): Minimum peak magnitude (magnitude at time=0) in dB
      for modal tracks"""
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
    sine_tracker_cls: type = ModalTracker,
    save_intermediate: bool = False,
    frequency_bounds: Tuple[Optional[float], Optional[float]] = (20, 16000),
    peak_threshold: float = -90,
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
      sine_tracker_cls=sine_tracker_cls,
      save_intermediate=save_intermediate,
    )
    self.frequency_bounds = frequency_bounds
    self.peak_threshold = peak_threshold

  @property
  def sine_tracker_kwargs(self) -> dict:
    """Arguments for sine tracker initialization"""
    kwargs = super().sine_tracker_kwargs
    kwargs.update(dict(
      frequency_bounds=self.frequency_bounds,
      peak_threshold=self.peak_threshold,
    ))
    return kwargs
