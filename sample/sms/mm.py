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
    frequency_bounds (float, float): Minimum and maximum accepted mean frequency
    peak_threshold (float): Minimum peak magnitude in dB for modal tracks
    kwargs: Keyword arguments, see :class:`sample.sms.sm.SineTracker`"""
  def __init__(
    self,
    frequency_bounds: Tuple[Optional[float], Optional[float]],
    peak_threshold: float,
    **kwargs,
  ):
    super().__init__(**kwargs)
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
      u, df = sm._min_key(
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
    frequency_bounds (float, float): Minimum and maximum accepted mean frequency
    peak_threshold (float): Minimum peak magnitude in dB for modal tracks
    (magnitude at time=0)
    kwargs: Keyword arguments, see :class:`sample.sms.sm.SinusoidalModel`"""
  def __init__(
    self,
    sine_tracker_cls: type = ModalTracker,
    frequency_bounds: Tuple[Optional[float], Optional[float]] = (20, 16000),
    peak_threshold: float = -90,
    **kwargs
  ):
    super().__init__(
      sine_tracker_cls=sine_tracker_cls,
      **kwargs,
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
