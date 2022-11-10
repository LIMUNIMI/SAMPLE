"""SAMPLE class for use in GUI"""
import time
import tkinter as tk
from typing import Optional, Sequence, Tuple

import numpy as np
import throttle

import sample.beatsdrop.sample
import sample.sample
import sample.utils
import sample.utils.learn
from sample.sms import mm, sm

utils = sample.utils


class SinusoidalModel4GUI(mm.ModalModel):
  """Sinusoidal tracker for use in the GUI. For a full list of
  arguments see :class:`sample.sms.mm.ModalModel`

  Args:
    progressbar (optional): Progressbar widget for visualizing
      the peak tracking progress"""

  def __init__(self,
               progressbar: Optional[tk.Widget] = None,
               w: Optional[np.ndarray] = None,
               n: int = 2048,
               t: float = -90,
               tracker: sm.SineTracker = None,
               intermediate: utils.learn.OptionalStorage = None,
               **kwargs):
    self.progressbar = progressbar
    super().__init__(w=w,
                     n=n,
                     t=t,
                     tracker=tracker,
                     intermediate=intermediate,
                     **kwargs)

  @utils.learn.default_property
  def intermediate(self):
    """Optionally-activatable storage"""
    return utils.learn.OptionalStorage(save=True)

  def fit(self, x: np.ndarray, y=None, **kwargs):
    """Analyze audio data

    Args:
      x (array): audio input
      y (ignored): exists for compatibility
      kwargs: Any parameter, overrides initialization

    Returns:
      SinusoidalModel: self"""
    self.set_params(**kwargs)
    self.w_ = self._normalized_window
    self.progress_start(len(list(self.time_frames(x))))
    s = super().fit(x=x, y=y)
    return s

  def progress_start(self,
                     maximum: int,
                     value: int = 0,
                     delay: Optional[float] = None):
    """Start progressbar and set the maximum"""
    if self.progressbar is not None:
      self.progressbar["maximum"] = -1
      self.progressbar.config(value=value, maximum=maximum)
      if delay is not None:
        time.sleep(delay)
      self.progressbar.update()

  @throttle.wrap(.0125, 1)
  def progressbar_update(self, value: Optional[float] = None):
    """Update the progress bar. This function is throttled"""
    if self.progressbar is not None:
      if value is not None:
        self.progressbar.config(value=value)
      self.progressbar.update()

  def time_frames(self, x: np.ndarray):
    """Generator of frames for a given input. Also,
    updates the progressbar if one has been specified

    Args:
      x (array): Input

    Returns:
      generator: Generator of overlapping frames of the padded input"""
    it = super().time_frames(x)
    if self.progressbar is not None and self.progressbar["maximum"] > 0:

      def func(t):
        self.progressbar_update(value=t[0])
        return t[-1]

      it = map(func, enumerate(it))
    for f in it:
      yield f


class SAMPLE4GUIMixin:
  """Mixin class for using a SAMPLE model in the GUI"""

  @utils.learn.default_property
  def sinusoidal(self):
    """Sinusoidal analysis model"""
    return SinusoidalModel4GUI()

  def _fit_track(self, i: int, t: np.ndarray,
                 track: dict) -> Sequence[Tuple[float, float, float, float]]:
    if i == 0:
      self.sinusoidal.progress_start(len(self.sinusoidal.tracks_))
    params = super()._fit_track(i=i, t=t, track=track)
    self.sinusoidal.progressbar_update(i + 1)
    return params

  def fit(self, *args, **kwargs):
    s = super().fit(*args, **kwargs)
    self.sinusoidal.progress_start(1, 1, delay=0.1)
    return s


class SAMPLE4GUI(SAMPLE4GUIMixin, sample.sample.SAMPLE):
  """SAMPLE model for use in the GUI. For a full list of arguments
  see :class:`sample.sample.SAMPLE`"""


class SAMPLEBeatsDROP4GUI(SAMPLE4GUIMixin,
                          sample.beatsdrop.sample.SAMPLEBeatsDROP):
  """SAMPLE+BeatsDROP model for use in the GUI. For a full list of arguments
  see :class:`sample.beatsdrop.sample.SAMPLEBeatsDROP`"""
