"""SAMPLE class for use in GUI"""
from sample import sample
from sample.sms import mm
import tkinter as tk
import numpy as np
import throttle
from typing import Optional, Tuple


class SAMPLE(sample.SAMPLE):
  """SAMPLE model for use in the GUI. For a full list of arguments see
  :class:`sample.sample.SAMPLE`"""

  class SinusoidalModel(mm.ModalModel):
    """Sinusoidal tracker for use in the GUI. For a full list of
    arguments see :class:`sample.sms.mm.ModalModel`

    Args:
      progressbar (optional): Progressbar widget for visualizing
        the peak tracking progress"""

    def __init__(
        self,
        progressbar: Optional[tk.Widget] = None,
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
        sine_tracker_cls: type = mm.ModalTracker,
        save_intermediate: bool = False,
        frequency_bounds: Tuple[Optional[float], Optional[float]] = (20, 16000),
        peak_threshold: float = -90,
        merge_strategy: str = "average",
        strip_t: Optional[float] = None,
    ):
      self.progressbar = progressbar
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
          frequency_bounds=frequency_bounds,
          peak_threshold=peak_threshold,
          merge_strategy=merge_strategy,
          strip_t=strip_t,
      )

    def fit(self, x: np.ndarray, y=None, **kwargs):
      """Analyze audio data

      Args:
        x (array): audio input
        y (ignored): exists for compatibility
        kwargs: Any parameter, overrides initialization

      Returns:
        SinusoidalModel: self"""
      self.set_params(**kwargs)
      self.w_ = self.normalized_window
      if self.progressbar is not None:
        self.progressbar["maximum"] = -1
        self.progressbar.config(value=0, maximum=len(list(self.time_frames(x))))
      s = super().fit(x=x, y=y)
      if self.progressbar is not None:
        self.progressbar.config(value=1, maximum=1)
      return s

    @throttle.wrap(.0125, 1)
    def progressbar_update(self, value: Optional[float] = None):
      """Update the progress bar. This function is throttled"""
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

  def __init__(self, sinusoidal_model=SinusoidalModel(), **kwargs):
    super().__init__(sinusoidal_model=sinusoidal_model, **kwargs)
