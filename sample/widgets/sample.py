"""SAMPLE class for use in GUI"""
import itertools
import threading
from typing import Optional

import throttle

import sample._training
import sample.beatsdrop
import sample.beatsdrop.sample
import sample.sample
import sample.utils
import sample.utils.learn
from sample.widgets import logging

bd = sample.beatsdrop
utils = sample.utils


class _GUIFitArgs(sample._training.FitArgs):  # pylint: disable=W0212
  """Fit arguments for GUI"""

  def __init__(self, starmap=itertools.starmap, progressbar=None) -> None:
    self._current_progress = 0
    self._lock = threading.Lock()
    super().__init__(starmap=starmap, progressbar=progressbar)

  def progress_start(self, maximum: int, value: int = 0):
    """Start progressbar and set the maximum"""
    with self._lock:
      self._current_progress = value
      if self.progressbar is not None:
        self.progressbar["maximum"] = -1
        self.progressbar.config(value=value, maximum=maximum)
        self.progressbar.update()

  def progress_stop(self):
    """Fill progressbar and reset the maximum"""
    self.progress_start(1, 1)

  def progress_update(self, value: Optional[float] = None):
    """Update the progress bar"""
    with self._lock:
      if value is None:
        value = self._current_progress + 1
      self._current_progress = value
      self._progress_update_inner(value=value)

  @throttle.wrap(0.2, 1)
  def _progress_update_inner(self, value: Optional[float] = None):
    """Update the progress bar. This function is throttled"""
    if self.progressbar is not None:
      if value is not None:
        self.progressbar.config(value=value)
      self.progressbar.update()


_default_kwargs = {
    "beat_decisor__intermediate__save": True,
    "sinusoidal__intermediate__save": True,
}


def sample_factory(beatsdrop: bool = False,
                   progressbar=None,
                   n_jobs: int = 0,
                   **kwargs):
  """Factory function for SAMPLE models for the GUI

  Args:
    beatsdrop (bool): Whether to use BeatsDROP or not
    n_jobs (int): Number of workers
    progressbar: Progressbar widget
    **kwargs: Parameters for the SAMPLE model

  Returns:
    SAMPLE, dict: Model and fit function arguments"""
  kwargs.update((k, v) for k, v in _default_kwargs.items() if k not in kwargs)
  cls = bd.sample.SAMPLEBeatsDROP if beatsdrop else sample.sample.SAMPLE
  self = cls()
  ok_kwargs = dict(filter(lambda t: t[0] in self.get_params(), kwargs.items()))
  logging.info("Building SAMPLE object of class %s with arguments: %s",
               cls.__name__, ok_kwargs)
  self.set_params(**ok_kwargs)
  fit_kwargs = {
      "_fit_args": _GUIFitArgs(progressbar=progressbar),
      "n_jobs": n_jobs
  }
  return self, fit_kwargs
