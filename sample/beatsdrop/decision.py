"""Decision models for discriminating beating trajectories"""
from typing import Sequence, Tuple

import numpy as np
from sklearn import base

import sample.beatsdrop.regression
import sample.sample
import sample.utils
import sample.utils.dsp
import sample.utils.learn

utils = sample.utils


class BeatDecisor(base.BaseEstimator):
  """Model responsible for deciding wether the trajectory is a beat or not"""

  def __init__(self,
               intermediate: utils.learn.OptionalStorage = None,
               **kwargs) -> None:
    self.intermediate = intermediate
    self.set_params(**kwargs)

  @utils.learn.default_property
  def intermediate(self):
    """Optionally-activatable storage"""
    return utils.learn.OptionalStorage()

  def output_params(
      self, params: Tuple[float, float, float]
  ) -> Sequence[Tuple[float, float, float, float]]:
    """Format single-partial parameters as output for the
    :meth:`sample.sample.SAMPLE._fit_track` method and phases

    Args:
      params (triplet of floats): Frequency, decay, and amplitude

    Returns:
      Parameters for :meth:`sample.sample.SAMPLE._fit_track` and phase"""
    return ((*params, 0),)

  def output_beat_params(
      self, beat_params: sample.beatsdrop.regression.BeatModelParams
  ) -> Sequence[Tuple[float, float, float, float]]:
    """Format beat parameters as output for the
    :meth:`sample.sample.SAMPLE._fit_track` method and phases

    Args:
      beat_params (octuplet of floats): a0, a1, f0, f1, d0, d1, p0, p1

    Returns:
      Parameters for :meth:`sample.sample.SAMPLE._fit_track` and phase"""
    a0, a1, f0, f1, d0, d1, p0, p1 = beat_params
    return ((f0, d0, a0, p0), (f1, d1, a1, p1))

  # pylint: disable=W0613
  def decide_beat(self, i: int, t: np.ndarray, track: dict,
                  beatsdrop: sample.beatsdrop.regression.BeatRegression,
                  params: Tuple[float, float, float]) -> bool:
    """Decision function. Should be overwritten by child classes.
    This method always decides for the single-partial parameters
    (:data:`False`) and never for the beats (:data:`True`)

    Args:
      i (int): Track index
      t (array): Time axis
      track (dict): Track
      beatsdrop (BeatRegression): Beat regression model
      params (triplet of floats): Frequency, decay, and amplitude

    Returns:
      bool: Decision"""
    return False

  def track_params(
      self,
      i: int,
      t: np.ndarray,
      track: dict,
      beatsdrop: sample.beatsdrop.regression.BeatRegression,
      params: Tuple[float, float, float],
      fit: bool = False) -> Sequence[Tuple[float, float, float, float]]:
    """Returns linear or beat parameters depending on the decision

    Args:
      i (int): Track index
      t (array): Time axis
      track (dict): Track
      beatsdrop (BeatRegression): Beat regression model
      params (triplet of floats): Frequency, decay, and amplitude
      fit (bool): If :data:`True`, then fit the :data:`beatsdrop` model

    Returns:
      Parameters for the :meth:`sample.sample.SAMPLE._fit_track` method"""
    if fit:
      beatsdrop.fit(t=t, a=track["mag"], f=track["freq"])
    d = self.intermediate.append("decision",
                                 self.decide_beat(i=i,
                                                  t=t,
                                                  track=track,
                                                  beatsdrop=beatsdrop,
                                                  params=params),
                                 index=i)
    return self.output_beat_params(
        beatsdrop.params_) if d else self.output_params(params)


class AlmostNotABeatDecisor(BeatDecisor):
  """Decide if a trajectory is a beat or not based on the fact that the two
  amplitudes are not almost zero and the two frequencies are not almost the
  same.

  Args:
    th (float): Inexact equality threshold, as a multiplier of floating
      point epsilon.
    intermediate (OptionalStorage): Optionally-activatable storage"""

  def __init__(self,
               th: float = 1,
               intermediate: utils.learn.OptionalStorage = None,
               **kwargs) -> None:
    self.th = th
    super().__init__(intermediate=intermediate, **kwargs)

  @property
  def _th(self) -> float:
    return self.th * np.finfo(float).eps

  def _not_amost_zero(self, values):
    return np.greater(values, self._th)

  def decide_beat(self, i: int, t: np.ndarray, track: dict,
                  beatsdrop: sample.beatsdrop.regression.BeatRegression,
                  params: Tuple[float, float, float]) -> bool:
    """Decision function based on normalized power square difference

    Args:
      i (int): Track index
      t (array): Time axis
      track (dict): Track
      beatsdrop (BeatRegression): Beat regression model
      params (triplet of floats): Frequency, decay, and amplitude

    Returns:
      bool: :data:`True` iff the correlation is not significant
      or is below the threshold"""
    a0, a1, f0, f1, _, _, _, _ = beatsdrop.params_

    d = self.intermediate.append("test", np.abs((f0 - f1, a0, a1)), index=i)
    return np.all(self._not_amost_zero(d))
