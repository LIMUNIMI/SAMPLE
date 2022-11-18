"""Decision models for discriminating beating trajectories"""
import collections
from typing import Sequence, Tuple

import numpy as np
from scipy import stats
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

  def amplitude_residuals(
      self, t: np.ndarray, track: dict,
      beatsdrop: sample.beatsdrop.regression.BeatRegression,
      params: Tuple[float, float, float]) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the amplitude residuals (prediction errors) for the two methods

    Args:
      t (array): Time axis
      track (dict): Track
      beatsdrop (BeatRegression): Beat regression model
      params (triplet of floats): Frequency, decay, and amplitude

    Returns:
      array, array: Residuals for the linear and beat model"""
    track_a = utils.dsp.db2a(track["mag"])

    _, d, a = params
    a_lin = a * np.exp(-2 * t / d)
    a_biz, = beatsdrop.predict(t, "am")

    np.subtract(track_a, a_lin, out=a_lin)
    np.subtract(track_a, a_biz, out=a_biz)
    return a_lin, a_biz


_PearsonRResult = collections.namedtuple("_PearsonRResult",
                                         ("statistic", "pvalue"))


def _pearsonr(x, y):
  """Wrapper for :func:`scipy.stats.pearsonr`
  This avoids errors due to the API change in scipy"""
  r = stats.pearsonr(x, y)
  return _PearsonRResult(*r) if isinstance(r, tuple) else r


class ResidualCorrelationBeatDecisor(BeatDecisor):
  """Decide if a trajectory is a beat or not based on the correlation
  between linear- and beat-regression residuals

  Args:
    alpha (float): p-value threshold. If the two residuals are uncorrelated
      (pvalue > alpha), then the trajectory is considered a beat
    statistic (float): Correlation statistic threshold. If the two correlation
      coefficient is less than this value, then
      the trajectory is considered a beat
    intermediate (OptionalStorage): Optionally-activatable storage"""

  def __init__(self,
               alpha: float = 0.05,
               statistic: float = 0,
               intermediate: utils.learn.OptionalStorage = None,
               **kwargs) -> None:
    self.alpha = alpha
    self.statistic = statistic
    super().__init__(intermediate=intermediate, **kwargs)

  def decide_beat(self, i: int, t: np.ndarray, track: dict,
                  beatsdrop: sample.beatsdrop.regression.BeatRegression,
                  params: Tuple[float, float, float]) -> bool:
    """Decision function based on residuals correlation

    Args:
      i (int): Track index
      t (array): Time axis
      track (dict): Track
      beatsdrop (BeatRegression): Beat regression model
      params (triplet of floats): Frequency, decay, and amplitude

    Returns:
      bool: :data:`True` iff the correlation is not significant
      or is below the threshold"""
    r_lin, r_biz = self.amplitude_residuals(t=t,
                                            track=track,
                                            beatsdrop=beatsdrop,
                                            params=params)
    t = self.intermediate.append("test", _pearsonr(r_lin, r_biz), index=i)
    return t.pvalue > self.alpha or t.statistic < self.statistic
