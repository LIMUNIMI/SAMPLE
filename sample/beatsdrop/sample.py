"""Module for integrating BeatsDROP in the SAMPLE model"""
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
from sklearn import base

import sample.beatsdrop
import sample.beatsdrop.decision
import sample.beatsdrop.regression
import sample.sample
import sample.utils
import sample.utils.dsp
import sample.utils.learn
from sample import hinge
from sample.sms import sm

utils = sample.utils
bd = sample.beatsdrop


class SAMPLEBeatsDROP(sample.sample.SAMPLE):
  """SAMPLE model with BeatsDROP beat decoupling

  Args:
    beatsdrop (BeatRegression): Beat regression model. Default is an instance
      of :class:`sample.beatsdrop.regression.DualBeatRegression`
    beat_decisor (BeatDecisor): Model responsible for deciding wether the
      trajectory is a beat or not. By default it is an instance of
      :class:`sample.beatsdrop.decision.AlmostNotABeatDecisor`
    sinusoidal (SinusoidalModel): Sinusoidal analysis model.
      Default is an instance of :class:`sample.sms.mm.ModalModel`
    regressor: Modal parameters regression model.
      Default is an instance of :class:`sample.hinge.HingeRegression`
    regressor_k (str): Attribute name for the estimated slope
      coefficient of :data:`regressor`
    regressor_q (str): Attribute name for the estimated intercept
      coefficient of :data:`regressor`
    freq_reduce (callable): Callable function for reducing the frequency track
      into a single frequency. Defaults to :func:`numpy.mean`
    max_n_modes (int): Number of modes to use in resynthesis. If :data:`None`
      (default), then synthesise all modes
    **kwargs: Additional parameters for sub-models. See
      :class:`sample.beatsdrop.regression.DualBeatRegression`,
      :class:`sample.beatsdrop.decision.AlmostNotABeatDecisor`,
      :class:`sample.sms.mm.ModalTracker`,
      :class:`sample.sms.mm.ModalModel`,
      :class:`sample.hinge.HingeRegression`, and
      :class:`sample.utils.learn.OptionalStorage`

  Attributes:
    param_matrix_ (array): 4-by-N matrix of modal parameters"""

  def __init__(
      self,
      beatsdrop: bd.regression.BeatRegression = None,
      beat_decisor: bd.decision.BeatDecisor = None,
      sinusoidal: sm.SinusoidalModel = None,
      regressor: hinge.HingeRegression = None,
      regressor_k: str = "k_",
      regressor_q: str = "q_",
      freq_reduce: Callable[[np.ndarray], float] = np.mean,
      max_n_modes: Optional[int] = None,
      **kwargs,
  ):
    self.beatsdrop = beatsdrop
    self.beat_decisor = beat_decisor
    super().__init__(sinusoidal=sinusoidal,
                     regressor=regressor,
                     regressor_k=regressor_k,
                     regressor_q=regressor_q,
                     freq_reduce=freq_reduce,
                     max_n_modes=max_n_modes,
                     **kwargs)

  @utils.learn.default_property
  def beatsdrop(self):
    """Beat regression model"""
    return bd.regression.DualBeatRegression()

  @utils.learn.default_property
  def beat_decisor(self):
    """Beat decision model"""
    return bd.decision.AlmostNotABeatDecisor()

  _PARAM_MATRIX_NROWS: int = 4

  def _fit_track(self, i: int, t: np.ndarray,
                 track: dict) -> Sequence[Tuple[float, float, float, float]]:
    """Fit parameters for one track.

    Args:
      i (int): Track index
      t (array): Time axis
      track (dict): Track

    Returns:
      ((float, float, float),): Frequency, decay, and amplitude"""
    b = base.clone(self.beatsdrop)
    b.set_params(fs=self.sinusoidal.fs)
    return self.beat_decisor.track_params(i=i,
                                          t=t,
                                          track=track,
                                          beatsdrop=b,
                                          params=super()._fit_track(
                                              i, t, track)[0],
                                          fit=True)

  @property
  def phases_(self) -> np.ndarray:
    """Learned sinusoidal phases. Relevant between pairs of beating partials"""
    return self.param_matrix_[3, :]

  def predict(self, x: np.ndarray, **kwargs) -> np.ndarray:
    """Resynthesize audio

    Args:
      x (array): Time axis
      **kwargs: Keyword arguments for :meth:`sample.sample.SAMPLE.predict`

    Returns:
      array: Array of audio samples"""
    if "phases" not in kwargs:
      kwargs["phases"] = self.phases_
    return super().predict(x, **kwargs)
