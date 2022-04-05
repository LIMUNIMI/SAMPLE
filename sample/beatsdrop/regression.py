"""Regression models for beats"""
from typing import Callable, Optional, Tuple

import numpy as np
from sample import beatsdrop
from sample.utils import dsp as dsp_utils
from sample.sms import dsp as sms_dsp
from scipy import optimize
from sklearn import base, linear_model

BeatModelParams = Tuple[float, float, float, float, float, float, float, float]
BeatParamsInit = Callable[
    [np.ndarray, np.ndarray, np.ndarray, "BeatRegression"], BeatModelParams]
BeatBoundsFunc = Callable[
    [np.ndarray, np.ndarray, np.ndarray, BeatModelParams, "BeatRegression"],
    Tuple[BeatModelParams, BeatModelParams]]


class BeatRegression(base.RegressorMixin, base.BaseEstimator):
  """Regressor for fitting to a beat pattern

  Args:
    fs (float): Sampling frequency for beat model integration
    lpf (float): Corner frequency for LPF of spectrum in autocorrelation
      computation
    linear_regressor (sklearn.base.BaseEstimator): Linear regression model
      instance. Must be sklearn-compatible
    linear_regressor_k (str): Attribute name for the estimated slope
      coefficient of the linear regression
    linear_regressor_q(str): Attribute name for the estimated intercept
      coefficient of the linear regression
    params_init (callable): Initializer for beat model parameters.
      Signature should be
      :data:`f(t, a, f, model) -> a0, a1, f0, f1, d0, d1, p0, p1`.
      It should return initial parameters for nonlinear least squares using
      input data :data:`t`, :data:`a`, and :data:`f`, and the
      :class:`BeatRegression` instance :data:`model`. If :data:`None`,
      use default
    bounds (callable): Callable for computing beat model coefficient
      boundaries. Signature should be
      :data:`bounds(t, a, f, p, model) -> (bounds_min, bounds_max)`.
      It should return lower and upper boundaries for all eight parameters
      using input data :data:`t`, :data:`a`, and :data:`f`, initial parameter
      estimates :data:`p`, and the :class:`BeatRegression` instance
      :data:`model`. If :data:`None`, use default"""

  def __init__(
      self,
      fs: float = 44100,
      lpf: float = 12,
      linear_regressor=None,
      linear_regressor_k: str = "coef_",
      linear_regressor_q: str = "intercept_",
      params_init: Optional[BeatParamsInit] = None,
      bounds: Optional[BeatBoundsFunc] = None,
  ):
    super().__init__()
    self.fs = fs
    self.lpf = lpf
    self.linear_regressor = linear_regressor
    self.linear_regressor_k = linear_regressor_k
    self.linear_regressor_q = linear_regressor_q
    self.params_init = params_init
    self.bounds = bounds

  def _residual_fn(self, t: np.ndarray, a: np.ndarray,
                   f: np.ndarray) -> Callable:  # pylint: disable=W0613
    """Residual function for the provided data

    Args:
      t (array): Time. Must be a column vector (shape like (-1, 1))
      a (array): Amplitude at time :data:`t`
      f (array): Frequency at time :data:`t`

    Returns:
      callable: Residual function"""
    # Time axis for model integration
    t_min = np.min(t)
    t_max = np.max(t)
    u = np.arange(np.ceil((t_max - t_min) * self.fs).astype(int), dtype=float)
    np.true_divide(u, self.fs, out=u)
    a_lin = dsp_utils.db2a(a)

    def _residual_fn_(beat_args: BeatModelParams) -> np.ndarray:
      a_est = np.interp(t, u, beatsdrop.ModalBeat(*beat_args).am(u))
      return np.reshape(np.subtract(a_lin, a_est), newshape=(-1,))

    return _residual_fn_

  @property
  def k_(self) -> float:
    """Linear regression slope"""
    return np.squeeze(getattr(self.linear_regressor, self.linear_regressor_k))

  @property
  def q_(self) -> float:
    """Linear regression intercept"""
    return np.squeeze(getattr(self.linear_regressor, self.linear_regressor_q))

  @staticmethod
  def _default_params_init(t: np.ndarray, a: np.ndarray, f: np.ndarray,
                           model: "BeatRegression") -> BeatModelParams:
    """Default parameter initializer

    Args:
      t (array): Time. Must be a column vector (shape like (-1, 1))
      a (array): Amplitude at time :data:`t` (in dB)
      f (array): Frequency at time :data:`t`
      model (BeatRegression): Regression model

    Returns:
      (float, float, float, float, float, float, float, float): Starting
      values for nonlinear least squares for a0, a1, f0, f1, d0, d1, p0,
      and p1"""
    a_dt = dsp_utils.db2a(dsp_utils.detrend(a, t, model=model.linear_regressor))
    corr = dsp_utils.lombscargle_autocorrelogram(t,
                                                 a_dt,
                                                 fs=model.fs,
                                                 lpf=model.lpf)

    # Amplitudes
    a0 = a1 = dsp_utils.db2a(model.q_) * 2
    # Frequencies
    carrier_freq = np.mean(f)
    am_freq, _ = sms_dsp.peak_detect_interp(corr)
    am_freq = model.fs / (2 * am_freq[0])
    f0 = carrier_freq + am_freq
    f1 = carrier_freq - am_freq
    # Decays
    d0 = d1 = -40 * np.log10(np.e) / model.k_
    # Phases
    # without loss of generality, consider p0 = 0
    p_hat = np.mod(
        np.angle(
            np.dot(dsp_utils.db2a(a), dsp_utils.expi(-4 * np.pi * am_freq * t)))
        / 2, np.pi)

    return a0, a1, f0, f1, d0, d1, 0.0, -2 * p_hat

  @property
  def _params_init(self) -> BeatParamsInit:
    """Parameters initializer

    Returns:
      User-provided function if given, otherwise
        :meth:`_default_params_init`"""
    if self.params_init is None:
      return self._default_params_init
    return self.params_init

  @staticmethod
  def _default_bounds(
      t: np.ndarray, a: np.ndarray, f: np.ndarray, p: BeatModelParams,
      model: "BeatRegression"
  ) -> Tuple[BeatModelParams, BeatModelParams]:  # pylint: disable=W0613
    """Default boundaries

    Args:
      t (array): Time. Must be a column vector (shape like (-1, 1))
      a (array): Amplitude at time :data:`t` (in dB)
      f (array): Frequency at time :data:`t`
      p (tuple): Initial parameters
      model (BeatRegression): Regression model

    Returns:
      (tuple, tuple): Minimum and maximum bounds for a0, a1, f0, f1, d0, d1,
      p0, and p1"""
    if len(t) <= 1:
      raise ValueError(f"Got a track of length={len(t)}. "
                       "Consider increasing the minimum sine length")
    # Amplitude bounds
    a_lin = dsp_utils.db2a(a)
    a_min = np.min(a_lin)
    a_max = np.max(a_lin)
    if a_min == a_max:
      a_min = 0.0
      if a_min == a_max:
        a_max = 1.0
    else:
      a_d = a_max - a_min
      a_max += a_d
      a_min = max(0.0, a_min - a_d)

    # Frequency bounds
    f_min = np.min(f)
    f_max = np.max(f)

    eps = np.finfo(float).eps

    bounds_min = (a_min, a_min, f_min, f_min, eps, eps, -np.pi, p[-1] - np.pi)
    bounds_max = (a_max, a_max, f_max, f_max, 100., 100., np.pi, p[-1] + np.pi)
    return bounds_min, bounds_max

  @property
  def _bounds(self) -> BeatBoundsFunc:
    """Boundary function

    Returns:
      User-provided function if given, otherwise
        :meth:`_default_bounds`"""
    if self.bounds is None:
      return self._default_bounds
    return self.bounds

  def fit(self,
          t: np.ndarray,
          a: np.ndarray,
          f: np.ndarray,
          method: str = "dogbox",
          **kwargs):
    """Fit beat pattern

    Args:
      t (array): Time. Must be a column vector (shape like (-1, 1))
      a (array): Amplitude at time :data:`t` (in dB)
      f (array): Frequency at time :data:`t`
      **kwargs: Keyword arguments for :func:`scipy.optimize.least_squares`

    Returns:
      BeatRegression: self"""
    if self.linear_regressor is None:
      self.linear_regressor = linear_model.LinearRegression()
    self.initial_params_ = self._params_init(t, a, f, self)  # pylint: disable=E1102
    self.bounds_ = self._bounds(t, a, f, self.initial_params_, self)  # pylint: disable=E1102
    self.result_ = optimize.least_squares(fun=self._residual_fn(t, a, f),
                                          x0=self.initial_params_,
                                          bounds=self.bounds_,
                                          method=method,
                                          **kwargs)
    self.params_ = np.concatenate((self.result_.x[:2] * 2, self.result_.x[2:]))
    return self
