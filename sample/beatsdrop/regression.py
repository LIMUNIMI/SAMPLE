"""Regression models for beats"""
import inspect
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
from scipy import optimize
from sklearn import base, linear_model

import sample.sample
import sample.utils
import sample.utils.dsp
import sample.utils.learn
from sample import beatsdrop, psycho
from sample.sms import dsp as sms_dsp

utils = sample.utils

BeatModelParams = Tuple[float, float, float, float, float, float, float, float]
BeatParamsInit = Callable[
    [np.ndarray, np.ndarray, np.ndarray, "BeatRegression"], BeatModelParams]
BeatBoundsFunc = Callable[
    [np.ndarray, np.ndarray, np.ndarray, BeatModelParams, "BeatRegression"],
    Tuple[BeatModelParams, BeatModelParams]]
BeatResidualFunc = Callable[[BeatModelParams], np.ndarray]


def _get_notnone_attr(obj: Any, *args) -> Any:
  """Return the first attribute amongst the ones provided that the object has
  and that is not :data:`None`

  Args:
    obj (object): Object
    *args: Names of the attributes to inspect. The first one that exists and
      is not :data:`None` is returned. Arguments coming after are not evaluated

  Returns:
    object: The first valid argument"""
  for k in args:
    if hasattr(obj, k):
      v = getattr(obj, k)
      if v is not None:
        return v
  raise AttributeError(
      "No valid values found for attributes of object of "
      f"class '{type(obj).__name__}': {utils.comma_join_quote(args)}")


def _param_energies(p: BeatModelParams) -> Tuple[float, float]:
  """Compute the energies from the parameters

  Args:
    p (tuple): Beat model parameters

  Returns:
    float, float: The enrgies of the two partials"""
  return sample.sample.modal_energy(p[:2], p[4:6])


def sort_params(p: BeatModelParams,
                key: Callable[[BeatModelParams],
                              Tuple[float, float]] = _param_energies,
                descending: bool = False) -> BeatModelParams:
  """Sort beat model parameters

  Args:
    p (tuple): Beat model parameters
    key (callable): Function with respect to which to sort.
      Defaults to modal energy
    descending (bool): If :data:`True`, then sort in descending order

  Returns:
    tuple: Sorted beat model parameters (maintaining association)"""
  is_descending: bool = np.argmin(key(p)).astype(bool)
  if descending ^ is_descending:
    p = (p[1], p[0], p[3], p[2], p[5], p[4], p[7], p[6])
  return p


class BeatRegression(base.RegressorMixin, base.BaseEstimator):
  """Regressor for fitting to a beat pattern

  Args:
    fs (float): Sampling frequency for autocorrelogram computation
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
      :data:`f(t, a, f, res_fn, model) -> a0, a1, f0, f1, d0, d1, p0, p1`.
      It should return initial parameters for nonlinear least squares using
      input data :data:`t`, :data:`a`, and :data:`f`, the residual function
      :data:`res_fn`, and the :class:`BeatRegression` instance :data:`model`.
      If :data:`None`, use default
    bounds (callable): Callable for computing beat model coefficient
      boundaries. Signature should be
      :data:`bounds(t, a, f, p, model) -> (bounds_min, bounds_max)`.
      It should return lower and upper boundaries for all eight parameters
      using input data :data:`t`, :data:`a`, and :data:`f`, initial parameter
      estimates :data:`p`, and the :class:`BeatRegression` instance
      :data:`model`. If :data:`None`, use default"""

  def __init__(
      self,
      fs: float = 100,
      lpf: float = 50,
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
                   f: np.ndarray) -> BeatResidualFunc:  # pylint: disable=W0613
    """Residual function for the provided data

    Args:
      t (array): Time. Must be a column vector (shape like (-1, 1))
      a (array): Amplitude at time :data:`t`
      f (array): Frequency at time :data:`t`

    Returns:
      callable: Residual function"""
    # Time axis for model integration
    t_max = np.max(t)
    u = np.arange(np.ceil(t_max * self.fs).astype(int), dtype=float)
    np.true_divide(u, self.fs, out=u)
    a_lin = utils.dsp.db2a(a)

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

  @utils.learn.default_property
  def params_init(self):
    return self._default_params_init

  @utils.learn.default_property
  def bounds(self):
    return self._default_bounds

  @staticmethod
  def _default_params_init(
      t: np.ndarray,
      a: np.ndarray,
      f: np.ndarray,
      res_fn: BeatResidualFunc,  # pylint: disable=W0613
      model: "BeatRegression") -> BeatModelParams:
    """Default parameter initializer

    Args:
      t (array): Time. Must be a column vector (shape like (-1, 1))
      a (array): Amplitude at time :data:`t` (in dB)
      f (array): Frequency at time :data:`t`
      res_fn (callable): Residual function
      model (BeatRegression): Regression model

    Returns:
      (float, float, float, float, float, float, float, float): Starting
      values for nonlinear least squares for a0, a1, f0, f1, d0, d1, p0,
      and p1"""
    a_dt = utils.dsp.db2a(utils.dsp.detrend(a, t, model=model.linear_regressor))
    corr = utils.dsp.lombscargle_autocorrelogram(t,
                                                 a_dt,
                                                 fs=model.fs,
                                                 lpf=model.lpf)

    # Amplitudes
    a0 = a1 = utils.dsp.db2a(model.q_) / 2
    # Frequencies
    carrier_freq = np.mean(f)
    am_freq, _ = sms_dsp.peak_detect_interp(corr)
    am_freq = 0 if len(am_freq) == 0 else model.fs / (2 * am_freq[0])
    f0 = carrier_freq + am_freq
    f1 = carrier_freq - am_freq
    # Decays
    d0 = d1 = -40 * np.log10(np.e) / model.k_
    # Phases
    # without loss of generality, consider p0 = 0
    p_hat = np.mod(
        np.angle(
            np.dot(utils.dsp.db2a(a), utils.dsp.expi(-4 * np.pi * am_freq * t)))
        / 2, np.pi)

    return a0, a1, f0, f1, d0, d1, 0.0, -2 * p_hat

  @staticmethod
  def _default_bounds(
      t: np.ndarray,
      a: np.ndarray,
      f: np.ndarray,
      p: BeatModelParams,
      model: "BeatRegression"  # pylint: disable=W0613
  ) -> Tuple[BeatModelParams, BeatModelParams]:
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
    a_lin = utils.dsp.db2a(a)
    a_min = np.min(a_lin)
    a_max = np.max(a_lin)
    if a_min == a_max:
      a_min = 0.0
      if a_min == a_max:
        a_max = 1.0
    else:
      a_max = max(a_max, *p[:2])
      a_min = min(a_min, *p[:2])
      a_d = a_max - a_min
      a_max += a_d
      a_min = max(0.0, a_min - a_d)

    # Frequency bounds
    f_min = np.min(f)
    f_max = np.max(f)
    df = min(f_max - f_min, 1)
    f_max = max(*p[2:4], f_max) + df
    f_min = min(*p[2:4], f_min) - df

    eps = np.finfo(float).eps

    bounds_min = (a_min, a_min, f_min, f_min, eps, eps, -np.pi, p[-1] - np.pi)
    bounds_max = (a_max, a_max, f_max, f_max, 100., 100., np.pi, p[-1] + np.pi)
    return bounds_min, bounds_max

  def fit(self,
          t: np.ndarray,
          a: np.ndarray,
          f: np.ndarray,
          tr_solver: Optional[str] = "lsmr",
          method: str = "dogbox",
          loss: str = "cauchy",
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
    self.res_fn_ = self._residual_fn(t, a, f)
    self.initial_params_ = self.params_init(t, a, f, self.res_fn_, self)
    self.bounds_ = self.bounds(t, a, f, self.initial_params_, self)
    feasible = np.logical_and(np.less(self.bounds_[0], self.initial_params_),
                              np.less(self.initial_params_, self.bounds_[1]))
    if not np.all(feasible):
      nl = "\n"
      msg = "Starting value is unfeasible" + "".join(
          f"{nl}  {k}={x0} not between"
          f"{nl}   {' ' * len(k)}{lb} and"
          f"{nl}   {' ' * len(k)}{ub}" for f, k, lb, ub, x0 in zip(
              feasible,
              inspect.signature(beatsdrop.ModalBeat).parameters, *self.bounds_,
              self.initial_params_) if not f)
      raise ValueError(msg)
    self.result_ = optimize.least_squares(fun=self.res_fn_,
                                          x0=self.initial_params_,
                                          bounds=self.bounds_,
                                          tr_solver=tr_solver,
                                          method=method,
                                          loss=loss,
                                          **kwargs)
    self.params_ = self.result_.x
    self.beat_ = beatsdrop.ModalBeat(*self.params_)
    return self

  def predict(self, t: np.ndarray, *args) -> List[np.ndarray]:
    """Predict beat pattern

    Args:
      t (array): Time
      args: The names of the outputs to predict. If none is specified,
        then the audio signal (:data:`"x"`) is predicted. For a list of
        available output names, see :data:`BeatRegression.beat_.variables`

    Returns:
      list of arrays: One array per output"""
    if len(args) == 0:
      args = ("x",)
    return self.beat_.compute(np.squeeze(t), output=args)


BeatLossFunction = Callable[[np.ndarray, np.ndarray], np.ndarray]


class DualBeatRegression(BeatRegression):
  """Regressor for fitting to a beat pattern, using information
  from both the amplitude modulation and the frequency modulation

  Args:
    fs (float): Sampling frequency for autocorrelogram computation
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
      :data:`f(t, a, f, res_fn, model) -> a0, a1, f0, f1, d0, d1, p0, p1`.
      It should return initial parameters for nonlinear least squares using
      input data :data:`t`, :data:`a`, and :data:`f`, the residual function
      :data:`res_fn`, and the :class:`DualBeatRegression` instance
      :data:`model`. If :data:`None`, use default
    bounds (callable): Callable for computing beat model coefficient
      boundaries. Signature should be
      :data:`bounds(t, a, f, p, model) -> (bounds_min, bounds_max)`.
      It should return lower and upper boundaries for all eight parameters
      using input data :data:`t`, :data:`a`, and :data:`f`, initial parameter
      estimates :data:`p`, and the :class:`DualBeatRegression` instance
      :data:`model`. If :data:`None`, use default
    freq_loss (callable): Loss function for frequencies. Signature should be
      :data:`freq_loss(f_true, f_est) -> f_loss`. If :data:`None`, use default
    amp_loss (callable): Loss function for amplitudes. Signature should be
      :data:`amp_loss(a_true, a_est) -> a_loss`. If :data:`None`, use default
    freq_w (float): Weighting coefficient for the frequency loss function
    freq_clip (float): Maximum estimatable frequency deviation (as a multiple
      of the iterquartile range)
    disambiguate (bool): If :data:`True` (default), then disambiguate
      association between frequencies and amplitudes"""

  def __init__(
      self,
      fs: float = 100,
      lpf: float = 50,
      linear_regressor=None,
      linear_regressor_k: str = "coef_",
      linear_regressor_q: str = "intercept_",
      params_init: Optional[BeatParamsInit] = None,
      bounds: Optional[BeatBoundsFunc] = None,
      freq_loss: Optional[BeatLossFunction] = None,
      amp_loss: Optional[BeatLossFunction] = None,
      freq_w: float = 1 / 3800,
      freq_clip: Optional[float] = 2.5,
      disambiguate: bool = True,
  ):
    super().__init__(
        fs=fs,
        lpf=lpf,
        linear_regressor=linear_regressor,
        linear_regressor_k=linear_regressor_k,
        linear_regressor_q=linear_regressor_q,
        params_init=params_init,
        bounds=bounds,
    )
    self.freq_loss = freq_loss
    self.amp_loss = amp_loss
    self.freq_w = freq_w
    self.freq_clip = freq_clip
    self.disambiguate = disambiguate

  @utils.learn.default_property
  def freq_loss(self):
    return self._default_freq_loss

  @utils.learn.default_property
  def amp_loss(self):
    return self._default_amp_loss

  @staticmethod
  def _default_freq_loss(f_true: np.ndarray, f_est: np.ndarray) -> np.ndarray:
    """Default frequency loss function (Mel difference)

    Args:
      f_true (array): True frequency values
      f_est (array): Estimated frequency values

    Returns:
      array: Frequency differences on the Mel scale"""
    return np.subtract(*list(map(psycho.hz2mel, (f_true, f_est))))

  @staticmethod
  def _default_amp_loss(a_true: np.ndarray, a_est: np.ndarray) -> np.ndarray:
    """Default amplitude loss function (difference)

    Args:
      a_true (array): True amplitude values
      a_est (array): Estimated amplitude values

    Returns:
      array: Amplitude differences"""
    return np.subtract(a_true, a_est)

  def _residual_fn(self, t: np.ndarray, a: np.ndarray,
                   f: np.ndarray) -> BeatResidualFunc:
    """Residual function for the provided data

    Args:
      t (array): Time. Must be a column vector (shape like (-1, 1))
      a (array): Amplitude at time :data:`t`
      f (array): Frequency at time :data:`t`

    Returns:
      callable: Residual function"""
    # Time axis for model integration
    a_lin = utils.dsp.db2a(a)
    f_abs = np.abs(f)

    def _residual_fn_(beat_args: BeatModelParams) -> np.ndarray:
      a_est, f_est = beatsdrop.ModalBeat(*beat_args).compute(t, ("am", "fm"))
      a_loss = self.amp_loss(a_lin, a_est)
      np.true_divide(f_est, 2 * np.pi, out=f_est)
      np.abs(f_est, out=f_est)
      if self.freq_clip is not None:
        q1, q2, q3 = np.quantile(f_est, (0.25, 0.5, 0.75))
        max_df = self.freq_clip * (q3 - q1) / 2
        np.clip(f_est, q2 - max_df, q2 + max_df, out=f_est)
      f_loss = self.freq_loss(f_abs, f_est)
      np.multiply(self.freq_w, f_loss, out=f_loss)
      return np.reshape([a_loss, f_loss], newshape=(-1,))

    return _residual_fn_

  def fit(self, t: np.ndarray, a: np.ndarray, f: np.ndarray, **kwargs):
    """Fit beat pattern

    Args:
      t (array): Time. Must be a column vector (shape like (-1, 1))
      a (array): Amplitude at time :data:`t` (in dB)
      f (array): Frequency at time :data:`t`
      **kwargs: Keyword arguments for :func:`scipy.optimize.least_squares`

    Returns:
      DualBeatRegression: self"""
    super().fit(t, a, f, **kwargs)
    if self.disambiguate:
      hypoteses = (
          self.params_,
          (
              *self.params_[:2],  #     a0 a1
              *self.params_[3:1:-1],  # f1 f0
              *self.params_[4:6],  #    d0 d1
              *self.params_[7:5:-1],  # p1 p0
          ),
      )

      def _ssr(p: BeatModelParams) -> float:
        """Sum of squared residuals
        for current residual function

        Args:
          p: Beat model parameters

        Returns:
          float: The SSR"""
        res = self.res_fn_(p)
        np.square(res, out=res)
        return np.sum(res)

      self.params_ = min(hypoteses, key=_ssr)
      self.beat_ = beatsdrop.ModalBeat(*self.params_)
    return self
