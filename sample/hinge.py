"""Hinge functions and regression models"""
from typing import Callable, Optional, Tuple

import numpy as np
from scipy import optimize
from sklearn import base, linear_model

import sample.utils
import sample.utils.learn

utils = sample.utils


@utils.numpy_out
def hinge_function(x: np.ndarray,
                   a: float,
                   k: float,
                   q: float,
                   out: Optional[np.ndarray] = None) -> np.ndarray:
  r"""Hinge function

  Args:
    x (array): Independent variable
    a (float): Knee point
    k (float): Slope
    q (float): Intercept
    out (array): Optional. Array to use for storing results

  Returns:
    array: :math:`h(x) = k \cdot min(x, a) + q`"""
  np.minimum(x, a, out=out)
  np.multiply(k, out, out=out)
  np.add(q, out, out=out)
  return out


HingeCoeffs = Tuple[float, float, float]
HingeCoeffsInit = Callable[[np.ndarray, np.ndarray, float, float], HingeCoeffs]
HingeBoundsFunc = Callable[[np.ndarray, np.ndarray, float, float],
                           Tuple[HingeCoeffs, HingeCoeffs]]


class HingeRegression(base.RegressorMixin, base.BaseEstimator):
  r"""Regressor for fitting to a hinge function

  :math:`h(x) = k \cdot min(x, a) + q`

  Args:
    linear_regressor (sklearn.base.BaseEstimator): Linear regression model
      instance. Must be sklearn-compatible
    linear_regressor_k (str): Attribute name for the estimated slope
      coefficient of the linear regression
    linear_regressor_q(str): Attribute name for the estimated intercept
      coefficient of the linear regression
    method (str): Nonlinear least squares method
      See :func:`scipy.optimize.least_squares` for options and detailed
      explanation. Defaults to "dogbox"
    coeffs_init (callable): Initializer for hinge function coefficients.
      Signature should be :data:`coeffs_init(x, y, k, q) -> a, k, q`. It
      should return initial parameters for the nonlinear least squares using
      input data :data:`x` and :data:`y`, and linearly estimated
      coefficients :data:`k` and :data:`q`. If None, use default
    bounds (callable): Callable for computing hinge function coefficient
      boundaries. Signature should be :data:`bounds(x, y, k, q) ->
      ((a_min, k_min, q_min), (a_max, k_max, q_max))`.
      It should return lower and upper boundaries for all three parameters
      using input data :data:`x` and :data:`y`, and linearly estimated
      coefficients :data:`k` and :data:`q`. If None, use default

  Attributes:
    coeffs_ (array): Learned parameters (a, k, q). They are also accessible
      via their individual properties
    result_ (OptimizeResult): Optimization result of the nonlinear least
      squares procedure"""

  def __init__(
      self,
      linear_regressor=None,
      linear_regressor_k: str = "coef_",
      linear_regressor_q: str = "intercept_",
      method: str = "dogbox",
      coeffs_init: HingeCoeffsInit = None,
      bounds: HingeBoundsFunc = None,
  ):
    self.linear_regressor = linear_regressor
    self.linear_regressor_k = linear_regressor_k
    self.linear_regressor_q = linear_regressor_q
    self.method = method
    self.coeffs_init = coeffs_init
    self.bounds = bounds

  @utils.learn.default_property
  def linear_regressor(self):
    """Linear regression model"""
    return linear_model.LinearRegression()

  @utils.learn.default_property
  def coeffs_init(self):
    """Initializer for hinge function coefficients"""
    return self._default_coeffs_init

  @utils.learn.default_property
  def bounds(self):
    """Callable for computing hinge function coefficient boundaries"""
    return self._default_bounds

  @staticmethod
  def _default_coeffs_init(
      x: np.ndarray,
      y: np.ndarray,  # pylint: disable=W0613
      k: float,
      q: float) -> HingeCoeffs:
    """Default coefficient initializer

    Args:
      x (array): Input independent variable array
      y (array): Input dependent variable array
      k (float): Linearly estimated slope
      q (float): Linearly estimated intercept

    Returns:
      (float, float, float): Starting values for nonlinear least squares
        for a, k and q"""
    a = (np.min(x) + np.max(x)) / 2
    return a, k, q

  @staticmethod
  def _default_bounds(x: np.ndarray, y: np.ndarray, k: float,
                      q: float) -> Tuple[HingeCoeffs, HingeCoeffs]:
    """Default boundaries

    Args:
      x (array): Input independent variable array
      y (array): Input dependent variable array
      k (float): Linearly estimated slope
      q (float): Linearly estimated intercept

    Returns:
      ((float, float, float), (float, float, float)): Minimum and maximum bounds
        for a, k and q"""
    # Knee point bounds
    if len(x) <= 1:
      raise ValueError(f"Got a track of length={len(x)}. "
                       "Consider increasing the minimum sine length")
    else:
      a_min = np.min(x)
      a_max = np.max(x)
    if a_min == a_max:
      a_min -= 1
      a_max += 1

    # Intercept bounds
    dq = np.max(np.abs(y - q)) or 1
    q_min = q - dq
    q_max = q + dq

    # Slope bounds
    if k == 0:
      k_min = -1
      k_max = 1
    else:
      k_min = 8 * k if k < 0 else k
      k_max = k if k < 0 else 8 * k

    return ((a_min, k_min, q_min), (a_max, k_max, q_max))

  def predict(self, x: np.ndarray):
    """Evaluate learned hinge function

    Args:
      x (array): Input independent variable array

    Returns:
      array: :data:`h(x)`"""
    return hinge_function(x, *self.coeffs_)

  @staticmethod
  def _residual(x: np.ndarray, y: np.ndarray) -> Callable[..., np.ndarray]:
    """Function for residual computation

    Args:
      x (array): Input independent variable array
      y (array): Input dependent variable array

    Returns:
      Residuals as function of the parameter array"""
    x_ = np.squeeze(x)
    y_ = np.squeeze(y)

    def residual(p, *args, **kwargs):  # pylint: disable=W0613
      return hinge_function(x_, *p) - y_

    return residual

  @property
  def a_(self) -> float:
    """Learned knee point"""
    return self.coeffs_[0]

  @property
  def k_(self) -> float:
    """Learned slope"""
    return self.coeffs_[1]

  @property
  def q_(self) -> float:
    """Learned intercept"""
    return self.coeffs_[2]

  def fit(self, x: np.ndarray, y: np.ndarray, **kwargs):
    """Fit hinge function

    Args:
      x (array): Independent variable. Must be a column vector
      (shape like (-1, 1))
      y (array): Dependent variable
      **kwargs: Keyword arguments for :func:`scipy.optimize.least_squares`

    Returns:
      HingeRegression: self"""
    self.linear_regressor.fit(x, y)
    lin_k = np.squeeze(getattr(self.linear_regressor, self.linear_regressor_k))
    lin_q = np.squeeze(getattr(self.linear_regressor, self.linear_regressor_q))
    self.coeffs_ = self.coeffs_init(x, y, lin_k, lin_q)

    self.result_ = optimize.least_squares(fun=self._residual(x, y),
                                          x0=self.coeffs_,
                                          bounds=self.bounds(
                                              x, y, lin_k, lin_q),
                                          method=self.method,
                                          **kwargs)
    self.coeffs_ = self.result_.x

    return self
