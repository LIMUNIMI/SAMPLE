"""Regression models"""
from sklearn import linear_model, base
from scipy import optimize
import numpy as np
from typing import Tuple, Optional, Callable


def hinge_function(x: np.ndarray, a: float, k: float, q: float) -> np.ndarray:
  """Hinge function

  Args:
    x (array): Independent variable
    a (float): Knee point
    k (float): Slope
    q (float): Intercept

  Returns:
    array: :math:`h(x) = k * min(x, a) + q`"""
  return k * np.minimum(x, a) + q


coeff_init_type: type = Callable[
  [np.ndarray, np.ndarray, float, float],
  Tuple[float, float, float]
]


bounds_fun_type: type = Callable[
  [np.ndarray, np.ndarray, float, float],
  Tuple[Tuple[float, float, float], Tuple[float, float, float]]
]


class HingeRegression(base.RegressorMixin, base.BaseEstimator):
  """Regressor for fitting to a hinge function

  :math:`h(x) = k * min(x, a) + q`

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
    linear_regressor=linear_model.LinearRegression(),
    linear_regressor_k: str = "coef_",
    linear_regressor_q: str = "intercept_",
    method: str = "dogbox",
    coeffs_init: Optional[coeff_init_type] = None,
    bounds: Optional[bounds_fun_type] = None,
  ):
    self.linear_regressor = linear_regressor
    self.linear_regressor_k = linear_regressor_k
    self.linear_regressor_q = linear_regressor_q
    self.method = method
    self.coeffs_init = coeffs_init
    self.bounds = bounds

  @staticmethod
  def _default_coeffs_init(
    x: np.ndarray,
    y: np.ndarray,  # pylint: disable=W0613
    k: float,
    q: float
  ) -> Tuple[float, float, float]:
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

  @property
  def _coeffs_init(self) -> coeff_init_type:
    """Coefficient initializer

    Returns:
      User-provided function if given, otherwise
        :meth:`_default_coeffs_init`"""
    if self.coeffs_init is None:
      return self._default_coeffs_init
    return self.coeffs_init

  def _default_bounds(
    self,
    x: np.ndarray,
    y: np.ndarray,
    k: float,
    q: float  # pylint: disable=W0613
  ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """Default boundaries

    Args:
      x (array): Input independent variable array
      y (array): Input dependent variable array
      k (float): Linearly estimated slope
      q (float): Linearly estimated intercept

    Returns:
      ((float, float, float), (float, float, float)): Minimum and maximum bounds
        for a, k and q"""
    dq = np.max(np.abs(y - q))
    return (
      (np.min(x), 4 * k if k < 0 else k, q - dq),
      (np.max(x), k if k < 0 else 4 * k, q + dq)
    )

  @property
  def _bounds(self) -> bounds_fun_type:
    """Boundary function

    Returns:
      User-provided function if given, otherwise
        :meth:`_default_bounds`"""
    if self.bounds is None:
      return self._default_bounds
    return self.bounds

  def predict(self, x: np.ndarray):
    """Evaluate learned hinge function

    Args:
      x (array): Input independent variable array

    Returns:
      array: :data:`h(x)`"""
    return hinge_function(x, *self.coeffs_)

  @staticmethod
  def _residual(
    x: np.ndarray,
    y: np.ndarray
  ) -> Callable[..., np.ndarray]:
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
    self.coeffs_ = self._coeffs_init(  # pylint: disable=E1102 (false positive)
      x, y,
      np.squeeze(getattr(self.linear_regressor, self.linear_regressor_k)),
      np.squeeze(getattr(self.linear_regressor, self.linear_regressor_q)),
    )

    self.result_ = optimize.least_squares(
      fun=self._residual(x, y),
      x0=self.coeffs_,
      bounds=self._bounds(x, y, *self.coeffs_[1:]),  # pylint: disable=E1102 (false positive)
      method=self.method,
      **kwargs
    )
    self.coeffs_ = self.result_.x

    return self
