"""Module for using the entire SAMPLE method pipeline"""
from sklearn import base
from sample import abc
from sample.sms import mm
from sample.regression import HingeRegression
import numpy as np
import functools
import copy


class SAMPLE(base.RegressorMixin, base.BaseEstimator):
  """SAMPLE (Spectral Analysis for Modal Parameter Linear Estimate) model

  Args:
    sinusoidal_model: Sinusoidal model. Default is an instance of
      :class:`sample.sms.mm.ModalModel`
    regressor: Regressor. Default is an instance of
      :class:`sample.regression.HingeRegression`
    regressor_k (str): Attribute name for the estimated slope
      coefficient of :data:`regressor`
    regressor_q (str): Attribute name for the estimated intercept
      coefficient of :data:`regressor`
    **kwargs: Keyword arguments, will be set as parameters of submodels. For a
      complete list of all parameter names and default values, please, run
      :data:`SAMPLE().get_params()`. For an explanation of the parameters,
      please, refer to the documentation of the submodels"""
  def __init__(
    self,
    sinusoidal_model: abc.AbstractSinusoidalModel = mm.ModalModel(),
    regressor: abc.AbstractLinearRegressor = HingeRegression(),
    regressor_k: str = "k_",
    regressor_q: str = "q_",
    **kwargs,
  ):
    self.sinusoidal_model = sinusoidal_model
    self.regressor = regressor
    self.regressor_k = regressor_k
    self.regressor_q = regressor_q
    self.set_params(**kwargs)

  @property
  def sinusoidal_model(self):
    return self._sinusoidal_model

  @sinusoidal_model.setter
  def sinusoidal_model(self, model):
    self._sinusoidal_model = copy.deepcopy(model)

  @property
  def regressor(self):
    return self._regressor

  @regressor.setter
  def regressor(self, model):
    self._regressor = copy.deepcopy(model)

  def fit(self, x: np.ndarray, y=None, **kwargs):
    """Analyze audio data

    Args:
      x (array): audio input
      y (ignored): exists for compatibility
      kwargs: Any parameter, overrides initialization

    Returns:
      SAMPLE: self"""
    self.set_params(**kwargs)
    tracks = self.sinusoidal_model.fit(x, y).tracks_
    self.param_matrix_ = np.zeros((3, len(tracks)))
    for i, t in enumerate(tracks):
      notnans = np.logical_not(np.isnan(t["mag"]))
      self.param_matrix_[0, i] = np.mean(t["freq"][notnans])
      x_ = (t["start_frame"] + np.arange(t["mag"].size)[notnans]) * \
           self.sinusoidal_model.h / self.sinusoidal_model.fs
      y_ = t["mag"][notnans]
      self.regressor.fit(np.reshape(x_, (-1, 1)), y_)
      self.param_matrix_[1, i] = \
        -40 * np.log10(np.e) / getattr(self.regressor, self.regressor_k)
      self.param_matrix_[2, i] = \
        2 * 10**(getattr(self.regressor, self.regressor_q) / 20)
    return self

  @property
  def freqs_(self) -> np.ndarray:
    """Learned modal frequencies"""
    return self.param_matrix_[0, :]

  @property
  def decays_(self) -> np.ndarray:
    """Learned modal decays"""
    return self.param_matrix_[1, :]

  @property
  def amps_(self) -> np.ndarray:
    """Learned modal amplitudes"""
    return self.param_matrix_[2, :]

  @property
  def energies_(self) -> np.ndarray:
    """Learned modal energies"""
    return 4 * self.amps_**2 / self.decays_

  def sdt_params_(self, order: str = "energies", reverse: bool = True) -> dict:
    """SDT parameters as a JSON serializable dictionary

    Args:
      order (str): Feature to use for ordering modes. Default is
        :data:`"energies"`, so that reducing active modes in SDT keeps the
        modes with most energy. Other options are :data:`"freqs"`,
        :data:`"amps"` and :data:`"decays"`
      reverse (bool): Whether the order should be reversed (decreasing).
        Defaults to :data:`True`

    Returns:
      dict: SDT parameters"""
    n_modes = self.freqs_.size
    m_ord = np.argsort(getattr(self, "{}_".format(order))).tolist()
    if reverse:
      m_ord = list(reversed(m_ord))
    return {
      "nModes": n_modes,
      "nPickups": 1,
      "activeModes": n_modes,
      "fragmentSize": 1.0,
      "freqs": self.freqs_[m_ord].tolist(),
      "decays": self.decays_[m_ord].tolist(),
      "weights": np.full(n_modes, 1 / n_modes).tolist(),
      "gains": [self.amps_[m_ord].tolist()],
    }

  def predict(self, x: np.ndarray) -> np.ndarray:
    """Resynthesize audio

    Args:
      x (array): Time axis

    Returns:
      array: Array of audio samples"""
    row = functools.partial(np.reshape, newshape=(1, -1))
    col = functools.partial(np.reshape, newshape=(-1, 1))

    osc = np.sin(2 * np.pi * col(x) @ row(self.freqs_))
    dec = np.exp(col(x) @ row(-2 / self.decays_))
    amp = col(self.amps_)
    return np.squeeze((dec * osc) @ amp)
