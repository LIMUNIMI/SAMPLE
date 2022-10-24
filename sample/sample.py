"""Module for using the entire SAMPLE method pipeline"""
import copy
import functools
from typing import Callable, Dict, List, Optional, Sequence, Union

import numpy as np
from sklearn import base

from sample import utils
from sample import hinge
from sample.sms import mm
from sample.utils import dsp as dsp_utils


@utils.numpy_out(dtype=float)
def modal_energy(a: np.ndarray,
                 d: np.ndarray,
                 out: Optional[np.ndarray] = None) -> np.ndarray:
  """Compute the integrals of the modal amplitude envelopes

  Args:
    a (array): Amplitudes
    d (array): Decays
    out (array): Optional. Array to use for storing results

  Returns:
    array: Energies"""
  # d * a^2 / 4
  np.square(a, out=out)
  np.multiply(out, d, out=out)
  np.true_divide(out, 4, out=out)
  return out


class SAMPLE(base.RegressorMixin, base.BaseEstimator):
  """SAMPLE (Spectral Analysis for Modal Parameter Linear Estimate) model

  Args:
    sinusoidal_model: Sinusoidal model. Default is an instance of
      :class:`sample.sms.mm.ModalModel`
    regressor: Regressor. Default is an instance of
      :class:`sample.hinge.HingeRegression`
    regressor_k (str): Attribute name for the estimated slope
      coefficient of :data:`regressor`
    regressor_q (str): Attribute name for the estimated intercept
      coefficient of :data:`regressor`
    freq_reduce (callable): Callable function for reducing the frequency track
      into a single frequency. Defaults to :func:`numpy.mean`
    max_n_modes (int): Number of modes to use in resynthesis. If :data:`None`
      (default), then synthesise all modes
    **kwargs: Keyword arguments, will be set as parameters of submodels. For a
      complete list of all parameter names and default values, please, run
      :data:`SAMPLE().get_params()`. For an explanation of the parameters,
      please, refer to the documentation of the submodels"""

  def __init__(
      self,
      sinusoidal_model=mm.ModalModel(),
      regressor=hinge.HingeRegression(),
      regressor_k: str = "k_",
      regressor_q: str = "q_",
      freq_reduce: Callable[[np.ndarray], float] = np.mean,
      max_n_modes: Optional[int] = None,
      **kwargs,
  ):
    self.sinusoidal_model = sinusoidal_model
    self.regressor = regressor
    self.regressor_k = regressor_k
    self.regressor_q = regressor_q
    self.freq_reduce = freq_reduce
    self.max_n_modes = max_n_modes
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
    self.param_matrix_ = np.empty((3, len(tracks)))
    for i, t in enumerate(tracks):
      notnans = np.logical_not(np.isnan(t["mag"]))
      self.param_matrix_[0, i] = self.freq_reduce(t["freq"][notnans])
      x_ = (t["start_frame"] + np.arange(t["mag"].size)[notnans]) * \
           self.sinusoidal_model.h / self.sinusoidal_model.fs
      y_ = t["mag"][notnans]
      if getattr(self.sinusoidal_model, "reverse", False):
        x_ = np.size(x) / self.sinusoidal_model.fs - x_
      self.regressor.fit(np.reshape(x_, (-1, 1)), y_)
      self.param_matrix_[1, i] = \
        -40 * np.log10(np.e) / getattr(self.regressor, self.regressor_k)
      self.param_matrix_[2, i] = \
        2 * dsp_utils.db2a(getattr(self.regressor, self.regressor_q))
    self.param_matrix_ = self.param_matrix_[:, self._valid_params_]
    return self

  @property
  def _valid_params_(self) -> Sequence[bool]:
    return np.isfinite(self.param_matrix_).all(axis=0)

  @property
  def freqs_(self) -> np.ndarray:
    """Learned modal frequencies"""
    return self.param_matrix_[0, :]

  @freqs_.setter
  def freqs_(self, f: np.ndarray):
    self.param_matrix_[0, :] = f

  @property
  def decays_(self) -> np.ndarray:
    """Learned modal decays"""
    return self.param_matrix_[1, :]

  @decays_.setter
  def decays_(self, d: np.ndarray):
    self.param_matrix_[1, :] = d

  @property
  def amps_(self) -> np.ndarray:
    """Learned modal amplitudes"""
    return self.param_matrix_[2, :]

  @amps_.setter
  def amps_(self, a: np.ndarray):
    self.param_matrix_[2, :] = a

  @property
  def energies_(self) -> np.ndarray:
    """Learned modal energies"""
    return modal_energy(self.amps_, self.decays_)

  def mode_argsort_(self,
                    order: str = "energies",
                    reverse: bool = True) -> List[int]:
    """Get the indices for sorting modes

    Args:
      order (str): Feature to use for ordering modes. Default is
        :data:`"energies"`, so that reducing active modes keeps the
        modes with most energy. Other options are :data:`"freqs"`,
        :data:`"amps"` and :data:`"decays"`
      reverse (bool): Whether the order should be reversed (decreasing).
        Defaults to :data:`True`

    Returns:
      array: Array of indices for sorting"""
    asort = np.argsort(getattr(self, f"{order}_")).tolist()
    if reverse:
      asort = list(reversed(asort))
    return asort

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
    m_ord = self.mode_argsort_(order=order, reverse=reverse)
    active_modes = n_modes if self.max_n_modes is None else min(
        self.max_n_modes, n_modes)
    return {
        "nModes": n_modes,
        "nPickups": 1,
        "activeModes": active_modes,
        "fragmentSize": 1.0,
        "freqs": self.freqs_[m_ord].tolist(),
        "decays": self.decays_[m_ord].tolist(),
        "weights": np.full(n_modes, 1 / n_modes).tolist(),
        "gains": [self.amps_[m_ord].tolist()],
    }

  def predict(self,
              x: np.ndarray,
              n_modes: Optional[int] = None,
              order: str = "energies",
              reverse: bool = True,
              **kwargs) -> np.ndarray:
    """Resynthesize audio

    Args:
      x (array): Time axis
      n_modes (int): Number of modes to synthesize. If :data:`None`,
        then use the :data:`max_n_modes` parameter
      order (str): Feature to use for ordering modes. Default is
        :data:`"energies"`, so that reducing active modes keeps the
        modes with most energy. Other options are :data:`"freqs"`,
        :data:`"amps"` and :data:`"decays"`
      reverse (bool): Whether the order should be reversed (decreasing).
        Defaults to :data:`True`
      **kwargs: Keyword arguments for :func:`additive_synth`

    Returns:
      array: Array of audio samples"""
    if n_modes is None:
      if self.max_n_modes is None:
        n_modes = self.freqs_.size
      else:
        n_modes = self.max_n_modes
    m_ord = self.mode_argsort_(order=order, reverse=reverse)[:n_modes]
    return additive_synth(x, self.freqs_[m_ord], self.decays_[m_ord],
                          self.amps_[m_ord], **kwargs)


def _random_phases(n: int,
                   seed: Optional[int] = None,
                   rng: Optional[np.random.Generator] = None) -> np.ndarray:
  """Randomize a number of phase values between 0 and 2*pi

  Args:
    n (int): Number of phase values to sample
    rng (np.random.Generator): Random number generator
    seed (int): Seed for random number generator

  Returns:
    array: Random phase values"""
  if rng is None:
    rng = np.random.default_rng(seed=seed)
  return rng.uniform(0, 2 * np.pi, n)


_phases_funcs: Dict[str, Callable[[int], np.ndarray]] = {
    "random": _random_phases,
}


def additive_synth(x,
                   freqs: Sequence[float],
                   decays: Sequence[float],
                   amps: Sequence[float],
                   phases: Optional[Union[Sequence[float], str]] = None,
                   analytical: bool = False,
                   **kwargs) -> np.array:
  """Additively synthesize audio

    Args:
      x (array): Time axis
      freqs (array): Modal frequencies
      decays (array): Modal decays
      amps (array): Modal amplitudes
      phases (array): Starting phase for every mode, optional
      analytical (bool): If :data:`True`, use a complex
        exponential as an oscillator
      **kwargs: Keyword arguments for random phase generator

    Returns:
      array: Array of audio samples"""
  row = functools.partial(np.reshape, newshape=(1, -1))
  col = functools.partial(np.reshape, newshape=(-1, 1))

  osc = (2 * np.pi * col(x)) @ row(freqs)
  if phases is not None:
    if isinstance(phases, str):
      try:
        phases = _phases_funcs[phases](np.size(freqs), **kwargs)
      except KeyError as e:
        raise ValueError(
            f"Unsupported option for phases: '{phases}'. "
            f"Supported options are: {utils.comma_join_quote(_phases_funcs)}"
        ) from e
    np.add(osc, row(phases), out=osc)
  osc = (dsp_utils.expi if analytical else np.cos)(osc)
  dec = col(x) @ (-2 / row(decays))
  np.exp(dec, out=dec)
  amp = col(amps)
  return np.squeeze((dec * osc) @ amp)
