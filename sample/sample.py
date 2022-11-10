"""Module for using the entire SAMPLE method pipeline"""
import functools
import itertools
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from sklearn import base

import sample.utils
import sample.utils.dsp
import sample.utils.learn
from sample import hinge
from sample.sms import mm

utils = sample.utils


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


def _decorate_sample(func):

  @utils.deprecated_argument("sinusoidal_model", "sinusoidal", prefix=True)
  @functools.wraps(func)
  def func_(*args, **kwargs):
    return func(*args, **kwargs)

  return func_


class SAMPLE(base.RegressorMixin, base.BaseEstimator):
  """SAMPLE (Spectral Analysis for Modal Parameter Linear Estimate) model

  Args:
    sinusoidal (SinusoidalModel): Sinusoidal analysis model.
      Default is an instance of :class:`ModalModel`
    regressor: Modal parameters regression model.
      Default is an instance of :class:`HingeRegression`
    regressor_k (str): Attribute name for the estimated slope
      coefficient of :data:`regressor`
    regressor_q (str): Attribute name for the estimated intercept
      coefficient of :data:`regressor`
    freq_reduce (callable): Callable function for reducing the frequency track
      into a single frequency. Defaults to :func:`numpy.mean`
    max_n_modes (int): Number of modes to use in resynthesis. If :data:`None`
      (default), then synthesise all modes
    **kwargs: Additional parameters for sub-models. See
      :class:`sample.sms.mm.ModalTracker`,
      :class:`sample.sms.mm.ModalModel`,
      :class:`sample.hinge.HingeRegression`, and
      :class:`sample.utils.learn.OptionalStorage`

  Attributes:
    param_matrix_ (array): 3-by-N matrix of modal parameters"""

  @_decorate_sample
  def __init__(
      self,
      sinusoidal=None,
      regressor=None,
      regressor_k: str = "k_",
      regressor_q: str = "q_",
      freq_reduce: Callable[[np.ndarray], float] = np.mean,
      max_n_modes: Optional[int] = None,
      **kwargs,
  ):
    self.sinusoidal = sinusoidal
    self.regressor = regressor
    self.regressor_k = regressor_k
    self.regressor_q = regressor_q
    self.freq_reduce = freq_reduce
    self.max_n_modes = max_n_modes
    self.set_params(**kwargs)

  @_decorate_sample
  def set_params(self, **kwargs):
    return super().set_params(**kwargs)

  @utils.learn.default_property
  def sinusoidal(self):
    """Sinusoidal analysis model"""
    return mm.ModalModel()

  @utils.learn.default_property
  def regressor(self):
    """Modal parameters regression model"""
    return hinge.HingeRegression()

  def _preprocess_track(self, i: int, x: np.ndarray,
                        t: dict) -> Tuple[int, np.ndarray, dict]:
    """Compute time axis and nan-filter track"""
    notnans = np.logical_not(np.isnan(t["mag"]))
    time_axis = (t["start_frame"] +
                 np.arange(t["mag"].size)[notnans]) / self.sinusoidal.frame_rate
    t_filtered = {
        k: v if k == "start_frame" else v[notnans] for k, v in t.items()
    }
    if getattr(self.sinusoidal.tracker, "reverse", False):
      # Reverse time axis
      time_axis = np.size(x) / self.sinusoidal.fs - time_axis
    # Compensate for spectral halving
    t_filtered["mag"] = t_filtered["mag"] + utils.dsp.DOUBLE_DB
    return i, time_axis, t_filtered

  _D2K_CONST: float = -40 * np.log10(np.e)

  def _fit_track(
      self,
      i: int,  # pylint: disable=W0613
      t: np.ndarray,
      track: dict) -> Tuple[Tuple[float, float, float]]:
    """Fit parameters for one track.

    Args:
      i (int): Track index
      tim (array): Time axis
      track (dict): Track

    Returns:
      ((float, float, float),): Frequency, decay, and amplitude"""
    # Hinge regression
    r = base.clone(self.regressor).fit(np.reshape(t, (-1, 1)), track["mag"])

    f = self.freq_reduce(track["freq"])
    d = self._D2K_CONST / getattr(r, self.regressor_k)
    a = utils.dsp.db2a(getattr(r, self.regressor_q))
    return ((f, d, a),)

  def fit(self, x: np.ndarray, y=None, **kwargs):
    """Analyze audio data

    Args:
      x (array): audio input
      y (ignored): exists for compatibility
      kwargs: Any parameter, overrides initialization

    Returns:
      SAMPLE: self"""
    self.set_params(**kwargs)
    tracks = self.sinusoidal.fit(x, y).tracks_
    tracks = itertools.starmap(
        self._preprocess_track, map(lambda t: (t[0], x, t[1]),
                                    enumerate(tracks)))
    params = itertools.chain.from_iterable(
        itertools.starmap(self._fit_track, tracks))
    self.param_matrix_ = np.array(list(params)).T
    if self.param_matrix_.size == 0:
      self.param_matrix_ = np.empty((3, 0))
    self.param_matrix_ = self.param_matrix_[:, self._valid_params_]
    return self

  @property
  def _valid_params_(self) -> Sequence[bool]:
    """Boolean array that indicates parameter validity"""
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
  osc = (utils.dsp.expi if analytical else np.cos)(osc)
  dec = col(x) @ (-2 / row(decays))
  np.exp(dec, out=dec)
  amp = col(amps)
  return np.squeeze((dec * osc) @ amp)
