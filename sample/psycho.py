"""Classes and functions related to psychoacoustic models"""
import functools
from typing import (Any, Callable, Dict, Iterable, Optional, Sequence, Tuple,
                    Union)

import numpy as np
import sklearn.exceptions
from scipy import signal

from sample import utils
from sample.utils import dsp as dsp_utils


@utils.function_with_variants(key="mode", default="traunmuller")
def hz2bark(f, mode: str = "traunmuller", out: Optional[np.ndarray] = None):  # pylint: disable=W0613
  """Convert Hertz to Bark

  Args:
    f: Frequency value(s) in Hertz
    mode (str): Name of the Bark definition (zwicker, traunmuller, or wang)
    out (array): Optional. Array to use for storing results

  Returns:
    Frequency value(s) in Bark"""
  pass  # pragma: no cover


@utils.function_with_variants(key="mode", default="traunmuller")
def bark2hz(b, mode: str = "traunmuller", out: Optional[np.ndarray] = None):  # pylint: disable=W0613
  """Convert Bark to Hertz

  Args:
    b: Frequency value(s) in Bark
    mode (str): Name of the Bark definition (traunmuller, or wang)
    out (array): Optional. Array to use for storing results

  Returns:
    Frequency value(s) in Hertz"""
  pass  # pragma: no cover


@utils.function_variant(hz2bark, "zwicker")
@utils.numpy_out(dtype=float)
def _hz2bark_zwicker(f, out: Optional[np.ndarray] = None):
  """Original definition of the Bark scale (Zwicker & Terhardt (1980))

  Args:
    f: Frequency value(s) in Hertz
    out (array): Optional. Array to use for storing results

  Returns:
    Frequency value(s) in Bark"""
  # 13.0 * arctan(7.6e-4 * f)
  tmp = np.empty_like(out)
  np.multiply(7.6e-4, f, out=tmp)
  np.arctan(tmp, out=tmp)
  np.multiply(13.0, tmp, out=tmp)
  # 3.5 * arctan((f / 7500.0)^2)
  np.true_divide(f, 7500.0, out=out)
  np.square(out, out=out)
  np.arctan(out, out=out)
  np.multiply(3.5, out, out=out)
  #  13.0 * arctan(7.6e-4 * f) + 3.5 * arctan((f / 7500.0)^2)
  return np.add(tmp, out, out=out)


@utils.function_variant(hz2bark, "traunmuller")
@utils.numpy_out(dtype=float)
def _hz2bark_traunmuller(f, out: Optional[np.ndarray] = None):
  """Definition of the Bark scale by Traunmuller
  (Analytical expressions for the tonotopic sensory scale, 1990)

  Args:
    f: Frequency value(s) in Hertz
    out (array): Optional. Array to use for storing results

  Returns:
    Frequency value(s) in Bark"""
  # 1960.0 + f
  tmp = np.empty_like(out)
  np.add(1960.0, f, out=tmp)
  # (26.81 * f / (1960.0 + f)) - 0.53
  np.multiply(26.81, f, out=out)
  np.true_divide(out, tmp, out=out)
  return np.subtract(out, 0.53, out=out)


@utils.function_variant(hz2bark, "wang")
@utils.numpy_out(dtype=float)
def _hz2bark_wang(f, out: Optional[np.ndarray] = None):
  """Definition of the Bark scale by Wang et al. (An objective measure for
  predicting subjective quality of speech coders, 1992)

  Args:
    f: Frequency value(s) in Hertz
    out (array): Optional. Array to use for storing results

  Returns:
    Frequency value(s) in Bark"""
  # 6.0 * arcsinh(f / 600.0)
  np.true_divide(f, 600.0, out=out)
  np.arcsinh(out, out=out)
  return np.multiply(6.0, out, out=out)


@utils.function_variant(bark2hz, "traunmuller")
@utils.numpy_out(dtype=float)
def _bark2hz_traunmuller(b, out: Optional[np.ndarray] = None):
  """Definition of the Bark scale by Traunmuller
  (Analytical expressions for the tonotopic sensory scale, 1990)

  Args:
    b: Frequency value(s) in Bark
    out (array): Optional. Array to use for storing results

  Returns:
    Frequency value(s) in Hertz"""
  # 0.53 + b
  tmp = np.empty_like(out)
  np.add(0.53, b, out=tmp)
  # (0.53 + b) / (26.28 - b) * 1960.0
  np.subtract(26.28, b, out=out)
  np.true_divide(tmp, out, out=out)
  return np.multiply(out, 1960.0, out=out)


@utils.function_variant(bark2hz, "wang")
@utils.numpy_out(dtype=float)
def _bark2hz_wang(b, out: Optional[np.ndarray] = None):
  """Definition of the Bark scale by Wang et al. (An objective measure for
  predicting subjective quality of speech coders, 1992)

  Args:
    b: Frequency value(s) in Bark
    out (array): Optional. Array to use for storing results

  Returns:
    Frequency value(s) in Hertz"""
  # 600.0 * sinh(b / 6.0)
  np.true_divide(b, 6.0, out=out)
  np.sinh(out, out=out)
  return np.multiply(600.0, out, out=out)


@utils.function_with_variants(key="mode", this="default")
@utils.numpy_out(dtype=float)
def hz2mel(f, mode: str = "default", out: Optional[np.ndarray] = None):  # pylint: disable=W0613
  """Convert Hertz to Mel

  Args:
    f: Frequency value(s) in Hertz
    mode (str): Name of the Mel definition (default, fant)
    out (array): Optional. Array to use for storing results

  Returns:
    Frequency value(s) in Mel"""
  # 2595.0 * log10(1.0 + f / 700.0)
  np.true_divide(f, 700.0, out=out)
  np.add(1.0, out, out=out)
  np.log10(out, out=out)
  return np.multiply(2595.0, out, out=out)


@utils.function_with_variants(key="mode", this="default")
@utils.numpy_out(dtype=float)
def mel2hz(m, mode: str = "default", out: Optional[np.ndarray] = None):  # pylint: disable=W0613
  """Convert Mel to Hertz

  Args:
    m: Frequency value(s) in Mel
    mode (str): Name of the Mel definition (default, fant)
    out (array): Optional. Array to use for storing results

  Returns:
    Frequency value(s) in Hertz"""
  # (10^(m / 2595.0) - 1.0) * 700.0
  np.true_divide(m, 2595.0, out=out)
  np.power(10.0, out, out=out)
  np.subtract(out, 1.0, out=out)
  return np.multiply(out, 700.0, out=out)


@utils.function_variant(hz2mel, "fant")
@utils.numpy_out(dtype=float)
def _hz2mel_fant(f, out: Optional[np.ndarray] = None):
  """Definition of the Mel scale by Fant (Analysis and synthesis
  of speech processes, 1968)

  Args:
    f: Frequency value(s) in Hertz
    out (array): Optional. Array to use for storing results

  Returns:
    Frequency value(s) in Mel"""
  # 1000.0 * log2(1.0 + f / 1000.0)
  np.true_divide(f, 1000.0, out=out)
  np.add(1.0, out, out=out)
  np.log2(out, out=out)
  return np.multiply(1000.0, out, out=out)


@utils.function_variant(mel2hz, "fant")
@utils.numpy_out(dtype=float)
def _mel2hz_fant(m, out: Optional[np.ndarray] = None):
  """Definition of the Mel scale by Fant (Analysis and synthesis
  of speech processes, 1968)

  Args:
    m: Frequency value(s) in Mel
    out (array): Optional. Array to use for storing results

  Returns:
    Frequency value(s) in Hertz"""
  # (2^(m / 1000.0) - 1.0) * 1000.0
  np.true_divide(m, 1000.0, out=out)
  np.power(2.0, out, out=out)
  np.subtract(out, 1.0, out=out)
  return np.multiply(out, 1000.0, out=out)


@utils.function_with_variants(key="degree",
                              default="quadratic",
                              this="quadratic")
@utils.numpy_out(dtype=float)
def erb(
    f: float,
    degree: str = "quadratic",  # pylint: disable=W0613
    out: Optional[np.ndarray] = None) -> float:
  """Definition of equivalent rectangular bandwidth by Moore and
  Glasberg, "Suggested formulae for calculating auditory-filter
  bandwidths and excitation patterns"

  Args:
    f (array): Frequency value(s) in Hertz
    degree (str): Name of the ERB definition (linear, quadratic)
    out (array): Optional. Array to use for storing results

  Returns:
    array: Equivalent recrangular bandwidths at the given frequencies"""
  # 0.009339 * f
  tmp = np.empty_like(out)
  np.multiply(0.009339, f, out=tmp)
  # 6.23e-6 * f^2 + 0.009339 * f + 28.52
  np.square(f, out=out)
  np.multiply(6.23e-6, out, out=out)
  np.add(out, tmp, out=out)
  return np.add(out, 28.52, out=out)


@utils.function_variant(erb, "linear")
@utils.numpy_out(dtype=float)
def _erb_linear(f: float, out: Optional[np.ndarray] = None) -> float:
  """Definition of equivalent rectangular bandwidth by Moore and
  Glasberg, "Derivation of auditory filter shapes from
  notched-noise data"

  Args:
    f (array): Frequency value(s) in Hertz
    out (array): Optional. Array to use for storing results

  Returns:
    array: Equivalent recrangular bandwidths at the given frequencies"""
  # 24.7 * (0.00437 * f + 1)
  np.multiply(0.00437, f, out=out)
  np.add(out, 1, out=out)
  return np.multiply(24.7, out, out=out)


@utils.function_with_variants(key="degree",
                              default="quadratic",
                              this="quadratic")
@utils.numpy_out(dtype=float)
def hz2cams(
    f: float,
    degree: str = "quadratic",  # pylint: disable=W0613
    out: Optional[np.ndarray] = None) -> float:
  """Quadratic definition of ERB-rate-scale

  Args:
    f (array): Frequency value(s) in Hertz
    degree (str): Name of the ERB definition (linear, quadratic)
    out (array): Optional. Array to use for storing results

  Returns:
    array: Frequency value(s) in Cams"""
  # f + 312
  tmp = np.empty_like(out)
  np.add(f, 312, out=tmp)
  # 11.17 * logn((f + 312) / (f + 14675)) + 43.0
  np.add(f, 14675, out=out)
  np.true_divide(tmp, out, out=out)
  np.log(out, out=out)
  np.multiply(11.17, out, out=out)
  return np.add(out, 43.0, out=out)


@utils.function_variant(hz2cams, "linear")
@utils.numpy_out(dtype=float)
def _hz2cams_linear(f: float, out: Optional[np.ndarray] = None) -> float:
  """Linear definition of ERB-rate-scale

  Args:
    f (array): Frequency value(s) in Hertz
    out (array): Optional. Array to use for storing results

  Returns:
    array: Frequency value(s) in Cams"""
  # 21.4 * log10(1 + 0.00437 * f)
  np.multiply(0.00437, f, out=out)
  np.add(1, out, out=out)
  np.log10(out, out=out)
  return np.multiply(21.4, out, out=out)


@utils.function_with_variants(key="degree",
                              default="quadratic",
                              this="quadratic")
@utils.numpy_out(dtype=float)
def cams2hz(
    c: float,
    degree: str = "quadratic",  # pylint: disable=W0613
    out: Optional[np.ndarray] = None) -> float:
  """Quadratic definition of ERB-rate-scale

  Args:
    c (array): Frequency value(s) in Cams
    degree (str): Name of the ERB definition (linear, quadratic)
    out (array): Optional. Array to use for storing results

  Returns:
    array: Frequency value(s) in Hz"""
  # k = e^((c - 43) / 11.17)
  k = np.empty_like(out)
  np.subtract(c, 43, out=k)
  np.true_divide(k, 11.17, out=k)
  np.exp(k, out=k)
  # f = (312 - 14675 * k) / (k - 1)
  np.multiply(14675, k, out=out)
  np.subtract(312, out, out=out)
  np.subtract(k, 1, out=k)
  return np.true_divide(out, k, out=out)


@utils.function_variant(cams2hz, "linear")
@utils.numpy_out(dtype=float)
def _cams2hz_linear(c: float, out: Optional[np.ndarray] = None) -> float:
  """Linear definition of ERB-rate-scale

  Args:
    c (array): Frequency value(s) in Cams
    out (array): Optional. Array to use for storing results

  Returns:
    array: Frequency value(s) in Hz"""
  # (10^(c / 21.4) - 1) / 0.00437
  np.true_divide(c, 21.4, out=out)
  np.power(10, out, out=out)
  np.subtract(out, 1, out=out)
  return np.true_divide(out, 0.00437, out=out)


@utils.numpy_out(dtype=float)
def a_weighting(f: float,
                db: bool = True,
                out: Optional[np.ndarray] = None) -> float:
  """A-Weighting weights for input frequencies, as of "Electroacoustics - Sound
  level meters - Part 1: Specifications" (2013)

  Args:
    f (array): Frequency values in Hertz
    db (bool): If :data:`True` (default), return the gain to apply in dB
      with reference at 1kHz (:data:`a_weighting(1000) = 0`)
    out (array): Optional. Array to use for storing results

  Returns:
    array: A-weights"""
  if db:
    return dsp_utils.a2db(a_weighting(f, db=False, out=out)) - _a_weight_1kHz_db
  f2 = np.empty_like(out)
  np.square(f, out=f2)
  # f^2 + 107.7^2
  np.add(f2, 11599.29, out=out)
  # (f^2 + 107.7^2) (f^2 + 737.9^2)
  tmp = np.empty_like(out)
  np.multiply(out, np.add(f2, 544496.41, out=tmp), out=out)
  # sqrt((f^2 + 107.7^2) (f^2 + 737.9^2))
  np.sqrt(out, out=out)
  # (f^2 + 20.6^2) sqrt((f^2 + 107.7^2) (f^2 + 737.9^2)) (f^2 + 12194^2)
  np.multiply(out, np.add(f2, 424.36, out=tmp), out=out)
  np.multiply(out, np.add(f2, 148693636, out=tmp), out=out)
  # 12194^2 f^4 / (...)
  np.multiply(148693636, np.square(f2, out=f2), out=tmp)
  np.true_divide(tmp, out, out=out)
  return out


_a_weight_1kHz_db: float = dsp_utils.a2db(a_weighting(1e3, db=False))

OptionalValueOrFunc = Optional[Union[float, Callable[["GammatoneFilter"],
                                                     float]]]


class GammatoneFilter:
  """Gammatone filter

  Args:
    f (float): Center frequency. Default is :data:`1`
    n (int): Filter order. Default is :data:`4`
    bandwidth (callable or float): Filter bandwidth in Hz. If callable, it
      must accept a single argument of type :class:`GammatoneFilter`. If
      :data:`None` (default), then the ERB (:func:`erb`) is used
    t_c (callable or float): Leading time in seconds. If callable, it
      must accept a single argument of type :class:`GammatoneFilter`. If
      :data:`None` (default), then use the group-delay (non-causal filter)
    phi (callable or float): Phase in radians at time t=0. If callable, it
      must accept a single argument of type :class:`GammatoneFilter`. If
      :data:`None` (default), then use a phase value coherent with the
      leading time
    normalize (bool): If :data:`True`, then
      normalize the IR so that :math:`|k(t)|^2 = f_s`
    a (callable or float): Scale parameter. If callable, it
      must accept a single argument of type :class:`GammatoneFilter`. If
      :data:`None` (default), then do not rescale IR"""

  def __init__(self,
               f: float = 1,
               n: int = 4,
               bandwidth: OptionalValueOrFunc = None,
               t_c: OptionalValueOrFunc = None,
               phi: OptionalValueOrFunc = None,
               normalize: bool = False,
               a: OptionalValueOrFunc = None):
    self._a = a
    self.f = f
    self.n = n
    self._bandwidth = bandwidth
    self._t_c = t_c
    self._phi = phi
    self.normalize = normalize

  @property
  def bandwidth(self) -> float:
    """Filter bandwidth in Hz"""
    if self._bandwidth is None:
      return erb(self.f)
    if callable(self._bandwidth):
      return self._bandwidth(self)
    return self._bandwidth

  @bandwidth.setter
  def bandwidth(self, v: OptionalValueOrFunc):
    self._bandwidth = v

  @property
  def raw_group_delay(self) -> float:
    """Raw group delay of the gammatone_filter in seconds
    (without accounting for the leading time)"""
    return (self.n - 1) / (2 * np.pi * self.bandwidth)

  @property
  def group_delay(self) -> float:
    """Group delay of the gammatone_filter in seconds
    (accounting for the leading time)"""
    return self.raw_group_delay - self.t_c

  @group_delay.setter
  def group_delay(self, v: float):
    self._t_c = self.raw_group_delay - v

  @property
  def t_c(self) -> float:
    """Leading time in seconds"""
    if self._t_c is None:
      return self.raw_group_delay
    if callable(self._t_c):
      return self._t_c(self)
    return self._t_c

  @t_c.setter
  def t_c(self, v: OptionalValueOrFunc):
    self._t_c = v

  @property
  def phi(self) -> float:
    """Initial phase in radians"""
    if self._phi is None:
      return -2 * np.pi * self.f * self.t_c
    if callable(self._phi):
      return self._phi(self)
    return self._phi

  @phi.setter
  def phi(self, v: OptionalValueOrFunc):
    self._phi = v

  @property
  def a(self) -> float:
    """Scale parameter"""
    if callable(self._a):
      return self._a(self)
    return self._a

  @a.setter
  def a(self, v: OptionalValueOrFunc):
    self._a = v

  def t60(self,
          steps: int = 32,
          n_starts: int = 16,
          initial_range: Optional[float] = None,
          floor: float = -60,
          warn_th: Optional[float] = 1e-3) -> float:
    """Numerically compute the t60 for the IR envelope, i.e. the time instant
    at which the IR envelope goes 60 dB below the envelope peak

    Args:
      steps (int): Dichotomic search steps for t60 computation
      n_starts (int): Number of restarts for determining the initial search
        range before raising an exception
      initial_range (float): Width of the initial search range. In case the
        t60 is not in the range, the width is doubled :data:`n_starts` times
      floor (float): Threshold for the t60 in decibel. Default is :data:`-60`
      warn_th (float): If not :data:`None`, then raise an exception if the
        amplitude at the found t60 value is not within :data:`warn_th` dB from
        the target value (:data:`floor`)

    Returns:
      float: The t60 value"""
    hi = self.group_delay
    p = dsp_utils.db2a(floor) * self.envelope(hi)
    e = None
    if initial_range is None:
      initial_range = 2 * (hi + self.t_c)

    def _foo(t):
      return self.envelope(t) - p

    for _ in range(n_starts):
      try:
        t = dsp_utils.dychotomic_zero_crossing(_foo,
                                               lo=hi + initial_range,
                                               hi=hi,
                                               steps=steps)
      except ValueError as ex:
        hi = hi + initial_range
        initial_range = initial_range * 2
        e = ex
      else:
        break
    else:
      raise ValueError(
          f"Could not find a suitable starting range in {n_starts} iterations"
      ) from e
    if warn_th is not None:
      f_t = dsp_utils.a2db(self.envelope(t) / self.envelope(self.group_delay))
      if abs(f_t - floor) > warn_th:
        raise sklearn.exceptions.ConvergenceWarning(
            f"Amplitude at t60 is not within {warn_th} dB from target "
            f"({floor} dB) after {steps} steps (a({t}) = {f_t}). "
            "Consider increasing the search steps")
    return t

  def ir_size(self, fs: float = 1, **kwargs) -> int:
    """Suggested IR size in samples, based on the t60

    Args:
      fs (float): Sample frequency
      **kwargs: Keyword arguments for :func:`t60`

    Returns:
      int: Suggested IR size"""
    w = 2 * np.pi * self.f
    # phase at t60
    phase60 = self.phi + w * self.t60(**kwargs)
    # quantize at next multiple of pi/2 => either zero-value or zero-derivative
    phase60 = np.ceil(phase60 * 2 / np.pi) * np.pi / 2
    # quantized t60
    t60 = (phase60 - self.phi) / w
    return np.floor((t60 + self.t_c) * fs).astype(int)

  @staticmethod
  @utils.numpy_out(dtype=float)
  def _envelope(t: float,
                n: int,
                t_c: float,
                b: float,
                out: Optional[float] = None) -> np.ndarray:
    """Envelope function for the IR

    Args:
      t (array): Time axis
      n (int): Filter order
      t_c (float): Leading time
      b (float): FIlter bandwidth
      out (array): Optional. Array to use for storing results

    Returns:
      array: The IR envelope function evaluated at :data:`t`"""
    tmp = np.empty_like(t)
    # t + t_c
    t_ = np.empty_like(t)
    np.add(t, t_c, out=t_)
    # u(t + t_c)
    np.greater_equal(t_, 0, out=out)
    # u(t + t_c) * (t + t_c)^(n-1)
    np.power(t_, n - 1, out=tmp)
    np.multiply(tmp, out, out=out)
    # u(t + t_c) * (t + t_c)^(n-1) * exp(-2pi * (t + t_c) * b)
    np.multiply(-2 * np.pi * b, t_, out=tmp)
    np.exp(tmp, out=tmp)
    np.multiply(tmp, out, out=out)
    return out

  def envelope(self,
               t: np.ndarray,
               out: Optional[float] = None,
               **kwargs) -> np.ndarray:
    """Envelope function for the IR

    Args:
      t (array): Time axis
      out (array): Optional. Array to use for storing results
      **kwargs: Keyword arguments for :func:`_envelope`

    Returns:
      array: The IR envelope function evaluated at :data:`t`"""
    return self._envelope(t,
                          n=self.n,
                          t_c=self.t_c,
                          b=self.bandwidth,
                          out=out,
                          **kwargs)

  def wavefun(self,
              t: float,
              analytical: bool = False,
              out: Optional[np.ndarray] = None) -> np.ndarray:
    """Filter wave function for the IR (non-scaled)

    Args:
      t (array): Time axis
      analytical (bool): If :data:`True`, use a complex exponential as
        oscillator, instead of a cosine
      out (array): Optional. Array to use for storing results

    Returns:
      array: The wave function evaluated at :data:`t`"""
    out = self.envelope(t,
                        out=out,
                        **({
                            "dtype": complex
                        } if analytical and out is None else {}))
    tmp = np.empty_like(out)
    # cos(2pi * f * t + phi)
    np.multiply(2 * np.pi * self.f, t, out=tmp)
    np.add(tmp, self.phi, out=tmp)
    if analytical:
      np.multiply(1j, tmp, out=tmp)
      np.exp(tmp, out=tmp)
    else:
      np.cos(tmp, out=tmp)
    np.multiply(out, tmp, out=out)
    return out

  def ir(self,
         t: Optional[float] = None,
         fs: Optional[float] = None,
         analytical: bool = False,
         out: Optional[np.ndarray] = None,
         **kwargs) -> np.ndarray:
    """Filter IR (scaled)

    Args:
      t (array): Time axis
      fs (float): Sample frequency
      analytical (bool): If :data:`True`, use a complex exponential as
        oscillator, instead of a cosine
      out (array): Optional. Array to use for storing results
      **kwargs: Keyword arguments for :func:`ir_size`

    Returns:
      array: The wave function evaluated at :data:`t`"""
    if t is None:
      if fs is None:
        raise ValueError(
            "Please, specify either the time axis or a sample frequency")
      t = np.arange(self.ir_size(fs=fs, **kwargs)) / fs - self.t_c
    out = self.wavefun(t, analytical=analytical, out=out)
    # Scale
    a = self.a
    if self.normalize:
      # Divide a by the square norm of ir (can be complex)
      a = (1 if a is None else a) / np.sqrt(
          np.real(np.dot(out, np.conjugate(out))))
    if a is not None:
      np.multiply(a, out, out=out)
    return out


class GammatoneFilterbank:
  """Bank of gammatone filters

  Args:
    filters (iterable of GammatoneFilter): Filters that make up the bank.
      If :data:`None` (default), then build filters using other arguments
    freqs: The center frequencies of the gammatone filters. If :data:`None`
      (default), then decide frequencies using other arguments
    n_filters (int): Number of gammatone filters. If :data:`None`
      (default), then decide number of filters using other arguments
    flim (float, float): Limits for the frequency response of the
      gammatone filters
    freq_transform: Couple of callables that implement transformations
      from and to Hertz, respectively. The center frequencies of the
      gammatone filters will be chosen linearly between :data:`flim[0]`
      and :data:`flim[1]` in the transformed space. Default is
      :func:`hz2cams`, :func:`cams2hz` for linear spacing on the ERB-rate scale
    **kwargs: Keyword arguments for :class:`GammatoneFilter`"""

  def __init__(self,
               filters: Optional[Iterable[GammatoneFilter]] = None,
               freqs: Optional[Sequence[float]] = None,
               n_filters: Optional[int] = None,
               flim: Tuple[float, float] = (20, 20000),
               freq_transform: Tuple[Callable[[float], float],
                                     Callable[[float],
                                              float]] = (hz2cams, cams2hz),
               **kwargs) -> "GammatoneFilterbank":
    if filters is None:
      if freqs is None:
        flim_t = freq_transform[0]((flim[0], flim[-1]))
        if n_filters is None:
          n_filters = np.ceil(2 * (flim_t[-1] - flim_t[0]) - 1).astype(int)
        freqs = freq_transform[1]((np.arange(n_filters, dtype=float) + 0.5) *
                                  (flim_t[-1] - flim_t[0]) / (n_filters + 1) +
                                  flim_t[0])
      filters = (GammatoneFilter(f=f, **kwargs) for f in freqs)
    self._filters = tuple(filters)

  def __iter__(self) -> Iterable[GammatoneFilter]:
    """Iterate through filters"""
    return iter(self._filters)

  def __getitem__(self, i: int) -> GammatoneFilter:
    """Get the i-th filter"""
    return self._filters[i]

  def __len__(self) -> int:
    """Number of filters"""
    return len(self._filters)

  def __getattr__(self, key: str):
    """If attribute is not found, try getting a tuple of attributes,
    one from each filter"""
    try:
      return tuple(getattr(f, key) for f in self)
    except AttributeError:
      pass
    return super().__getattr__(key)

  class PrecomputedIRBank:
    """Precomputed IR bank for a :class:`GammatoneFilterbank`

    Args:
      parent (GammatoneFilterbank): Gammatone filterbank to render
      fs (float): Sample frequency
      analytical (bool): If :data:`True`, then the IRs are complex-valued.
        Convolving the complex IRs is faster than convolving
        the real IRs and then computing the analytical signal of the
        cochleagram. The resulting cochleagram will be complex. The real
        part will be the ordinary cochleagram. The absolute value will be
        the AM envelope of the cochleagram"""

    def __init__(self,
                 parent: "GammatoneFilterbank",
                 fs: float,
                 analytical: bool = False,
                 **kwargs):
      self.freqs = parent.f
      self.analytical = analytical
      # Initially offsets are the leading times in samples
      self.offsets = np.ceil(np.multiply(parent.t_c, fs)).astype(int)

      # Time axes start accordindly to the leading times
      time_axes = ((np.arange(f.ir_size(fs=fs), dtype=float) - off) / fs
                   for off, f, in zip(self.offsets, parent))
      irs = (f.ir(t=t, analytical=analytical, **kwargs)
             for t, f in zip(time_axes, parent))
      self.irs = tuple(irs)

      # Offsets now are the delays in samples to apply to each IR
      # for them to be correctly aligned
      self.offsets = self.offsets.max() - self.offsets
      self._ir_size = max(
          ir.size + off for ir, off in zip(self.irs, self.offsets))

    def __len__(self) -> int:
      """Number of IRs"""
      return len(self.irs)

    def convolve(self,
                 x: np.ndarray,
                 method: Optional[str] = None,
                 stride: Optional[int] = None):
      """Convolve the IRs and organize the outputs in an aligned matrix

      Args:
        x (array): Input signal
        method (str): Convolution method (either :data:`"auto"`,
          :data:`"fft"`, :data:`"direct"`, or :data:`"overlap-add"`)
        stride (int): Time-step for output signal.
          Can't be used in conjunction with :data:`method`

      Returns:
        matrix: Cochleagram, will be complex if the IRs are analytical"""
      if stride is None:
        if method is None or method == "overlap-add":
          convolve = signal.oaconvolve
        else:
          convolve = functools.partial(signal.convolve, method=method)
        out = np.zeros((len(self), x.size + self._ir_size - 1),
                       dtype=np.result_type(x, *self.irs))
        for i, (ir, off) in enumerate(zip(self.irs, self.offsets)):
          y = convolve(x, ir, mode="full")
          # Delay is LTI, so we shift the result instead of zero-padding
          # the start of the IR in order to save computation time
          out[i, off:(off + y.size)] = y
      elif method is not None:
        raise ValueError(
            f"Cannot specify both stride={stride} and method='{method}'. "
            "Please leave either one as 'None'")
      elif np.iscomplexobj(x):
        raise ValueError(
            f"Strided convolution unsupported for input of dtype '{x.dtype}'. ",
            "Must be a real dtype")
      else:
        # Zero-padding for full-convolution
        x_pad = np.zeros(x.size + 2 * (self._ir_size - 1), dtype=x.dtype)
        x_pad[(self._ir_size - 1):-(self._ir_size - 1)] = x
        # Output delay (quantized to stride)
        offsets_s = self.offsets // stride
        # Input delay (quantization residual)
        offsets_x = self.offsets - offsets_s * stride
        # IR lengths
        ir_sizes = np.array([ir.size for ir in self.irs])
        # Starting index
        x_begins = self._ir_size - ir_sizes - offsets_x
        # Input lengths
        x_sizes = np.minimum(x_pad.size - x_begins, x.size + 2 * (ir_sizes - 1))
        # Output lengths
        n_wins = np.array([
            dsp_utils.n_windows(x_size, ir_size, hop=stride)
            for x_size, ir_size in zip(x_sizes, ir_sizes)
        ])
        out = np.zeros((len(self), np.max(n_wins + offsets_s)),
                       dtype=np.result_type(x, *self.irs))
        if self.analytical:
          convolve = dsp_utils.strided_convolution_complex_kernel
        else:
          convolve = dsp_utils.strided_convolution
        for i, ir in enumerate(self.irs):
          convolve(x_pad[x_begins[i]:(x_begins[i] + x_sizes[i])],
                   ir,
                   stride=stride,
                   out=out[i:(i + 1),
                           offsets_s[i]:(offsets_s[i] + n_wins[i])].T)
      return out

  def precompute(self,
                 fs: float,
                 analytical: bool = False) -> PrecomputedIRBank:
    """Precompute IRs for this filterbank

    Args:
      fs (float): Sample frequency
      analytical (bool): If :data:`True`, compute a complex IR bank

    Returns:
      PrecomputedIRBank: Precomputed IR bank"""
    return GammatoneFilterbank.PrecomputedIRBank(parent=self,
                                                 fs=fs,
                                                 analytical=analytical)

  def convolve(self,
               x: np.ndarray,
               fs: float,
               analytical: Optional[str] = None,
               method: Optional[str] = None,
               **kwargs):
    # pylint: disable=C0303
    """Filter the input with the filterbank and produce a cochleagram

    Args:
      x (array): Input signal
      fs (float): Sample frequency
      analytical (str): Compute the analytical signal of the cochleagram:
      
        - if :data:`"input"`, then compute the analytical signal
          of the input (fast, accurate in the middle, bad boundary conditions)
        - if :data:`"ir"` (suggested), then compute the analytical signal
          of the IRs (fast, tends to underestimate amplitude,
          good boundary conditions)
        - if :data:`"output"`, then compute the analytical signal
          of the output (slowest, most accurate)
      postprocess (callable): If not :data:`None`, then apply this function
        to the cochleagram matrix. Default is :func:`hwr`, if the cochleagram
        is real, otherwise it is :data:`None`
      method (str): Convolution method (either :data:`"auto"`,
        :data:`"fft"`, :data:`"direct"`, or :data:`"overlap-add"`)
      stride (int): Time-step for output signal.
        Can't be used in conjunction with :data:`mehtod`

    Returns:
      matrix: Cochleagram"""
    return cochleagram(x=x,
                       fs=fs,
                       filterbank=self,
                       analytical=analytical,
                       method=method,
                       **kwargs)[0]


@utils.numpy_out
def hwr(a: np.ndarray, th: float = 0, out: Optional[np.ndarray] = None):
  """Half-wave rectification

  Args:
    a (array): Input signal
    th (float): Threshold. Default is :data:`0`
    out (array): Optional. Array to use for storing results

  Returns:
    array: Half-wave rectified copy of input signal"""
  return np.maximum(a, th, out=out)


def cochleagram(
    x: Sequence[float],
    fs: Optional[float] = None,
    filterbank: Optional[Union[GammatoneFilterbank,
                               GammatoneFilterbank.PrecomputedIRBank]] = None,
    analytical: Optional[str] = None,
    method: Optional[str] = None,
    stride: Optional[int] = None,
    **kwargs):
  # pylint: disable=C0303
  """Compute the cochleagram for the signal

  Args:
    x (array): Array of audio samples
    fs (float): Sampling frequency
    filterbank (GammatoneFilterbank): Filterbank object, or precomputed IRs.
      If unspecified, it will be specified using :data:`**kwargs`
    postprocessing (callable): If not :data:`None`, then apply this function
      to the cochleagram matrix. Default is :func:`hwr`, if the cochleagram
      is real, otherwise it is :data:`None`
    analytical (str): Compute the analytical signal of the cochleagram:
      
      - if :data:`"input"`, then compute the analytical signal
        of the input (fast, accurate in the middle, bad boundary conditions)
      - if :data:`"ir"` (suggested), then compute the analytical signal
        of the IRs (fast, tends to underestimate amplitude,
        good boundary conditions)
      - if :data:`"output"`, then compute the analytical signal
        of the output (slowest, most accurate)
    method (str): Convolution method (either :data:`"auto"`,
      :data:`"fft"`, :data:`"direct"`, or :data:`"overlap-add"`)
    stride (int): Time-step for output signal.
      Can't be used in conjunction with :data:`method`
    **kwargs: Keyword arguments for :class:`GammatoneFilterbank`

  Returns:
    matrix, array: Cochleagram matrix (filter x time) and the array of center
    frequencies"""
  if isinstance(
      filterbank,
      GammatoneFilterbank.PrecomputedIRBank) and filterbank.analytical:
    if analytical not in (None, "ir"):
      raise ValueError("When IR bank is analytical, only None and 'ir' are "
                       "supported as options for argument 'analytical'. "
                       f"Got: '{analytical}'")
    analytical = "ir"
  if "postprocessing" in kwargs:
    postprocessing = kwargs.pop("postprocessing")
  elif analytical is None:
    postprocessing = hwr
  else:
    postprocessing = None
  if filterbank is None:
    filterbank = GammatoneFilterbank(**kwargs)
  if not isinstance(filterbank, GammatoneFilterbank.PrecomputedIRBank):
    if fs is None:
      raise TypeError("cochleagram() missing required argument "
                      "'fs' for non-precomputed filterbank")
    filterbank = filterbank.precompute(fs=fs, analytical=analytical == "ir")
  if analytical == "input":
    x = signal.hilbert(x)
  out = filterbank.convolve(x, method=method, stride=stride)
  if analytical == "output":
    out = signal.hilbert(out)
  if postprocessing is not None:
    out = postprocessing(out)
  return out, np.array(filterbank.freqs)


def mel_triangular_filterbank(
    freqs: Sequence[float],
    n_filters: Optional[int] = None,
    bandwidth: Optional[Callable[[float], float]] = None,
    flim: Optional[Sequence[float]] = None,
    freq_transform: Tuple[Callable[[float], float],
                          Callable[[float], float]] = (hz2mel, mel2hz),
):
  """Compute a frequency-domain triangular filterbank. Specify at least one of
  :data:`n_filters`, :data:`bandwidth`, or :data:`flim`

  Args:
    freqs (array): Frequency axis for frequency-domain filters
    n_filters (int): Number of filters. If :data:`None` (default), infer from
      other arguments
    bandwidth (callable): Bandwidth function that maps a center frequency to
      the -3 dB bandwidth of the filter at that frequency. If :data:`None`,
      (default), then one filter's -inf dB cutoff frequencies will be the
      center frequencies of the previous and the next filter (50% overlapping
      filters). In this case, the frequency limits :data:`flim` include the
      lower cutoff frequency of the first filter and the higher cutoff
      frequency of the last filter.
      If a function is provided, then the frequency limits :data:`flim` are
      only the center frequencies of the filters
    flim (array): Corner/center frequencies for the filters. If
      :data:`n_filters` and :data:`bandwidth` are both :data:`None`, they must
      be 2 more than the number of desired filters. If :data:`None`, then
      it will be set to the first and last elements of :data:`freqs`
    freq_transform: Couple of callables that implement transformations
      from and to Hertz, respectively. If :data:`n_filters` is not
      :data:`None`, the center frequencies of the triangular filters will be
      chosen linearly between :data:`freqs[0]` and :data:`freqs[1]` in the
      transformed space. Default is :func:`hz2mel`, :func:`mel2hz` for
      linear spacing on the Mel scale

  Returns:
    matrix, array: The triangular filterbank matrix (filter x frequency) and
    the array of center frequencies"""
  if flim is None:
    flim = freqs[0], freqs[-1]
  f = bandwidth is None
  if n_filters is None:
    # Use flims as center frequencies
    n_filters = len(flim)
    if f:
      # or as corner frequencies
      n_filters -= 2
  else:
    # Use flims as corner frequencies of the first and last filter
    # => interpolate center frequencies
    flim = freq_transform[1](np.linspace(freq_transform[0](flim[0]),
                                         freq_transform[0](flim[-1]),
                                         n_filters + 2))
    if not f:
      # Center frequencies only
      flim = flim[1:-1]
  if n_filters <= 0:
    n_freqs = len(flim)
    raise ValueError(
        "Specify either a bandwidth function or at least 3 corner frequencies "
        f"(at least 1 filter). Got: {n_freqs} corner frequencies and "
        f"""{"no " if f else ""}bandwidth function """
        f"""{"" if f else f"'{bandwidth} '"}({n_filters} filters)""")
  if f:
    # 50% overlapping triangular windows
    freqs_l = flim[:-2]
    freqs_c = flim[1:-1]
    freqs_r = flim[2:]
  else:
    # User-specified bandwidth
    # Do not divide, because
    #   - width is double the "radius"        => /2
    #   - -3dB is at half-way of the triangle => *2
    b = np.array(list(map(bandwidth, flim)))
    freqs_l = flim - b
    freqs_c = flim
    freqs_r = flim + b
  filts = np.empty((n_filters, *np.shape(freqs)))
  for i in range(n_filters):
    filts[i, ...] = np.interp(freqs, [freqs_l[i], freqs_c[i], freqs_r[i]],
                              [0, 1, 0])
  return filts, freqs_c


def stft2mel(stft: Sequence[Sequence[complex]],
             freqs: Sequence[float],
             filterbank: Optional[Sequence[Sequence[float]]] = None,
             power: Optional[float] = 2,
             **kwargs):
  """Compute the mel-spectrogram from a STFT

  Args:
    stft (matrix): STFT matrix (frequency x time)
    freqs (array): Frequencies axis for :data:`stft`
    filterbank (matrix): Filterbank matrix. If unspecified, it will be
      computed with :func:`mel_triangular_filterbank`
    power (float): Power for magnitude computation before frequency-domain
      filtering. After filtering, the inverse power is computed for
      consistence. Default is :data`2`. If :data`None`, then filter the
      complex stft matrix
    **kwargs: Keyword arguments for :func:`mel_triangular_filterbank`

  Returns:
    matrix, array: Mel-spectrogram matrix (filter x time) and the array of
    center frequencies (only if :data:`filterbank` is unspecified,
    otherwise :data:`None`)"""
  if filterbank is None:
    filterbank, c_freqs = mel_triangular_filterbank(freqs, **kwargs)
  else:
    c_freqs = None
  p = False
  if power is not None:
    stft = np.abs(stft)
    if power != 1:
      p = True
      np.power(stft, power, out=stft)
  melspec = filterbank @ stft
  if p:
    np.power(melspec, 1 / power, out=melspec)
  return melspec, c_freqs


def mel_spectrogram(x: Sequence[float],
                    stft_kws: Optional[Dict[str, Any]] = None,
                    **kwargs):
  """Compute the mel-spectrogram from a STFT

  Args:
    x (array): Array of audio samples
    stft_kws: Keyword arguments for :func:`scipy.signal.stft`
    **kwargs: Keyword arguments for :func:`stft2mel`

  Returns:
    array, array, matrix: The array of center frequencies, the array of
    time-steps, and the Mel-spectrogram matrix (filter x time)"""
  freqs, times, stft = signal.stft(x, **stft_kws)
  melspec, c_freqs = stft2mel(stft=stft, freqs=freqs, **kwargs)
  return c_freqs, times, melspec
