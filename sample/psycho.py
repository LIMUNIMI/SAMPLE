"""Classes and functions related to psychoacoustic models"""
import functools
from typing import Callable, Optional, Sequence, Tuple, Union, Dict, Any

import numpy as np
from scipy import signal

from sample import utils


@utils.numpy_out(dtype=float)
def db2a(d: float, out: Optional[np.ndarray] = None) -> float:
  """Convert decibels to linear amplitude values

  Args:
    d (array): Decibel values
    out (array): Optional. Array to use for storing results

  Returns:
    array: Amplitude values"""
  np.true_divide(d, 20.0, out=out)
  return np.power(10.0, out, out=out)


@utils.numpy_out(dtype=float)
def a2db(a: np.ndarray,
         floor: Optional[np.ndarray] = None,
         floor_db: bool = False,
         out: Optional[np.ndarray] = None):
  """Convert linear amplitude values to decibel

  Args:
    a (array): Amplitude values
    floor (array): Floor value(s). If specified, the amplitude values will be
      clipped to this value. Use this to avoid computing the logarithm of zero
    floor_db (bool): Set this to :data:`True` if :data:`floor` is
      specified in decibel
    out (array): Optional. Array to use for storing results

  Returns:
    array: Decibel values"""
  if floor is None:
    return a2db.floorless(a, out=out)
  if floor_db:
    floor = db2a(floor)
  return a2db.floored(a, floor, out=out)


@utils.numpy_out(dtype=float)
def _a2db_floorless(a: float, out: Optional[np.ndarray] = None) -> float:
  """Convert linear amplitude values to decibel

  Args:
    a (array): Amplitude values
    out (array): Optional. Array to use for storing results

  Returns:
    array: Decibel values"""
  np.log10(a, out=out)
  return np.multiply(20.0, out, out=out)


a2db.floorless = _a2db_floorless


@utils.numpy_out(dtype=float)
def _a2db_floored(a: float,
                  f: float,
                  out: Optional[np.ndarray] = None) -> float:
  """Convert linear amplitude values to decibel, specifying a floor

  Args:
    a (array): Amplitude values
    floor (array): Floor value(s). If specified, the amplitude values will be
      clipped to this value. Use this to avoid computing the logarithm of zero
    out (array): Optional. Array to use for storing results

  Returns:
    array: Decibel values"""
  np.maximum(a, f, out=out)
  return _a2db_floorless(out, out=out)


a2db.floored = _a2db_floored


@utils.numpy_out(dtype=float, dtype_promote=False)
def complex2db(c, out=None, **kwargs):
  """Convert linear complex values to decibel

  Args:
    c (array): Amplitude values
    out (array): Optional. Array to use for storing results
    **kwargs: Keyword arguments for :func:`a2db`

  Returns:
    array: Decibel values"""
  np.abs(c, out=out)
  return a2db(out, out=out, **kwargs)


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


def gammatone_leadtime(n: int, b: float) -> float:
  """Default leading time for gammatone fiters

  Args:
    n (int): Filter order
    b (float): Filter bandwidth

  Returns:
    float: Leading time"""
  return (n - 1) / (2 * np.pi * b)


def gammatone_phase(f: float, t_c: float) -> float:
  """Default phase for gammatone fiters

  Args:
    f (float): Center frequency
    t_c (float): Leading time

  Returns:
    float: Phase"""
  return -2 * np.pi * f * t_c


def _preprocess_gammatone_time(t=None,
                               size: Optional[int] = None,
                               fs: float = 1):
  if t is not None:
    return t
  if size is None:
    raise ValueError("Please, specify either time axis or filter size")
  return np.arange(size) / fs


def gammatone_filter(
    f: float,
    t=None,
    size: Optional[int] = None,
    fs: float = 1,
    n: int = 4,
    a: float = 1,
    b: Union[float, Callable[[float], float]] = erb,
    t_c: Union[float, Callable[[int, float], float]] = gammatone_leadtime,
    phi: Union[float, Callable[[float, float], float]] = gammatone_phase,
    a_norm: bool = False,
):
  """Compute a gammatone filter IR

  Args:
    f (float): Center frequency
    t (array): Time axis. If provided, arguments :data:`size` and :data:`fs`
      will be ignored
    size (int): Number of samples in the filter
    fs (float): Sample frequency
    n (int): Filter order
    a (float): IR amplitude
    b (float): Filter bandwidth. If callable, it is a function of the center
      frequency. Default is :func:`erb`
    t_c (float): Leading time for alignment. If callable, it is a function of
      the filter order and the bandwidth.
      Default is :func:`gammatone_leadtime`
    phi (float): Phase for the filter tone. If callable, it is a function of
      the center frequency and the leading time.
      Default is :func:`gammatone_phase`
    a_norm (bool): If :data:`True`, then normalize IR square norm to :data:`a`

  Returns:
    array: Gammatone filter IR"""
  t = _preprocess_gammatone_time(t=t, size=size, fs=fs)

  if callable(b):
    b = b(f)
  if callable(t_c):
    t_c = t_c(n, b)
  if callable(phi):
    phi = phi(f, t_c)

  tmp = np.empty_like(t)
  filt = np.empty_like(t)
  # t + t_c
  t_ = np.empty_like(t)
  np.add(t, t_c, out=t_)
  # u(t + t_c)
  np.greater_equal(t_, 0, out=filt)
  # u(t + t_c) * (t + t_c)^(n-1)
  np.power(t_, n - 1, out=tmp)
  np.multiply(tmp, filt, out=filt)
  # u(t + t_c) * (t + t_c)^(n-1) * exp(-2pi * (t + t_c) * b)
  np.multiply(-2 * np.pi * b, t_, out=tmp)
  np.exp(tmp, out=tmp)
  np.multiply(tmp, filt, out=filt)
  # u(t + t_c) * (t + t_c)^(n-1) * exp(-2pi * (t + t_c) * b) *
  # * cos(2pi * f * t + phi)
  np.multiply(2 * np.pi * f, t, out=tmp)
  np.add(tmp, phi, out=tmp)
  np.cos(tmp, out=tmp)
  np.multiply(tmp, filt, out=filt)
  # normalize
  if a_norm:
    np.square(filt, out=tmp)
    norm = np.sqrt(np.sum(tmp) / fs)
    np.true_divide(filt, norm, out=filt)
  # scale
  np.multiply(a, filt, out=filt)
  return filt


def gammatone_filterbank(freqs: Sequence[float] = (20, 20000),
                         n_filters: Optional[int] = None,
                         freq_transform: Tuple[Callable[[float], float],
                                               Callable[[float],
                                                        float]] = (hz2cams,
                                                                   cams2hz),
                         t=None,
                         size: Optional[int] = None,
                         fs: float = 1,
                         n: int = 4,
                         t_c: Union[float,
                                    Callable[[int, float],
                                             float]] = gammatone_leadtime,
                         **kwargs):
  """Compute the IRs of a gamatone filter-bank

  Args:
    freqs: If :data:`n_filters` is :data:`None`, the center
      frequencies of the gammatone filters. Otherwise,
      the center frequencies of the first and the last gammatone filters
    n_filters (int): Number of gammatone filters
    freq_transform: Couple of callables that implement transformations
      from and to Hertz, respectively. If :data:`n_filters` is not
      :data:`None`, the center frequencies of the gammatone filters will be
      chosen linearly between :data:`freqs[0]` and :data:`freqs[1]` in the
      transformed space. Default is :func:`hz2cams`, :func:`cams2hz` for
      linear spacing on the ERB-rate scale
    t (array): Time axis. If provided, arguments :data:`size` and :data:`fs`
      will be ignored
    size (int): Number of samples in the filters
    fs (float): Sample frequency
    n (int): Filter order
    t_c (float): Leading time for alignment. If callable, it is a function of
      the filter order and the bandwidth.
      Default is :func:`gammatone_leadtime`
    **kwargs: Keyword arguments for :func:`gammatone_filter`

  Returns:
    matrix, array: The gammatone filterbank matrix (filter x time) and the
    array of center frequencies"""
  if n_filters is not None:
    freqs = freq_transform[1](np.linspace(freq_transform[0](freqs[0]),
                                          freq_transform[0](freqs[-1]),
                                          n_filters))
  t_ = _preprocess_gammatone_time(t=t, size=size, fs=fs)
  if t is None:
    t_ -= max(map(functools.partial(t_c, n), freqs)) if callable(t_c) else t_c
  return np.array(
      [gammatone_filter(f=f, t=t_, t_c=t_c, **kwargs) for f in freqs]), freqs


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


def cochleagram(x: Sequence[float],
                filterbank: Optional[Sequence[Sequence[float]]] = None,
                nonlinearity: Optional[Callable[[np.ndarray],
                                                np.ndarray]] = hwr,
                convolve_kws: Optional[Dict[str, Any]] = None,
                **kwargs):
  """Compute the cochleagram for the signal

  Args:
    x (array): Array of audio samples
    filterbank (matrix): Filterbank matrix. If unspecified, it will be
      computed with :func:`gammatone_filterbank`
    nonlinearity (callable): If not :data:`None`, then apply this nonlinearity
      to the cochleagram matrix. Default is :func:`hwr`
    convolve_kws: Keyword arguments for :func:`scipy.signal.convolve`
    **kwargs: Keyword arguments for :func:`gammatone_filterbank`

  Returns:
    matrix, array: Cochleagram matrix (filter x time) and the array of center
    frequencies (only if :data:`filterbank` is unspecified, otherwise
    :data:`None`)"""
  if filterbank is None:
    filterbank, freqs = gammatone_filterbank(**kwargs)
  else:
    freqs = None
  if convolve_kws is None:
    convolve_kws = {}
  m = np.array(
      [signal.convolve(x, filt, **convolve_kws) for filt in filterbank])
  if nonlinearity is not None:
    try:
      nonlinearity(m, out=m)
    except TypeError:
      m = nonlinearity(m)
  return m, freqs


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
  filts = np.empty((n_filters, *freqs.shape))
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
