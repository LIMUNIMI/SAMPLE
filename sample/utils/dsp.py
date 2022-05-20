"""DSP utilities"""
import functools
from typing import Callable, Optional

import numpy as np
from sample import utils
from scipy import signal
from sklearn import base, linear_model


@utils.function_with_variants(key="mode", default="peak", this="peak")
@utils.numpy_out
def normalize(
    x: np.ndarray,
    mode: str = "peak",  # pylint: disable=W0613
    out: Optional[np.ndarray] = None) -> np.ndarray:
  """Normalize the array to have absolute peak at one

  Args:
    x (array): Array to normalize
    mode (str): Type of normalization (peak, rms, range)
    out (array): Optional. Array to use for storing results

  Returns:
    array: Normalized array"""
  return np.true_divide(x, np.abs(x).max(), out=out)


@utils.function_variant(normalize, "rms")
@utils.numpy_out
def _normalize_rms(x: np.ndarray,
                   out: Optional[np.ndarray] = None) -> np.ndarray:
  """Normalize the array to have zero mean and unitary variance

  Args:
    x (array): Array to normalize
    out (array): Optional. Array to use for storing results

  Returns:
    array: Normalized array"""
  np.subtract(x, np.mean(x), out=out)
  return np.true_divide(out, np.std(x) or 1, out=out)


@utils.function_variant(normalize, "range")
@utils.numpy_out
def _normalize_range(x: np.ndarray,
                     out: Optional[np.ndarray] = None) -> np.ndarray:
  """Normalize the array to have values betwen zero and one

  Args:
    x (array): Array to normalize
    out (array): Optional. Array to use for storing results

  Returns:
    array: Normalized array"""
  np.subtract(x, np.min(x), out=out)
  return np.true_divide(out, np.max(out), out=out)


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
def complex2db(c, out: Optional[np.ndarray] = None, **kwargs):
  """Convert linear complex values to decibel

  Args:
    c (array): Amplitude values
    out (array): Optional. Array to use for storing results
    **kwargs: Keyword arguments for :func:`a2db`

  Returns:
    array: Decibel values"""
  np.abs(c, out=out)
  return a2db(out, out=out, **kwargs)


def dychotomic_zero_crossing(func: Callable[[float], float],
                             lo: float,
                             hi: float,
                             steps: int = 16):
  """Dichotomicly search for a zero crossing

  Args:
    func (callable): Function to evaluate
    lo (float): Lower boundary for dichotomic search
    hi (float): Higher boundary for dichotomic search
    steps (int): Number of steps for the search

  Returns:
    float: Argument for which the function is close to zero"""
  f_lo = func(lo)
  f_hi = func(hi)
  if f_lo * f_hi > 0:
    raise ValueError("Function has the same sign at both boundaries: "
                     f"f({lo}) = {f_lo},  f({hi}) = {f_hi}")
  if f_lo > f_hi:
    f_lo, f_hi = f_hi, f_lo
    lo, hi = hi, lo
  a = (lo + hi) / 2
  for _ in range(steps):
    f_a = func(a)
    if f_a < 0:
      lo = a
    elif f_a > 0:
      hi = a
    else:
      break
    a = (lo + hi) / 2
  return a


@utils.numpy_out(dtype=np.complex64)
def expi(x: np.ndarray, out: Optional[np.ndarray] = None) -> np.ndarray:
  r"""Exponential of imaginary input

  Args:
    x (array): Input array, to be multiplied by :math:`i`
    out (array): Optional. Array to use for storing results

  Returns:
    array: :math:`e^{ix}`"""
  np.multiply(1j, x, out=out)
  np.exp(out, out=out)
  return out


column = functools.partial(np.reshape, newshape=(-1, 1))


def detrend(y: np.ndarray,
            x: Optional[np.ndarray] = None,
            model: Optional[base.BaseEstimator] = None) -> np.ndarray:
  """Remove trends from a signal

  Args:
    y (array): Array of signal samples. Must be monodimensional
    x (array): Optional. Time-steps for the signal. If unspecified
      :data:`y` is assumed to be uniformly sampled
    model (Estimator): Sklearn-like estimator to estimate signal trend.
      If :data:`None`, then construct a new instance of
      :class:`sklearn.linear_model.LinearRegression`

  Returns:
    array: The model prediction residuals"""
  if np.ndim(y) != 1:
    raise ValueError(
        f"detrend(): input arry must be 1d, but shape is {np.shape(y)}")
  if model is None:
    model = linear_model.LinearRegression()
  if x is None:
    x = np.arange(np.size(y))
  y = column(y)
  x = column(x)
  model.fit(x, y)
  return np.squeeze(y - model.predict(x))


def f2bin(f: np.ndarray, nfft: int, fs: float = 1) -> np.ndarray:
  """Convert frequency values to FFT bin indices

  Args:
    f (array): Frequency values in Hz
    nfft (int): FFT size
    fs (float): Sample frequency

  Returns:
    array: Bin indices"""
  out = np.multiply(f, nfft / fs)
  out = np.ceil(out, out=out)
  return out.astype(int)


@utils.numpy_out(dtype=float)
def bin2f(b: np.ndarray,
          nfft: int,
          fs: float = 1,
          out: Optional[np.ndarray] = None) -> np.ndarray:
  """Convert FFT bin indices to frequency values

  Args:
    b (array): Bin indices
    nfft (int): FFT size
    fs (float): Sample frequency
    out (array): Optional. Array to use for storing results

  Returns:
    array: Frequency values in Hz"""
  return np.multiply(b, fs / nfft, out=out)


def lombscargle_as_fft(t: np.ndarray,
                       x: np.ndarray,
                       nfft: int,
                       fs: float = 1,
                       hpf: Optional[float] = None,
                       lpf: Optional[float] = None,
                       **kwargs):
  """Compute a Lomb-Scargle periodogram for the same frequencies as a FFT

  Args:
    t (array): Sample times
    x (array): Signal
    nfft (int): FFT size
    fs (float): Sample frequency for the analogous uniformly-sampled signal
    hpf (float): If specified, cut off frequencies below this value (in Hz)
    lpf (float): If specified, cut off frequencies abow this value (in Hz)
    **kwargs: Keyword arguments for :func:`scipy.signal.lombscargle`

  Returns:
    array, array: The Lomb-Scargle periodogram and the frequencies
    at which it is evaluated"""
  n = nfft // 2 + 1
  ls = np.zeros(n)

  # FFT angular velocities
  w = np.arange(n, dtype=float)
  bin2f(w, nfft=nfft, fs=fs, out=w)
  np.multiply(2 * np.pi, w, out=w)

  # Band-pass corner bins
  i, j = f2bin((0 if hpf is None else hpf, fs if lpf is None else lpf),
               nfft=nfft,
               fs=fs)
  i_ = max(i, 1)
  j = min(j, n)

  # Compute
  ls[i_:j] = signal.lombscargle(x=t, y=x, freqs=w[i_:j], **kwargs)
  if i < 1:
    ls[0] = np.square(np.mean(x))

  # Angular velocity to frequency
  np.true_divide(w, 2 * np.pi, out=w)
  return ls, w


def fft2autocorrelogram(a: np.ndarray,
                        real: bool = True,
                        n: int = None,
                        nfft: int = None,
                        power: bool = False) -> np.ndarray:
  """Compute the autocorrelogram from the FFT of a signal

  Args:
    a (array): FFT
    real (bool): Wheter the FFT is one-sided (sufficient for real signals,
      default) or two-sided (output will be complex)
    n (int): Signal length in samples. Default is the size of :data:`a` for
      a two-sided FFT (:data:`real=False`) or twice as much for a one-sided
      FFT (:data:`real=True`)
    nfft (int): FFT size. Default is :data:`n`
    power (bool): If :data:`True`, the input :data:`a` is interpreted as the
      power spectrum, instead of the FFT or the magnitude

  Returns:
   array: The autocorrelogram of the signal"""
  if n is None:
    n = np.size(a)
    if real:
      n = 2 * (n - 1)
  if nfft is None:
    nfft = n
  if not power:
    a = np.multiply(a, np.conjugate(a)).real
  return (np.fft.irfft if real else np.fft.ifft)(a, n=nfft)[:n]


def lombscargle_autocorrelogram(t: np.ndarray,
                                x: np.ndarray,
                                n: Optional[int] = None,
                                nfft: Optional[int] = None,
                                fs: float = 1,
                                **kwargs) -> np.ndarray:
  """Compute the autocorrelogram of an unevenly sampled signal
  using the Lomb-Scargle periodogram

  Args:
    t (array): Sample times
    x (array): Signal
    n (int): Autocorrelogram size
    nfft (int): FFT size
    fs (float): Sample frequency for the analogous uniformly-sampled signal
    **kwargs: Keyword arguments for :func:`lombscargle_as_fft`

  Returns:
    array: Autocorrelogram"""
  if n is None:
    n = np.ceil(np.max(t) * fs).astype(int)
  if nfft is None:
    nfft = 2 * n

  ls, _ = lombscargle_as_fft(t=t, x=x, nfft=nfft, fs=fs, **kwargs)
  return fft2autocorrelogram(ls, real=True, n=n, nfft=nfft, power=True)


def n_windows(input_size: int, wsize: int, hop: Optional[int] = None) -> int:
  """Number of windows for overlapping windows operations

  Args:
    input_size (int): Size of input array
    wsize (int): Size of the windows
    hop (int): Distance in samples between consecutive windows

  Returns:
    int: Number of windows"""
  if hop is None:
    hop = wsize
  return 1 + (input_size - wsize) // hop


def overlapping_windows(a: np.ndarray,
                        wsize: int,
                        hop: Optional[int] = None,
                        writeable: bool = False):
  """Return a view on the array as a matrix of overlapping windows.
  Please note that this function does not alter in any way the input array.
  If some preprocessing is needed, e.g. zero-padding, it should be done before
  invoking this function

  Args:
    a (ndarray): The array. If it is not 1d already, it is linearized
    wsize (int): Size of the windows
    hop (int): Distance in samples between consecutive windows.
      It should be at least :data:`1`, it often is at most :data:`wsize`.
      If :data:`None`, then use :data:`wsize` (non-overlapping windows)
    writeable (bool): If :data:`False` (default) the output view is read-only

  Returns:
    ndarray: Overlapping windows. Shape is :data:`(n_windows, wsize)`

  Example:
    >>> from sample.utils.dsp import overlapping_windows
    >>> import numpy as np
    >>> a = np.arange(64)
    >>> b = overlapping_windows(a, wsize=16)
    >>> # Check array shape
    >>> b.shape
    (4, 16)
    >>> from sample.utils import numpy_id
    >>> # Test that the new array still refers to the same memory
    >>> numpy_id(a) == numpy_id(b)
    True
    >>> # Use view to perform mean filtering
    >>> np.mean(b, axis=-1)
    array([ 7.5, 23.5, 39.5, 55.5])"""
  a_lin = np.reshape(a, newshape=(-1,))
  if hop is None:
    hop = wsize
  # Stride for a single memory cell
  cell_stride, = a_lin.strides
  return np.lib.stride_tricks.as_strided(a_lin,
                                         shape=(n_windows(a_lin.size,
                                                          wsize=wsize,
                                                          hop=hop), wsize),
                                         strides=np.array(
                                             (hop, 1), dtype=int) * cell_stride,
                                         writeable=writeable)


def strided_convolution(x: np.ndarray,
                        kernel: np.ndarray,
                        stride: int = 1,
                        out: Optional[np.ndarray] = None):
  """Compute a strided convolution as a matrix multiplication between
  :func:`overlapping_windows` of the signal and the flipped kernel.

  Args:
    x (array): The input array
    kernel (array): Convolution kernel
    stride (int): Output sample period (in number of input samples).
      Note that for the default :data:`stride=1`, the output
      is the (non-strided) convolution. In this case, consider using
      other convolution functions, which will be more efficient
    out (array): Optional. Array to use for storing results

  Returns:
    array: Strided convolution"""
  a = overlapping_windows(x, wsize=np.size(kernel), hop=stride)
  b = np.reshape(kernel[::-1], newshape=(-1, 1))
  return np.matmul(a, b, out=out)


def strided_convolution_complex_kernel(x: np.ndarray,
                                       kernel: np.ndarray,
                                       stride: int = 1,
                                       out: Optional[np.ndarray] = None,
                                       dtype: Optional[type] = None):
  """Compute a strided convolution as a matrix multiplication between
  :func:`overlapping_windows` of the signal and the flipped kernel.
  Call this function when kernel is complex

  Args:
    x (array): The input array
    kernel (array): Convolution kernel
    stride (int): Output sample period (in number of input samples).
      Note that for the default :data:`stride=1`, the output
      is the (non-strided) convolution. In this case, consider using
      other convolution functions, which will be more efficient
    out (array): Optional. Array to use for storing results
    dtype (type): Explicit dtype for output when :data:`out` is unspecified

  Returns:
    array: Strided convolution"""
  if out is None:
    if dtype is None:
      dtype = np.result_type(x, kernel)
    out = np.zeros(n_windows(x.size, wsize=kernel.size, hop=stride),
                   dtype=dtype)
  for p in (np.real, np.imag):
    strided_convolution(x=x,
                        kernel=p(kernel),
                        stride=stride,
                        out=np.reshape(p(out), newshape=(-1, 1)))
  return out
