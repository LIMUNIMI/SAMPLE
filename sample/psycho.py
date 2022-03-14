"""Classes and functions related to psychoacoustic models"""
from typing import Optional

import numpy as np

from sample import utils


@utils.numpy_out
def db2a(d: float, out: Optional[np.ndarray] = None) -> float:
  """Convert decibels to linear amplitude values

  Args:
    d (array): Decibel values
    out (array): Optional. Array to use for storing results

  Returns:
    array: Amplitude values"""
  np.true_divide(d, 20.0, out=out)
  return np.power(10.0, out, out=out)


@utils.numpy_out
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


@utils.numpy_out
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


@utils.numpy_out
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
@utils.numpy_out
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
@utils.numpy_out
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
@utils.numpy_out
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
@utils.numpy_out
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
@utils.numpy_out
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
@utils.numpy_out
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
@utils.numpy_out
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
@utils.numpy_out
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
@utils.numpy_out
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
@utils.numpy_out
def erb(f: float,
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
  # 6.23 * (f/1000)^2 + 0.009339 * f + 28.52
  np.true_divide(f, 1000, out=out)
  np.square(out, out=out)
  np.multiply(6.23, out, out=out)
  np.add(out, tmp, out=out)
  return np.add(out, 28.52, out=out)


@utils.function_variant(erb, "linear")
@utils.numpy_out
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
@utils.numpy_out
def hz2cams(f: float,
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
@utils.numpy_out
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
@utils.numpy_out
def cams2hz(c: float,
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
@utils.numpy_out
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
