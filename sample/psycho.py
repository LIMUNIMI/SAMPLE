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
  return _a2db_floorless(a, out=out)


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
