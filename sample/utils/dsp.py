"""DSP utilities"""
from typing import Callable, Optional

import numpy as np
from sample import utils


@utils.function_with_variants(key="mode", default="peak", this="peak")
@utils.numpy_out
def normalize(
    x: np.ndarray,
    mode: str = "peak",  # pylint: disable=W0613
    out: Optional[np.ndarray] = None) -> np.ndarray:
  """Normalize the array to have absolute peak at one

  Args:
    x (array): Array to normalize
    mode (str): Type of normalization (peak, rms)
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
