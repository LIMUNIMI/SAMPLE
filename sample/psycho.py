"""Classes and functions related to psychoacoustic models"""
from typing import Optional
import functools

import numpy as np

from sample import utils


@functools.partial(np.frompyfunc, nin=1, nout=1)
def db2a(d: float) -> float:
  """Convert decibels to linear amplitude values

  Args:
    d (array): Decibel values
    **kwargs: For other keyword-only arguments, see the `ufunc docs`_

  Returns:
    array: Amplitude values

  .. _ufunc docs:
    https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs"""
  return 10**(d / 20)


def a2db(a: np.ndarray,
         floor: Optional[np.ndarray] = None,
         floor_db: bool = False,
         **kwargs):
  """Convert linear amplitude values to decibel

  Args:
    a (array): Amplitude values
    floor (array): Floor value(s). If specified, the amplitude values will be
      clipped to this value. Use this to avoid computing the logarithm of zero
    floor_db (bool): Set this to :data:`True` if :data:`floor` is
      specified in decibel
    **kwargs: For other keyword-only arguments, see the `ufunc docs`_

  Returns:
    array: Decibel values

  .. _ufunc docs:
    https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs"""
  if floor is None:
    return a2db.floorless(a, **kwargs)
  if floor_db:
    floor = db2a(floor)
  return a2db.floored(a, floor)


@functools.partial(np.frompyfunc, nin=1, nout=1)
def _a2db_floorless(a: float) -> float:
  """Convert linear amplitude values to decibel

  Args:
    a (array): Amplitude values
    **kwargs: For other keyword-only arguments, see the `ufunc docs`_

  Returns:
    array: Decibel values

  .. _ufunc docs:
    https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs"""
  return 20 * np.log10(a)


a2db.floorless = _a2db_floorless


@functools.partial(np.frompyfunc, nin=2, nout=1)
def _a2db_floored(a: float, f: float) -> float:
  """Convert linear amplitude values to decibel, specifying a floor

  Args:
    a (array): Amplitude values
    floor (array): Floor value(s). If specified, the amplitude values will be
      clipped to this value. Use this to avoid computing the logarithm of zero
    **kwargs: For other keyword-only arguments, see the `ufunc docs`_

  Returns:
    array: Decibel values

  .. _ufunc docs:
    https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs"""
  return _a2db_floorless(max(a, f))


a2db.floored = _a2db_floored


@utils.function_with_variants(key="mode", default="traunmuller")
def hz2bark(f, mode: str = "traunmuller"):  # pylint: disable=W0613
  """Convert Hertz to Bark

  Args:
    f: Frequency value(s) in Hertz
    mode (str): Name of the Bark definition (zwicker, traunmuller, or wang)

  Returns:
    Frequency value(s) in Bark"""
  pass  # pragma: no cover


@utils.function_with_variants(key="mode", default="traunmuller")
def bark2hz(b, mode: str = "traunmuller"):  # pylint: disable=W0613
  """Convert Bark to Hertz

  Args:
    b: Frequency value(s) in Bark
    mode (str): Name of the Bark definition (traunmuller, or wang)

  Returns:
    Frequency value(s) in Hertz"""
  pass  # pragma: no cover


@utils.function_variant(hz2bark, "zwicker")
def _hz2bark_zwicker(f):
  """Original definition of the Bark scale (Zwicker & Terhardt (1980))

  Args:
    f: Frequency value(s) in Hertz

  Returns:
    Frequency value(s) in Bark"""
  return 13.0 * np.arctan(7.6e-4 * f) + 3.5 * np.arctan(np.square(f / 7500.0))


@utils.function_variant(hz2bark, "traunmuller")
def _hz2bark_traunmuller(f):
  """Definition of the Bark scale by Traunmuller
  (Analytical expressions for the tonotopic sensory scale, 1990)

  Args:
    f: Frequency value(s) in Hertz

  Returns:
    Frequency value(s) in Bark"""
  return (26.81 * f / (1960.0 + f)) - 0.53


@utils.function_variant(hz2bark, "wang")
def _hz2bark_wang(f):
  """Definition of the Bark scale by Wang et al. (An objective measure for
  predicting subjective quality of speech coders, 1992)

  Args:
    f: Frequency value(s) in Hertz

  Returns:
    Frequency value(s) in Bark"""
  return 6.0 * np.arcsinh(f / 600.0)


@utils.function_variant(bark2hz, "traunmuller")
def _bark2hz_traunmuller(b):
  """Definition of the Bark scale by Traunmuller
  (Analytical expressions for the tonotopic sensory scale, 1990)

  Args:
    b: Frequency value(s) in Bark

  Returns:
    Frequency value(s) in Hertz"""
  return (0.53 + b) / (26.28 - b) * 1960.0


@utils.function_variant(bark2hz, "wang")
def _bark2hz_wang(b):
  """Definition of the Bark scale by Wang et al. (An objective measure for
  predicting subjective quality of speech coders, 1992)

  Args:
    b: Frequency value(s) in Bark

  Returns:
    Frequency value(s) in Hertz"""
  return 600.0 * np.sinh(b / 6.0)


@utils.function_with_variants(key="mode", this="default")
def hz2mel(f, mode: str = "default"):  # pylint: disable=W0613
  """Convert Hertz to Mel

  Args:
    f: Frequency value(s) in Hertz
    mode (str): Name of the Mel definition (default, fant)

  Returns:
    Frequency value(s) in Mel"""
  return 2595.0 * np.log10(1.0 + f / 700.0)


@utils.function_with_variants(key="mode", this="default")
def mel2hz(m, mode: str = "default"):  # pylint: disable=W0613
  """Convert Mel to Hertz

  Args:
    m: Frequency value(s) in Mel
    mode (str): Name of the Mel definition (default, fant)

  Returns:
    Frequency value(s) in Hertz"""
  return (np.power(10, m / 2595.0) - 1.0) * 700.0


@utils.function_variant(hz2mel, "fant")
def _hz2mel_fant(f):
  """Definition of the Mel scale by Fant (Analysis and synthesis
  of speech processes, 1968)

  Args:
    f: Frequency value(s) in Hertz

  Returns:
    Frequency value(s) in Mel"""
  return 1000.0 * np.log2(1.0 + f / 1000.0)


@utils.function_variant(mel2hz, "fant")
def _mel2hz_fant(m):
  """Definition of the Mel scale by Fant (Analysis and synthesis
  of speech processes, 1968)

  Args:
    m: Frequency value(s) in Mel

  Returns:
    Frequency value(s) in Hertz"""
  return (np.power(2.0, m / 1000.0) - 1.0) * 1000.0
