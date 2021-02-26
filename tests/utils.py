"""Utilities for test cases"""
import numpy as np
from scipy import special
import functools
import unittest
import os


def mantissa(x: np.array) -> np.ndarray:
  """Remove order of magnitude from array

  Args:
  x (array): Inputs

  Returns:
    array: Rescaled array"""
  y = np.log10(np.abs(x))
  return np.sign(x) * special.exp10(y - np.floor(np.max(y)))


def rmse(x: np.ndarray, y: np.ndarray) -> float:
  """Compute the root-mean-squared error

  Arguments:
    x (array): First array
    y (array): Second array

  Returns:
    (float): The RMSE between x and y"""
  return np.sqrt(np.mean(np.abs(x - y)**2))


_d = 1 - np.log10(5)


class SignificantPlacesAssertMixin:
  """Mixin class for Test cases that require assertions based on
  singificant places"""
  def assert_almost_equal_significant(
    self,
    first,
    second,
    places=0,
    msg=None,
    delta=None
  ):
    """Fail if the two objects are unequal as determined by their
    difference rounded to the given number of significant decimal places
    (default 0)"""
    return self.assertAlmostEqual(
      *mantissa([first, second]),
      places=places,
      msg=msg,
      delta=delta
    )


class RMSEAssertMixin:
  """Mixin class for Test cases that require assertions based on the RMSE"""
  def _assert_rmse(
    self,
    x: np.ndarray,
    y: np.ndarray,
    almost: bool,
    *args,
    **kwargs):
    """Assert that the RMSE is zero

    Arguments:
      x (array): First array
      y (array): Second array
      almost (bool): If True, then assert that the RMSE is almost zero
      *args: Positional arguments for :func:`unittest.TestCase.assertEqual`
        or :func:`unittest.TestCase.assertAlmostEqual`
      **kwargs: Keyword arguments for :func:`unittest.TestCase.assertEqual`
        or :func:`unittest.TestCase.assertAlmostEqual`"""
    return (self.assertAlmostEqual if almost else self.assertEqual)(
      rmse(x, y), 0,
      *args, **kwargs,
    )

  assert_equal_rmse = functools.partialmethod(_assert_rmse, almost=False)
  assert_almost_equal_rmse = functools.partialmethod(_assert_rmse, almost=True)


more_tests = unittest.skipUnless(
  os.environ.get("SAMPLE_MORE_TESTS", False),
  "enabled only if SAMPLE_MORE_TESTS is set"
)
