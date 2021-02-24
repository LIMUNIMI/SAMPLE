"""Utilities for test cases"""
import numpy as np
import functools
import unittest
import os


def rmse(x: np.ndarray, y: np.ndarray) -> float:
  """Compute the root-mean-squared error

  Arguments:
    x (array): First array
    y (array): Second array

  Returns:
    (float): The RMSE between x and y"""
  return np.sqrt(np.mean(np.abs(x - y)**2))


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
