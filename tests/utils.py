"""Utilities for test cases"""
import numpy as np
import functools
import unittest
import os


def test_audio(
  f: np.ndarray = np.array([440, 650, 690]),
  a: np.ndarray = np.array([1, .5, .45]),
  d: np.ndarray = np.array([3, 7, 5]),
  dur: float = 2,
  fs: int = 44100,
  noise_db: float = -60,
  seed: int = 42,
):
  """Synthesize a modal-like sound for test purposes

  Args:
    f (array): Modal frequencies
    a (array): Modal amplitudes
    d (array): Modal decays
    dur (float): Duration in seconds
    fs (int): Sampling frequency in Hz
    noise_db (float): Gaussian noise magnitude in dB
    seed (int): Gaussian noise seed

  Returns:
    array: Array of audio samples"""
  t = np.linspace(0, dur, int(dur * fs), endpoint=False)
  x = np.squeeze(np.reshape(a, (1, -1)) @ (
    np.exp(-np.reshape(d, (-1, 1)) * np.reshape(t, (1, -1))) *
    np.sin(np.reshape(f, (-1, 1)) * 2 * np.pi * np.reshape(t, (1, -1)))
  ))
  x = x / np.max(np.abs(x))

  np.random.seed(seed)
  x = x + 10**(noise_db/20) * np.random.randn(*x.shape)
  x = x / np.max(np.abs(x))
  return x


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
