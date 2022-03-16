"""Tests for utility functions"""
import itertools
import unittest

import numpy as np
from chromatictools import unittestmixins
from sample import utils


class TestUtils(unittestmixins.RMSEAssertMixin, unittest.TestCase):
  """Tests for utility functions"""

  def test_normalize_peak(self):
    """Test peak normalization"""
    np.random.seed(42)
    a = np.random.rand(1024)
    idxs = np.floor(np.random.rand(16) * a.size).astype(int)
    p = 1e3
    for i, s in itertools.product(idxs, (-1, 1)):
      a_ = a.copy()
      a_[i] = s * p
      b = utils.normalize(a_)
      with self.subTest(peak_position=i, sign=s, test="peak is one"):
        self.assertEqual(np.abs(b).max(), 1)
      with self.subTest(peak_position=i, sign=s, test="rescaling is correct"):
        self.assert_almost_equal_rmse(a_ / p, b)

  def test_normalize_rms(self):
    """Test RMS normalization"""
    np.random.seed(42)
    a = np.random.rand(1024)
    b = utils.normalize(a, mode="rms")
    with self.subTest(test="mean is zero"):
      self.assertAlmostEqual(np.mean(b), 0)
    with self.subTest(test="std is one"):
      self.assertAlmostEqual(np.std(b), 1)
