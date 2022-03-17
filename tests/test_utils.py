"""Tests for utility functions"""
import itertools
import unittest

import numpy as np
from chromatictools import unittestmixins
from sample.utils import dsp as dsp_utils

from tests import utils as test_utils


class TestDSP(unittestmixins.RMSEAssertMixin,
              unittestmixins.AssertDoesntRaiseMixin, unittest.TestCase):
  """Tests for dsp utility functions"""

  def test_normalize_peak(self):
    """Test peak normalization"""
    np.random.seed(42)
    a = np.random.rand(1024)
    idxs = np.floor(np.random.rand(16) * a.size).astype(int)
    p = 1e3
    for i, s in itertools.product(idxs, (-1, 1)):
      a_ = a.copy()
      a_[i] = s * p
      b = dsp_utils.normalize(a_)
      with self.subTest(peak_position=i, sign=s, test="peak is one"):
        self.assertEqual(np.abs(b).max(), 1)
      with self.subTest(peak_position=i, sign=s, test="rescaling is correct"):
        self.assert_almost_equal_rmse(a_ / p, b)

  def test_normalize_rms(self):
    """Test RMS normalization"""
    np.random.seed(42)
    a = np.random.rand(1024)
    b = dsp_utils.normalize(a, mode="rms")
    with self.subTest(test="mean is zero"):
      self.assertAlmostEqual(np.mean(b), 0)
    with self.subTest(test="std is one"):
      self.assertAlmostEqual(np.std(b), 1)

  @test_utils.coherence_check_method(fwd=dsp_utils.db2a,
                                     bak=dsp_utils.a2db,
                                     f=np.linspace(-60, 60, 1024))
  def test_db(self):
    """Test coherence of conversion for dB"""
    pass

  @test_utils.coherence_check_method(fwd=dsp_utils.db2a,
                                     bak=dsp_utils.a2db,
                                     f=np.linspace(-60, 60, 1024).tolist())
  def test_db_list(self):
    """Test coherence of conversion for dB using
    a list instead of a ndarray"""
    pass

  @test_utils.coherence_check_method(fwd=lambda *args, **kwargs: dsp_utils.db2a(
      *args, **kwargs).astype(complex),
                                     bak=dsp_utils.complex2db,
                                     f=np.linspace(-60, 60, 1024))
  def test_complex_db(self):
    """Test coherence of conversion for dB from complex"""
    pass

  def test_db_floor(self):
    """Test floor for dB conversion"""
    f = -60
    a = np.linspace(0, 1, 1024)
    dsp_utils.a2db(a, floor=f, floor_db=True, out=a)
    with self.subTest(check="dB"):
      self.assertTrue(np.greater_equal(a, f).all())
    dsp_utils.db2a(a, out=a)
    with self.subTest(check="a"):
      self.assertTrue(np.greater_equal(a, dsp_utils.db2a(f)).all())
