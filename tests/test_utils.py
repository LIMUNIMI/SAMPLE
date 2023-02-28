"""Tests for utility functions"""
import itertools
import unittest

import numpy as np
from chromatictools import unitdoctest, unittestmixins
from scipy import signal

import sample.utils
from sample.sms import dsp as sms_dsp
from sample.utils import dsp as dsp_utils
from sample.utils import learn as learn_utils
from tests import utils as test_utils


class DocTestLearn(metaclass=unitdoctest.DocTestMeta):
  """Doctests for :mod:`sample.utils.learn`"""
  _modules = (learn_utils,)


class DocTestDSP(metaclass=unitdoctest.DocTestMeta):
  """Doctests for :mod:`sample.utils.dsp`"""
  _modules = (dsp_utils,)


class DocTestUtils(metaclass=unitdoctest.DocTestMeta):
  """Doctests for :mod:`sample.utils`"""
  _modules = (sample.utils,)


class TestDSP(unittestmixins.SignificantPlacesAssertMixin,
              unittestmixins.RMSEAssertMixin,
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

  def test_normalize_range(self):
    """Test range normalization"""
    np.random.seed(42)
    a = np.random.randn(1024)
    dsp_utils.normalize(a, mode="range", out=a)
    with self.subTest(test="minimum is zero"):
      self.assertAlmostEqual(a.min(), 0)
    with self.subTest(test="maximum is one"):
      self.assertAlmostEqual(a.max(), 1)

  def test_detrend(self):
    """Test linear detrend"""
    np.random.seed(42)
    a = np.random.randn(1024) * 1e-3
    np.add(np.arange(a.size), a, out=a)
    self.assert_almost_equal_rmse(dsp_utils.detrend(a), 0, places=2)

  def test_rfft_autocorrelogram(self):
    """Test autocorrelogram from real FFT"""
    fs = 44100
    t = np.arange(fs) / fs
    np.random.seed(42)
    noise = np.random.randn(*t.shape)
    np.multiply(dsp_utils.db2a(-60), noise, out=noise)
    for f in (10, 1000, 10000):
      with self.subTest(f=f):
        x = np.multiply(2 * np.pi * f, t)
        np.cos(x, out=x)
        np.add(x, noise, out=x)
        x_fft = np.fft.rfft(x)
        x_corr = dsp_utils.fft2autocorrelogram(x_fft)
        self.assertEqual(x.size, x_corr.size)
        max_lag = sms_dsp.peak_detect_interp(x_corr)[0][0]
        max_f = fs / max_lag
        self.assert_almost_equal_significant(f, max_f, places=1)

  def test_fft_autocorrelogram(self):
    """Test autocorrelogram from FFT"""
    fs = 44100
    t = np.arange(fs) / fs
    np.random.seed(42)
    noise = np.random.randn(*t.shape)
    np.multiply(dsp_utils.db2a(-60), noise, out=noise)
    for f in (10, 1000, 10000):
      with self.subTest(f=f):
        x = np.multiply(2 * np.pi * f, t)
        np.cos(x, out=x)
        np.add(x, noise, out=x)
        x_fft = np.fft.fft(x)
        x_corr = dsp_utils.fft2autocorrelogram(x_fft, real=False)
        self.assertEqual(x.size, x_corr.size)
        max_lag = sms_dsp.peak_detect_interp(x_corr)[0][0]
        max_f = fs / max_lag
        self.assert_almost_equal_significant(f, max_f, places=1)

  def test_detrend_shape_error(self):
    """Test shape error in detrend"""
    for a in (np.empty(()), np.empty((2, 2))):
      with self.subTest(shape=a.shape):
        with self.assertRaises(ValueError):
          dsp_utils.detrend(a)

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

  def test_zero_reach(self):
    """Test reaching zero in :func:`dsp_utils.dychotomic_zero_crossing`"""

    def func(a: float):
      if abs(a) > 1:
        return a
      return 0

    a0 = dsp_utils.dychotomic_zero_crossing(func, 2, -2)
    self.assertEqual(func(a0), 0)

  def test_strided_nwindows(self):
    """Test coherence of strided convolution output size"""
    input_sizes = 1 + np.round(np.random.rand(32) * 1023).astype(int)
    wsize_f = np.random.rand(32)
    for input_size in input_sizes:
      for wsize in 1 + np.round(wsize_f * (input_size - 1)).astype(int):
        output_size = dsp_utils.n_windows(input_size=input_size, wsize=wsize)
        with self.subTest(input_size=input_size, wsize=wsize):
          self.assertLessEqual(output_size * wsize, input_size)
          self.assertGreater((output_size + 1) * wsize, input_size)

  def test_strided_convolution(self, n: int = 8, strides=(4, 8, 32)):
    """Test strided convolution"""
    np.random.seed(42)
    xs = np.random.randn(n, 1024)
    ks = np.empty((n, 8), dtype=np.complex64)
    ks.real, ks.imag = np.random.randn(2, *ks.shape)

    for i, (x, k) in enumerate(itertools.product(xs, ks)):
      x_c = signal.convolve(x, k, mode="valid")
      for stride in strides:
        x_cs = x_c[::stride]
        x_s = dsp_utils.strided_convolution_complex_kernel(x, k, stride=stride)
        with self.subTest(i=i, what="shape"):
          self.assertEqual(x_cs.shape, x_s.shape)
        with self.subTest(i=i, what="value"):
          self.assert_almost_equal_rmse(x_cs, x_s)
