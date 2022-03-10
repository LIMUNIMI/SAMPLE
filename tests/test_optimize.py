"""Tests for optimzation routines"""
import itertools
import unittest

from sample import optimize


class TestRemapper(unittest.TestCase):
  """Tests for argument remapper"""

  def test_passthrough(self):
    """Test that provided targets pass through"""
    kwargs = dict(
        sinusoidal_model__n=128,
        sinusoidal_model__w=list(itertools.repeat(1 / 128, 128)),
        sinusoidal_model__h=64,
    )
    self.assertEqual(kwargs, optimize.sample_kwargs_remapper(**kwargs))

  def test_wsize(self,
                 fft_log_ns=(7, 8, 11),
                 w_sizes=(0.25, 0.5, 0.75),
                 w_types=("hamming", "blackman")):
    """Test that window size is correct"""
    for log_n, ws, wt in itertools.product(fft_log_ns, w_sizes, w_types):
      kwargs = optimize.sample_kwargs_remapper(
          sinusoidal_model__log_n=log_n,
          sinusoidal_model__wsize=ws,
          sinusoidal_model__wtype=wt,
      )
      with self.subTest(log_n=log_n, ws=ws, wt=wt):
        self.assertEqual(kwargs["sinusoidal_model__w"].size,
                         int((1 << log_n) * ws))

  def test_hsize(
      self,
      fft_log_ns=(7, 8, 11),
      w_sizes=(0.25, 0.5, 0.75),
      olaps=(0.25, 0.5, 0.75),
  ):
    """Test that hop size is correct"""
    for log_n, ws, olap in itertools.product(fft_log_ns, w_sizes, olaps):
      kwargs = optimize.sample_kwargs_remapper(
          sinusoidal_model__log_n=log_n,
          sinusoidal_model__wsize=ws,
          sinusoidal_model__overlap=olap,
      )
      with self.subTest(log_n=log_n, ws=ws, olap=olap):
        self.assertAlmostEqual(
            olap, 1 -
            kwargs["sinusoidal_model__h"] / kwargs["sinusoidal_model__w"].size)
