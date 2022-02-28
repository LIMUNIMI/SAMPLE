"""Tests for evaluation"""
import unittest
import multiprocessing as mp

import numpy as np
from chromatictools import unittestmixins
from sample.evaluation import metrics, random


class TestMetrics(unittestmixins.AssertDoesntRaiseMixin,
                  unittestmixins.RMSEAssertMixin, unittest.TestCase):
  """Tests for evaluation metrics"""

  def setUp(self) -> None:
    """Setup test audio"""
    self.x, _, _ = random.BeatsGenerator(seed=1234).audio()

  def test_multiscale_zero(self):
    """Test multiscale spectral loss is zero for x == y"""
    self.assertEqual(0, metrics.multiscale_spectral_loss(self.x, self.x))

  def test_multiscale_zero_parallel(self):
    """Test multiscale spectral loss (in parallel) is zero for x == y"""
    self.assertEqual(0, metrics.multiscale_spectral_loss(self.x,
                                                         self.x,
                                                         njobs=6))

  def test_multiscale_coherence(self, dropout=(0, 0.20, 0.40, 0.60)):
    """Test coherence of multiscale spectral loss"""
    dropout = sorted(dropout)
    prev = prev_d = None
    np.random.seed(42)
    r = np.random.rand(np.size(self.x))
    with mp.Pool(processes=6) as pool:
      for d in dropout:
        y = np.greater(r, d).astype(float) * self.x
        curr = metrics.multiscale_spectral_loss(self.x, y, pool=pool)
        if d == 0:
          with self.subTest(test="zero"):
            self.assertEqual(0, curr)
        else:
          if prev is not None:
            with self.subTest(test="order", prev=prev_d, curr=d):
              self.assertLess(prev, curr)
        prev = curr
        prev_d = d
