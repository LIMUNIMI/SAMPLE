"""Tests random audio generation"""
import unittest

import numpy as np
from sample.evaluation import random


class TestRandom(unittest.TestCase):
  """Tests random audio generator"""

  def setUp(self, seed: int = 12345):
    """Initialize generator"""
    self.rng = random.BeatsGenerator(seed=seed)

  def test_beta_twosides(self, subcases: int = 256):
    """Test twosided beta distribution sampling"""
    for n, a, b, l, r in zip(
        self.rng.rng.integers(1, 1024, size=subcases),
        self.rng.rng.uniform(1, 16, size=subcases),
        self.rng.rng.uniform(1, 16, size=subcases),
        self.rng.rng.uniform(0, 128, size=subcases),
        self.rng.rng.uniform(0, 128, size=subcases),
    ):
      r += l
      with self.subTest(a=a, b=b, n=n, l=l, r=r):
        x = np.array(self.rng.beta_twosides(a=a, b=b, left=l, right=r,
                                            size=n)).flatten()
        with self.subTest(what="size"):
          self.assertEqual(np.size(x), n)
        with self.subTest(what="bounds"):
          for i in map(abs, x):
            self.assertGreaterEqual(i, l)
            self.assertGreaterEqual(r, i)

  def test_freqs(self, subcases: int = 256):
    """Test frequency sampling"""
    for n in self.rng.rng.integers(1, 1024, size=subcases):
      with self.subTest(n=n):
        x = np.array(self.rng.freqs(size=n)).flatten()  # pylint: disable=E1123
        with self.subTest(what="shape"):
          self.assertEqual(np.size(x), 3 * n)
        with self.subTest(what="bounds"):
          for i in x:
            self.assertGreaterEqual(i, self.rng.f_min)
            self.assertGreaterEqual(self.rng.f_max, i)

  def test_decays(self, subcases: int = 256):
    """Test decay sampling"""
    for n in self.rng.rng.integers(1, 1024, size=subcases):
      with self.subTest(n=n):
        x = np.array(self.rng.decays(size=n)).flatten()  # pylint: disable=E1123
        with self.subTest(what="size"):
          self.assertEqual(np.size(x), 3 * n)
        with self.subTest(what="bounds"):
          for i in x:
            self.assertGreater(i, 0)

  def test_amps(self, subcases: int = 256):
    """Test amplitude sampling"""
    for n in self.rng.rng.integers(1, 1024, size=subcases):
      with self.subTest(n=n):
        x = np.array(self.rng.amps(size=n)).flatten()  # pylint: disable=E1123
        with self.subTest(what="size"):
          self.assertEqual(np.size(x), 3 * n)
        with self.subTest(what="sum"):
          for i in np.reshape(x, newshape=(n, 3)):
            self.assertAlmostEqual(np.sum(i), self.rng.sine_amp)

  def test_snr(self, subcases: int = 256, snr_delta: float = 180):
    """Test snr"""
    for snr in map(lambda i: snr_delta * (i / (subcases - 1) - 0.5),
                   range(subcases)):
      with self.subTest(snr=snr):
        g = random.BeatsGenerator(snr=snr)
        self.assertAlmostEqual(snr, 20 * np.log10(g.sine_amp / g.noise_amp))
