"""Tests for psychoacoustic models"""
import unittest
from chromatictools import unittestmixins
from sample import psycho
import numpy as np


class TestPsycho(unittestmixins.AssertDoesntRaiseMixin,
                 unittestmixins.RMSEAssertMixin, unittest.TestCase):
  """Tests psychoacoustic models functions"""

  def test_bark(self):
    f = np.linspace(0, 2e4, 1024)
    modes = ("unsupported", "zwicker", "traunmuller", "wang")
    no_forward = ("unsupported",)
    no_backward = ("zwicker",)
    for m in modes:
      with self.subTest(mode=m, conversion="forward"):
        if m in no_forward:
          with self.assertRaises(ValueError):
            psycho.hz2bark(f, mode=m)
        else:
          with self.assert_doesnt_raise():
            b = psycho.hz2bark(f, mode=m)
      with self.subTest(mode=m, conversion="backward"):
        if m in no_forward:
          pass
        elif m in no_backward:
          with self.assertRaises(ValueError):
            psycho.bark2hz(b, mode=m)
        else:
          f_ = psycho.bark2hz(b, mode=m)
          self.assert_almost_equal_rmse(f, f_)
