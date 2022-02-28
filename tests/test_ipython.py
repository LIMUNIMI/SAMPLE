"""Test IPython utilities"""
import unittest

from chromatictools import unittestmixins
from sample import ipython
from sample.evaluation import random


class TestIPython(unittestmixins.AssertDoesntRaiseMixin, unittest.TestCase):
  """Test IPython utilities"""

  def test_webaudio(self):
    """Test WebAudio widget"""
    with self.assert_doesnt_raise():
      x, fs, _ = random.BeatsGenerator(seed=1234).audio()
      ipython.WebAudio(x=x, rate=fs)
