"""Tests for top-level module"""
import unittest

import sample
from chromatictools import unittestmixins

import io


class TestInit(unittestmixins.AssertPrintsMixin, unittest.TestCase):
  """Tests for __init__"""

  def test_main(self):
    """Test execution as module"""
    with self.assert_prints(sample.__doc__ + "\n\n" + "Version: " +
                            sample.__version__ + "\n"):
      sample()

  def test_main_logo(self):
    """Test execution as module, and logo print"""
    with self.assert_prints(sample.__doc__ + "\n\n" + "Version: " +
                            sample.__version__ + "\n"):
      sample(logo=dict())
