"""Tests for top-level module"""
import unittest
from chromatictools import unittestmixins
import sample


class TestInit(unittestmixins.AssertPrintsMixin, unittest.TestCase):
  """Tests for __init__"""
  def test_main(self):
    """Test execution as module"""
    with self.assert_prints(
      sample.__doc__ + "\n\n" + "Version: " + sample.__version__ + "\n"
    ):
      sample.main()
