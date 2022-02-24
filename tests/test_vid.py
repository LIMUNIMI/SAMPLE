"""Test logo generation"""
import io
import unittest

from chromatictools import unittestmixins
from sample import vid


class TestInit(unittestmixins.AssertDoesntRaiseMixin, unittest.TestCase):
  """Tests for logo generation"""

  def test_splash(self):
    """Test full-screen logo"""
    with self.assert_doesnt_raise():
      with io.BytesIO() as buf:
        vid.logo_plt_fn(buf)

  def test_icon(self):
    """Test icon logo"""
    with self.assert_doesnt_raise():
      with io.BytesIO() as buf:
        vid.icon_plt_fn(buf)
