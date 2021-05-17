"""Test logo generation"""
import unittest
from sample import vid
from chromatictools import unittestmixins
import io


class TestInit(
  unittestmixins.AssertDoesntRaiseMixin,
  unittest.TestCase
):
  """Tests for logo generation"""
  def test_splash(self):
    """Test full-screen logo"""
    with self.assert_doesnt_raise():
      with io.BytesIO() as buf:
        vid.logo(buf)

  def test_icon(self):
    """Test icon logo"""
    with self.assert_doesnt_raise():
      with io.BytesIO() as buf:
        vid.logo(buf, icon=True)
